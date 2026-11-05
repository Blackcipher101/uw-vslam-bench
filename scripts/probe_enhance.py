"""Does image enhancement make features easier to track? Same pipeline on any stereo bag.

Pipeline per frame (identical for every dataset and mode):
  decode (colour or mono; mono is copied to 3 channels) -> Enhancer(mode) -> greyscale -> resize
  -> CLAHE (SVIn's own preprocessing, unchanged)
Per frame it reports, for ORB (binary corners + descriptors; stand-in for SVIn's BRISK, which
OpenCV 5 no longer ships) and SuperPoint+LightGlue:
  temporal inliers  left(t-1) <-> left(t) matches surviving fundamental-matrix RANSAC
  stereo inliers    left(t) <-> right(t) matches surviving fundamental-matrix RANSAC
F-RANSAC on raw pixels is distortion-agnostic enough to compare modes on the same frames.

Usage (lightglue venv; rosbags reads ROS 2 bags without ROS):
  ~/.venvs/lightglue/bin/python scripts/probe_enhance.py <bag> --left T --right T \
      --windows 0:12 160:172 --modes none funie-gain funie-native --scale 0.5 --out results.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from rosbags.highlevel import AnyReader

from enhance import Enhancer


PAIR_TOL_S = 0.005
F_RANSAC_PX = 2.0
ORB_FEATURES = 1000
RATIO = 0.8


def decode(msg) -> np.ndarray:
    """BGR uint8 from sensor_msgs/Image (mono8/bgr8/rgb8) or CompressedImage."""
    if hasattr(msg, "format"):
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("failed to decode compressed image")
        return img
    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    if msg.encoding == "mono8":
        return cv2.cvtColor(np.ascontiguousarray(a[:, :msg.width]), cv2.COLOR_GRAY2BGR)
    img = a[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == "rgb8" else img.copy()


def stereo_frames(bag: Path, left: str, right: str, windows: list[tuple[float, float]]):
    """Yield (window_index, t_rel_s, left_bgr, right_bgr) for pairs inside the windows."""
    with AnyReader([bag]) as r:
        conns = [c for c in r.connections if c.topic in (left, right)]
        if len({c.topic for c in conns}) != 2:
            raise SystemExit(f"bag lacks {left} or {right}")
        pend: dict[str, tuple[float, object]] = {}
        t0 = None
        for conn, _, raw in r.messages(connections=conns):
            m = r.deserialize(raw, conn.msgtype)
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            t0 = t if t0 is None else t0
            rel = t - t0
            if rel > max(w[1] for w in windows):
                return
            wi = [i for i, w in enumerate(windows) if w[0] <= rel < w[1]]
            if not wi:
                continue
            side, other = ("l", "r") if conn.topic == left else ("r", "l")
            if other in pend and abs(pend[other][0] - t) < PAIR_TOL_S:
                o = pend.pop(other)[1]
                lm, rm = (m, o) if side == "l" else (o, m)
                yield wi[0], rel, decode(lm), decode(rm)
            else:
                pend[side] = (t, m)


def f_inliers(p0: np.ndarray, p1: np.ndarray) -> int:
    if len(p0) < 8:
        return 0
    _, mask = cv2.findFundamentalMat(p0, p1, cv2.FM_RANSAC, F_RANSAC_PX, 0.999)
    return int(mask.sum()) if mask is not None else 0


def orb_match(da, db, ka, kb) -> tuple[np.ndarray, np.ndarray]:
    if da is None or db is None or len(ka) < 2 or len(kb) < 2:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < RATIO * m[1].distance]
    return (np.float32([ka[m.queryIdx].pt for m in good]), np.float32([kb[m.trainIdx].pt for m in good]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path)
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--windows", nargs="+", required=True, help="start:end in seconds from bag start")
    ap.add_argument("--modes", nargs="+", default=["none", "funie-gain", "funie-native"])
    ap.add_argument("--scale", type=float, default=1.0, help="resize before CLAHE (SVIn resizeFactor)")
    ap.add_argument("--clahe-clip", type=float, default=1.0)
    ap.add_argument("--clahe-tiles", type=int, default=2)
    ap.add_argument("--max-kp", type=int, default=512)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save-examples", type=Path, help="write one side-by-side PNG per window here")
    ap.add_argument("--enhance-device", default="cuda", help="cpu if the GPU's cuDNN is unusable")
    a = ap.parse_args()
    windows = [tuple(float(x) for x in w.split(":")) for w in a.windows]

    dev = torch.device("cuda")
    sp = SuperPoint(max_num_keypoints=a.max_kp).eval().to(dev)
    lg = LightGlue(features="superpoint").eval().to(dev)
    orb = cv2.ORB_create(ORB_FEATURES)  # stand-in for BRISK (see module docstring)
    clahe = cv2.createCLAHE(a.clahe_clip, (a.clahe_tiles, a.clahe_tiles))
    enhancers = {m: Enhancer(m, a.enhance_device) for m in a.modes}

    def prep(bgr: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
        e = enhancers[mode](bgr)
        g = cv2.cvtColor(e, cv2.COLOR_BGR2GRAY)
        if a.scale != 1.0:
            g = cv2.resize(g, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_LINEAR)
        return clahe.apply(g), e

    def sp_feats(g: np.ndarray) -> dict:
        return sp.extract(torch.from_numpy(g).to(dev, torch.float32)[None, None] / 255.0)

    def lg_pts(fa: dict, fb: dict) -> tuple[np.ndarray, np.ndarray]:
        mm = lg({"image0": fa, "image1": fb})["matches"][0].cpu().numpy()
        return (fa["keypoints"][0].cpu().numpy()[mm[:, 0]], fb["keypoints"][0].cpu().numpy()[mm[:, 1]])

    rows = {m: [] for m in a.modes}
    prev: dict[str, dict] = {}
    last_w, examples_done, tic = None, set(), time.perf_counter()
    with torch.inference_mode():
        for wi, rel, lbgr, rbgr in stereo_frames(a.bag, a.left, a.right, windows):
            if wi != last_w:
                prev, last_w = {}, wi
            for mode in a.modes:
                gl, el = prep(lbgr, mode)
                gr, _ = prep(rbgr, mode)
                fl, fr = sp_feats(gl), sp_feats(gr)
                kl, dl = orb.detectAndCompute(gl, None)
                kr, dr = orb.detectAndCompute(gr, None)
                row = {"w": wi, "t": round(rel, 3), "sp_kp": int(fl["keypoints"].shape[1]), "orb_kp": len(kl),
                       "sp_stereo": f_inliers(*lg_pts(fl, fr)), "orb_stereo": f_inliers(*orb_match(dl, dr, kl, kr)),
                       "sp_temporal": 0, "orb_temporal": 0}
                if mode in prev:
                    p = prev[mode]
                    row["sp_temporal"] = f_inliers(*lg_pts(p["sp"], fl))
                    row["orb_temporal"] = f_inliers(*orb_match(p["bd"], dl, p["bk"], kl))
                    rows[mode].append(row)
                prev[mode] = {"sp": fl, "bk": kl, "bd": dl}
                if a.save_examples and (wi, mode) not in examples_done:
                    a.save_examples.mkdir(parents=True, exist_ok=True)
                    small = cv2.resize(el, None, fx=0.5, fy=0.5)
                    cv2.imwrite(str(a.save_examples / f"w{wi}_{mode}_enh.jpg"), small)
                    cv2.imwrite(str(a.save_examples / f"w{wi}_{mode}_svin_input.jpg"), gl)
                    examples_done.add((wi, mode))
            n = len(rows[a.modes[0]])
            if n and n % 50 == 0:
                msg = " | ".join(f"{m}: orb_t {np.median([r['orb_temporal'] for r in rows[m]]):.0f} "
                                 f"sp_t {np.median([r['sp_temporal'] for r in rows[m]]):.0f}" for m in a.modes)
                print(f"{n} frames ({time.perf_counter() - tic:.0f}s) {msg}", flush=True)

    summary = {}
    for mode in a.modes:
        r = rows[mode]
        summary[mode] = {k: {"median": float(np.median([x[k] for x in r])), "p10": float(np.percentile([x[k] for x in r], 10))}
                         for k in ("orb_kp", "orb_temporal", "orb_stereo", "sp_kp", "sp_temporal", "sp_stereo")}
        summary[mode]["frames"] = len(r)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"bag": str(a.bag), "windows": windows, "summary": summary, "rows": rows}, indent=1))
    print("== summary (median / p10 per frame)")
    keys = ("orb_temporal", "orb_stereo", "sp_temporal", "sp_stereo")
    print(f"{'mode':14s} " + " ".join(f"{k:>16s}" for k in keys))
    for mode in a.modes:
        s = summary[mode]
        print(f"{mode:14s} " + " ".join(f"{s[k]['median']:8.0f} /{s[k]['p10']:5.0f}" for k in keys))


if __name__ == "__main__":
    main()
