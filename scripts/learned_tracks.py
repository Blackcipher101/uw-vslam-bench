"""Precompute learned stereo feature tracks (ALIKED or SuperPoint + LightGlue) for uw_ct_vio.

Raw images in (the same internal CLAHE as uw_ct_vio's KLT tracker, no enhancement). Per stereo pair:
  1. extract features in the left and right image;
  2. temporal: match left(t-1) -> left(t), keep essential-matrix RANSAC inliers, carry track ids;
  3. re-association: features still without an id are matched against the last few keyframes and,
     if geometrically consistent, take back the id of a track that was lost for a few frames;
  4. stereo: match left -> right, keep matches within the epipolar band (fisheye or radtan aware).
Pixels are undistorted to normalised coordinates (x/z, y/z), which uw_ct_vio consumes directly.

Output (little-endian): header b"UWTRK1\\0\\0", uint32 version=1, uint32 reserved; per frame float64 t
(left header stamp, s), uint32 n; per track uint64 id, float32 lx ly rx ry, uint8 has_right.

Usage (lightglue venv):
  python scripts/learned_tracks.py <bag> <out.uwtrk> --intrinsics C --imucam C --left cam1 --right cam0 \
      --left-topic T --right-topic T [--features aliked] [--resize 0.5]
"""
from __future__ import annotations

import argparse
import struct
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from lightglue import ALIKED, LightGlue, SuperPoint
from rosbags.highlevel import AnyReader

from dehaze import load_rig
from dehaze_bag import pairs
from probe_enhance import decode

MAGIC, VERSION = b"UWTRK1\0\0", 1
REC = np.dtype([("id", "<u8"), ("lx", "<f4"), ("ly", "<f4"), ("rx", "<f4"), ("ry", "<f4"), ("has_right", "u1")])


def undistort(cam, pts: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return np.zeros((0, 2))
    p = pts.reshape(-1, 1, 2).astype(np.float64)
    f = cv2.fisheye.undistortPoints if cam.model == "equidistant" else cv2.undistortPoints
    return f(p, cam.K, cam.D).reshape(-1, 2)


def scaled(cam, s: float):
    if s == 1.0:
        return cam
    K = cam.K.copy()
    K[:2] *= s
    return type(cam)(cam.model, K, cam.D, (round(cam.size[0] * s), round(cam.size[1] * s)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--intrinsics", type=Path, required=True)
    ap.add_argument("--imucam", type=Path, required=True)
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--left-topic", required=True)
    ap.add_argument("--right-topic", required=True)
    ap.add_argument("--features", choices=("aliked", "superpoint"), default="aliked")
    ap.add_argument("--max-kp", type=int, default=512)
    ap.add_argument("--resize", type=float, default=1.0)
    ap.add_argument("--clahe-clip", type=float, default=1.0)
    ap.add_argument("--clahe-tiles", type=int, default=2)
    ap.add_argument("--ransac-px", type=float, default=1.0)
    ap.add_argument("--epipolar-px", type=float, default=2.0)
    ap.add_argument("--keyframe-every", type=int, default=3)
    ap.add_argument("--reassoc-keyframes", type=int, default=3, help="0 disables re-association")
    a = ap.parse_args()

    # Keep CPU threads modest: the GPU does the heavy work, and other runs share this machine.
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    camL, camR, R, t = load_rig(a.intrinsics, a.imucam, a.left, a.right)
    camL, camR = scaled(camL, a.resize), scaled(camR, a.resize)
    E = np.array([[0, -t[2, 0], t[1, 0]], [t[2, 0], 0, -t[0, 0]], [-t[1, 0], t[0, 0], 0]]) @ R  # x_r^T E x_l = 0
    f_px = float(camL.K[0, 0])
    clahe = cv2.createCLAHE(a.clahe_clip, (a.clahe_tiles, a.clahe_tiles))
    dev = torch.device("cuda")
    ext = (ALIKED(max_num_keypoints=a.max_kp) if a.features == "aliked" else SuperPoint(max_num_keypoints=a.max_kp)).eval().to(dev)
    lg = LightGlue(features=a.features).eval().to(dev)

    def feats(bgr: np.ndarray) -> dict:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if a.resize != 1.0:
            g = cv2.resize(g, None, fx=a.resize, fy=a.resize, interpolation=cv2.INTER_AREA)
        g = clahe.apply(g)
        img = torch.from_numpy(g).to(dev, torch.float32)[None, None] / 255.0
        if a.features == "aliked":
            img = img.expand(1, 3, -1, -1)  # ALIKED expects RGB
        return ext.extract(img)

    def match(fa: dict, fb: dict) -> tuple[np.ndarray, np.ndarray]:
        m = lg({"image0": fa, "image1": fb})["matches"][0].cpu().numpy()
        return m[:, 0], m[:, 1]

    def ransac(na: np.ndarray, nb: np.ndarray) -> np.ndarray:
        if len(na) < 8:
            return np.zeros(len(na), bool)
        _, mask = cv2.findEssentialMat(na, nb, np.eye(3), cv2.RANSAC, 0.999, a.ransac_px / f_px)
        return mask.ravel().astype(bool) if mask is not None else np.zeros(len(na), bool)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    prev = None                      # (feats, normalised pts, ids)
    keyframes: deque = deque(maxlen=max(a.reassoc_keyframes, 1))
    next_id, n, tic = 1, 0, time.perf_counter()
    stats = {"kp": [], "tracked": [], "reassoc": [], "stereo": []}
    with AnyReader([a.bag]) as rd, open(a.out, "wb") as fo, torch.inference_mode():
        fo.write(MAGIC + struct.pack("<II", VERSION, 0))
        for kind, *rest in pairs(rd, a.left_topic, a.right_topic, set()):
            if kind != "pair":
                continue
            (_, _, lm), (_, _, rm) = rest
            stamp = lm.header.stamp.sec + lm.header.stamp.nanosec * 1e-9
            fl, fr = feats(decode(lm)), feats(decode(rm))
            kl = fl["keypoints"][0].cpu().numpy()
            nl = undistort(camL, kl)
            ids = np.zeros(len(kl), np.uint64)
            if prev is not None and len(kl):
                i0, i1 = match(prev[0], fl)
                ok = ransac(prev[1][i0], nl[i1])
                ids[i1[ok]] = prev[2][i0[ok]]
            n_tracked = int((ids > 0).sum())
            n_re = 0
            if a.reassoc_keyframes > 0 and len(kl):
                for kf in reversed(keyframes):  # newest first
                    free = ids == 0
                    if free.sum() < 8:
                        break
                    i0, i1 = match(kf[0], fl)
                    sel = free[i1] & ~np.isin(kf[2][i0], ids)  # unassigned here, id not already present
                    if sel.sum() < 8:
                        continue
                    i0, i1 = i0[sel], i1[sel]
                    ok = ransac(kf[1][i0], nl[i1])
                    ids[i1[ok]] = kf[2][i0[ok]]
                    n_re += int(ok.sum())
            fresh = ids == 0
            ids[fresh] = np.arange(next_id, next_id + fresh.sum(), dtype=np.uint64)
            next_id += int(fresh.sum())

            rec = np.zeros(len(kl), REC)
            rec["id"], rec["lx"], rec["ly"] = ids, nl[:, 0], nl[:, 1]
            n_st = 0
            if len(kl) and fr["keypoints"].shape[1]:
                j0, j1 = match(fl, fr)
                if len(j0):
                    kr = fr["keypoints"][0].cpu().numpy()
                    nr = undistort(camR, kr[j1])
                    xl = np.hstack([nl[j0], np.ones((len(j0), 1))])
                    xr = np.hstack([nr, np.ones((len(j0), 1))])
                    line = xl @ E.T
                    d = np.abs(np.sum(xr * line, 1)) / np.hypot(line[:, 0], line[:, 1])
                    ok = d < a.epipolar_px / f_px
                    rec["rx"][j0[ok]], rec["ry"][j0[ok]] = nr[ok, 0], nr[ok, 1]
                    rec["has_right"][j0[ok]] = 1
                    n_st = int(ok.sum())
            fo.write(struct.pack("<dI", stamp, len(rec)))
            fo.write(rec.tobytes())
            prev = (fl, nl, ids)
            if n % a.keyframe_every == 0:
                keyframes.append((fl, nl, ids))
            n += 1
            for k, v in (("kp", len(kl)), ("tracked", n_tracked), ("reassoc", n_re), ("stereo", n_st)):
                stats[k].append(v)
            if n % 200 == 0:
                med = {k: np.median(v[-200:]) for k, v in stats.items()}
                print(f"{n} frames | last200 median kp {med['kp']:.0f} tracked {med['tracked']:.0f} "
                      f"reassoc {med['reassoc']:.0f} stereo {med['stereo']:.0f} | {n / (time.perf_counter() - tic):.1f} fps",
                      flush=True)
    print(f"done: {n} frames, {next_id - 1} tracks -> {a.out}")
    for k, v in stats.items():
        v = np.array(v)
        print(f"  {k:8s} median {np.median(v):6.0f}  p10 {np.percentile(v, 10):6.0f}  sum {v.sum()}")


if __name__ == "__main__":
    main()
