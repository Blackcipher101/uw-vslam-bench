"""Stage 1: keyframe selection, stereo rectification, label-frame export.

Writes <out>/keyframes/kf_NNNN_left.jpg, _right.jpg (rectified), keyframes.json, rect.json and
<out>/label_frames/kf_NNNN.jpg (small, for hand labelling).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import geometry as G
from ir_common import (StageTimer, decode_image, interp_pose, load_camchain, load_tum, pair_stamps,
                       path_length_m, raw_messages, select_keyframes)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--camchain", required=True)
    ap.add_argument("--traj", default="none", help="TUM file (left camera pose in world) or 'none'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-dt", type=float, default=2.0, help="min seconds between keyframes")
    ap.add_argument("--max-dt", type=float, default=4.0, help="seconds after which a keyframe is forced")
    ap.add_argument("--min-dist", type=float, default=0.3, help="metres travelled that trigger a keyframe")
    ap.add_argument("--max-keyframes", type=int, default=400)
    ap.add_argument("--n-label", type=int, default=40, help="evenly spaced frames exported for labelling")
    a = ap.parse_args(argv)

    out = Path(a.out)
    (out / "keyframes").mkdir(parents=True, exist_ok=True)
    (out / "label_frames").mkdir(exist_ok=True)
    timer = StageTimer(out)
    cc = load_camchain(a.camchain)
    topic_l, topic_r = cc["cam0"]["topic"], cc["cam1"]["topic"]
    tum = None if a.traj == "none" else load_tum(a.traj)

    # pass 1: stamps only
    with timer.time("scan_stamps"):
        stamps = {topic_l: [], topic_r: []}
        for topic, st, _, _ in raw_messages(a.bag, [topic_l, topic_r]):
            stamps[topic].append(st)
    sl, sr = stamps[topic_l], stamps[topic_r]
    pairs = pair_stamps(sl, sr)
    print(f"[keyframes] left={len(sl)} right={len(sr)} paired={len(pairs)}", flush=True)
    if not pairs:
        sys.exit("no stereo pairs found (check topics / stamps)")
    pst = [sl[i] for i, _ in pairs]
    poses = [interp_pose(tum, t) if tum is not None else None for t in pst]
    if tum is not None:
        cov = sum(p is not None for p in poses) / len(poses)
        print(f"[keyframes] pose coverage of paired frames: {cov:.1%}", flush=True)
        if cov < 0.5:
            print("[keyframes] WARNING: <50% frames have a pose; trajectory and bag clocks may differ", flush=True)
    sel = select_keyframes(pst, [None if p is None else p[:3, 3] for p in poses], a.min_dt, a.max_dt, a.min_dist)
    if len(sel) > a.max_keyframes:
        sel = [sel[int(k)] for k in np.linspace(0, len(sel) - 1, a.max_keyframes)]
    print(f"[keyframes] selected {len(sel)} keyframes", flush=True)

    # pass 2: decode selected
    want_l = {round(sl[pairs[k][0]], 6): n for n, k in enumerate(sel)}
    want_r = {round(sr[pairs[k][1]], 6): n for n, k in enumerate(sel)}
    imgs: dict[int, dict] = {}  # raw CDR blobs only (decoded one keyframe at a time to bound RAM)
    with timer.time("read_pass"):
        for topic, st, data, typ in raw_messages(a.bag, [topic_l, topic_r]):
            key = round(st, 6)
            side, table = ("l", want_l) if topic == topic_l else ("r", want_r)
            if key in table:
                imgs.setdefault(table[key], {})[side] = (data, typ)
    rc = None
    kfs = []
    n_label = min(a.n_label, len(sel))
    label_idx = set(int(round(x)) for x in np.linspace(0, len(sel) - 1, n_label)) if n_label else set()
    for n, k in enumerate(sel):
        d = imgs.get(n, {})
        if "l" not in d or "r" not in d:
            print(f"[keyframes] skip kf {n}: image missing", flush=True)
            continue
        with timer.time("decode_frame"):
            d = {s: decode_image(*d[s]) for s in ("l", "r")}
        if rc is None:
            h, w = d["l"].shape[:2]
            rc = G.build_rectifier(cc, (w, h))
            print(f"[keyframes] baseline={rc.baseline_m:.4f} m  f_rect={rc.P1[0,0]:.1f}px", flush=True)
        with timer.time("rectify"):
            rl, rr = G.rectify(rc, d["l"], d["r"])
        kid = f"kf_{n:04d}"
        cv2.imwrite(str(out / "keyframes" / f"{kid}_left.jpg"), rl, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out / "keyframes" / f"{kid}_right.jpg"), rr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if n in label_idx:
            s = 800 / rl.shape[1]
            cv2.imwrite(str(out / "label_frames" / f"{kid}.jpg"), cv2.resize(rl, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
        T = poses[k]
        kfs.append({
            "id": kid, "t": pst[k], "pose_world_cam0": None if T is None else T.tolist(),
            "T_world_rect": None if T is None else G.T_world_rect(T, rc.R1).tolist(),
            "left": f"keyframes/{kid}_left.jpg", "right": f"keyframes/{kid}_right.jpg",
        })
    json.dump({"bag": str(a.bag), "traj": a.traj, "frames": kfs}, open(out / "keyframes.json", "w"))
    json.dump({"P1": rc.P1.tolist(), "R1": rc.R1.tolist(), "baseline_m": rc.baseline_m, "size": list(rc.size)},
              open(out / "rect.json", "w"))
    cov = {
        "n_stereo_pairs": len(pairs), "n_keyframes": len(kfs),
        "n_keyframes_with_pose": sum(f["pose_world_cam0"] is not None for f in kfs),
        "t_start": pst[0], "t_end": pst[-1], "duration_s": pst[-1] - pst[0],
        "track_length_m": None if tum is None else path_length_m(tum, pst[0], pst[-1]),
        "traj_file": a.traj,
    }
    json.dump(cov, open(out / "coverage.json", "w"), indent=1)
    print(f"[keyframes] done: {cov}", flush=True)


if __name__ == "__main__":
    main()
