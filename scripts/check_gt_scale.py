"""Check the metric scale of a ground-truth trajectory against stereo geometry alone.

For frame pairs (t, t+dt) of a track file from lightglue_tracks.py, the stereo-triangulated
points of tracks seen in both frames give the camera motion by a rigid 3D-3D fit (RANSAC, then
Umeyama on inliers): metric from the stereo baseline only, independent of IMU and ground truth.
The ratio |t_gt| / |t_stereo| over many pairs is the ground truth's scale relative to the stereo
rig; ~1 means the ground truth is metric, the evaluator's fitted Sim3 scale is then the VIO's.

Usage: python3 scripts/check_gt_scale.py <tracks.svtrk> <gt.tum> [--dt-s 1.0]
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

from evaluate import associate, load_tum, to_mats
from geometry import umeyama

TRACK = np.dtype([("id", "<u8"), ("u0", "<f4"), ("v0", "<f4"), ("u1", "<f4"), ("v1", "<f4"),
                  ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("flags", "u1")])
HAS_POINT = 2


def read_frames(path: Path, max_depth_m: float) -> list[tuple[float, dict[int, np.ndarray]]]:
    data = path.read_bytes()
    if data[:8] != b"SVTRK1\0\0":
        raise SystemExit(f"{path} is not a track file")
    off, frames = 16, []
    while off < len(data):
        t_ns, n = struct.unpack_from("<qI", data, off)
        off += 12
        rec = np.frombuffer(data, TRACK, n, off)
        off += n * TRACK.itemsize
        ok = (rec["flags"] & HAS_POINT).astype(bool) & (rec["z"] > 0.2) & (rec["z"] < max_depth_m)
        pts = np.stack([rec["x"][ok], rec["y"][ok], rec["z"][ok]], 1).astype(np.float64)
        frames.append((t_ns / 1e9, dict(zip(rec["id"][ok].tolist(), pts))))
    return frames


def rigid_ransac(a: np.ndarray, b: np.ndarray, thresh_m: float, iters: int, rng) -> np.ndarray | None:
    """Inlier mask of the best rigid b = R a + t, from 3-point samples."""
    best = None
    for _ in range(iters):
        idx = rng.choice(len(a), 3, replace=False)
        _, r, t = umeyama(a[idx], b[idx], with_scale=False)
        inl = np.linalg.norm((r @ a.T).T + t - b, axis=1) < thresh_m
        if best is None or inl.sum() > best.sum():
            best = inl
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tracks", type=Path)
    ap.add_argument("gt", type=Path)
    ap.add_argument("--dt-s", type=float, default=1.0, help="time between the two frames of a pair")
    ap.add_argument("--max-depth-m", type=float, default=6.0, help="stereo depth error grows with z^2")
    ap.add_argument("--min-common", type=int, default=30)
    ap.add_argument("--min-motion-m", type=float, default=0.15, help="skip near-static pairs")
    ap.add_argument("--inlier-m", type=float, default=0.05)
    a = ap.parse_args()

    frames = read_frames(a.tracks, a.max_depth_m)
    gt = load_tum(a.gt)
    times = np.array([f[0] for f in frames])
    ig, ie = associate(gt[:, 0], times)
    gt_T = dict(zip(ie.tolist(), to_mats(gt[ig])))
    rng = np.random.default_rng(0)

    ratios, used = [], 0
    for i in range(len(frames)):
        j = int(np.searchsorted(times, times[i] + a.dt_s))
        if j >= len(frames) or i not in gt_T or j not in gt_T:
            continue
        common = frames[i][1].keys() & frames[j][1].keys()
        if len(common) < a.min_common:
            continue
        ids = list(common)
        pa = np.array([frames[i][1][k] for k in ids])  # points in cam0 at t_i
        pb = np.array([frames[j][1][k] for k in ids])  # same points in cam0 at t_j
        inl = rigid_ransac(pa, pb, a.inlier_m, 200, rng)
        if inl is None or inl.sum() < a.min_common:
            continue
        _, r, t = umeyama(pa[inl], pb[inl], with_scale=False)  # p_j = R p_i + t  =>  T_ji
        stereo_m = float(np.linalg.norm(-r.T @ t))  # camera displacement, metres
        gt_rel = np.linalg.inv(gt_T[i]) @ gt_T[j]  # GT is T_world_cam0
        gt_m = float(np.linalg.norm(gt_rel[:3, 3]))
        if stereo_m < a.min_motion_m:
            continue
        ratios.append(gt_m / stereo_m)
        used += 1

    r = np.array(ratios)
    if not len(r):
        raise SystemExit("no usable frame pairs")
    print(f"pairs used {used} (dt {a.dt_s}s, depth < {a.max_depth_m} m)")
    print(f"|t_gt| / |t_stereo|: median {np.median(r):.3f}  IQR [{np.percentile(r, 25):.3f}, "
          f"{np.percentile(r, 75):.3f}]  mean {r.mean():.3f}")


if __name__ == "__main__":
    main()
