"""Accuracy of dense stereo registrations against ground truth (step-6 diagnostics).

Compares, for every keyframe pair in a run_bag --dense-log CSV, the relative left-camera motion from
  init  the estimator's prediction (ICP initial guess), and
  icp   the dense registration,
with the ground-truth relative motion of the left camera. The GT body->camera transform is found by
hand-eye calibration against the run's own trajectory (as evaluate.py --hand-eye does).

Usage: python dense_vs_gt.py <gt.tum> <est.tum> <dense_log.csv> [--hand-eye-first-s 60]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import hand_eye, load_tum, to_mats  # noqa: E402

GT_TOL_S = 0.02


def pose(v: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(v[3:7]).as_matrix()
    T[:3, 3] = v[:3]
    return T


def err(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    D = np.linalg.inv(A) @ B
    return float(np.linalg.norm(D[:3, 3])), float(np.degrees(Rotation.from_matrix(D[:3, :3]).magnitude()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gt", type=Path)
    ap.add_argument("est", type=Path)
    ap.add_argument("log", type=Path)
    ap.add_argument("--hand-eye-first-s", type=float, default=60.0)
    a = ap.parse_args()
    gt, est = load_tum(a.gt), load_tum(a.est)
    X, info = hand_eye(gt, est, a.hand_eye_first_s)
    G = to_mats(gt) @ X  # GT left-camera poses
    tg = gt[:, 0]
    rows = []
    for line in a.log.read_text().splitlines():
        if line.startswith("#"):
            continue
        f = line.split()
        t_ref, t_cur, ok = float(f[0]), float(f[1]), f[2] == "1"
        i, j = np.abs(tg - t_ref).argmin(), np.abs(tg - t_cur).argmin()
        if abs(tg[i] - t_ref) > GT_TOL_S or abs(tg[j] - t_cur) > GT_TOL_S:
            continue
        rel = np.linalg.inv(G[i]) @ G[j]
        v = np.array(f[6:], float)
        init, icp, eig = pose(v[0:7]), pose(v[7:14]), v[14:20]
        rows.append((ok, *err(rel, init), *err(rel, icp), float(np.linalg.norm(rel[:3, 3])), eig[0], eig[5]))
    r = np.array(rows, float)
    print(f"hand-eye {info}")
    if not len(r):
        print("no registrations matched GT")
        return
    for name, m in (("all", np.ones(len(r), bool)), ("icp ok", r[:, 0] == 1), ("icp failed", r[:, 0] == 0)):
        s = r[m]
        if not len(s):
            continue
        print(f"{name:10s} n={len(s):4d}  motion med {np.median(s[:, 5]) * 100:5.1f} cm | "
              f"init err med {np.median(s[:, 1]) * 100:5.2f} cm {np.median(s[:, 2]):5.2f} deg  p90 {np.percentile(s[:, 1], 90) * 100:5.2f} cm | "
              f"icp err med {np.median(s[:, 3]) * 100:5.2f} cm {np.median(s[:, 4]):5.2f} deg  p90 {np.percentile(s[:, 3], 90) * 100:5.2f} cm | "
              f"icp better {np.mean(s[:, 3] < s[:, 1]) * 100:4.0f} %")


if __name__ == "__main__":
    main()
