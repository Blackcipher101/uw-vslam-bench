"""Map a MASt3R-SLAM keyframe trajectory back to bag time.

MASt3R-SLAM writes ``t x y z qx qy qz qw`` per keyframe (T_world_cam0) with
``t = frame_index / 30`` from its RGBFiles loader; ``stamps.txt`` from
``mast3r_prepare.py`` holds the real stamp (ns) of each exported frame.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

FAKE_FPS = 30.0  # RGBFiles in mast3r_slam/dataloader.py


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("traj", type=Path, help="MASt3R-SLAM trajectory .txt")
    ap.add_argument("stamps", type=Path, help="stamps.txt from mast3r_prepare.py")
    ap.add_argument("out", type=Path, help="TUM file in bag time (s)")
    args = ap.parse_args()

    stamps = np.loadtxt(args.stamps, dtype=np.int64, ndmin=1)
    rows = np.loadtxt(args.traj, ndmin=2)
    if rows.shape[1] != 8:
        raise SystemExit(f"{args.traj}: expected 8 columns, got {rows.shape[1]}")
    idx_f = rows[:, 0] * FAKE_FPS
    idx = np.rint(idx_f).astype(np.int64)
    # float32 stamps lose precision only beyond ~2^24/30 frames; anything off-grid means
    # the trajectory did not come from this export.
    if np.any(np.abs(idx_f - idx) > 0.05) or idx.min() < 0 or idx.max() >= len(stamps):
        raise SystemExit("trajectory stamps do not match stamps.txt (wrong export or fps)")
    if np.any(np.diff(idx) <= 0):
        raise SystemExit("keyframe indices are not strictly increasing")
    with args.out.open("w") as f:
        for i, r in zip(idx, rows):
            f.write(f"{stamps[i] / 1e9:.9f} " + " ".join(f"{v:.9g}" for v in r[1:]) + "\n")
    print(f"wrote {len(rows)} keyframe poses to {args.out}")


if __name__ == "__main__":
    main()
