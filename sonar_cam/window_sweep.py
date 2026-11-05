"""Number of comparable sonar/stereo points and residual statistics vs the accumulation window length.

  python3 window_sweep.py --cache out/depth_cache.npz --traj <tum> [--extrinsic svin|<calib.json>] [--future-s F]
"""
from __future__ import annotations

import argparse

from baseline import summarize
from project import load_extrinsic
from sonar_cam.evalcore import Dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--extrinsic", default="svin")
    ap.add_argument("--windows", type=float, nargs="+", default=[4.2, 10, 20, 40, 80])
    ap.add_argument("--future-s", type=float, default=0.0)
    ap.add_argument("--picker", choices=("svin", "contrast"), default="svin")
    a = ap.parse_args()
    ds = Dataset(a.cache, a.traj, picker=a.picker)
    T, sign = load_extrinsic(a.extrinsic)
    for w in a.windows:
        s = summarize(ds.window_pairs(T, 0.0, window_s=w, future_s=a.future_s, heading_sign=sign))
        print(f"window {w:5.1f}s future {a.future_s}: " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()), flush=True)


if __name__ == "__main__":
    main()
