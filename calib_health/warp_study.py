"""Image-level injection: re-render the RIGHT image as if the right camera were rotated, re-extract
tracks from the warped images and run the estimator with the unchanged GOOD calibration.

Unlike the calibration-side injection of baseline.py (which reuses identical matches, so the shift
is equivariant by construction), this changes the pixels, so KLT noise, interpolation blur and
the whole measurement chain are exercised. Expected shift of the estimate = injected w.

Usage: warp_study.py [t0,t0,...]   (default 105,170)   Writes results/warp_study.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import bagio
import calib
import estimators as E
import tracks

CASES = {"y+0.3": [0, .3, 0], "y-0.3": [0, -.3, 0], "y+0.1": [0, .1, 0], "x+0.3": [.3, 0, 0], "z+0.3": [0, 0, .3]}


def warp_maps(rig: calib.Rig, w_deg: np.ndarray):
    """Remap tables producing the right image of a camera for which the GOOD calibration shows error w."""
    Q = rig.R_lr.T @ calib.exp_so3(np.deg2rad(w_deg)) @ rig.R_lr  # x_r' = Q x_r
    W, H = rig.right.size
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    n = rig.right.normalize(np.stack([u.ravel(), v.ravel()], axis=1))
    ray = Q.T @ np.vstack([n.T, np.ones(len(n))])
    src = rig.right.project((ray[:2] / ray[2]).T)
    return (src[:, 0].reshape(H, W).astype(np.float32), src[:, 1].reshape(H, W).astype(np.float32))


def main() -> None:
    t0s = [float(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "105,170").split(",")]
    good = calib.load_ntnu_water()
    cases = {**CASES, "air": list(calib.air_vs_water_error_deg(good))}
    rows = []
    for t0 in t0s:
        base = E.estimate(E.prepare(good, tracks.StretchData.load(Path(__file__).parent / "cache" / f"ntnu_{int(t0)}_60_k6g6.npz")),
                          good, n_boot=10, seed=1)
        print(f"[warp] t0={t0:.0f} unwarped: fused={np.round(base.fused_deg, 3).tolist()}", flush=True)
        for name, w in cases.items():
            t = time.time()
            mx, my = warp_maps(good, np.array(w, float))
            fr = ((ts, l, cv2.remap(r, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
                  for ts, l, r in bagio.iter_pairs(bagio.NTNU_BAG, bagio.NTNU_LEFT, bagio.NTNU_RIGHT, t0, t0 + 60, compressed=False))
            sd = tracks.extract(fr)
            r = E.estimate(E.prepare(good, sd), good, n_boot=10, seed=1)
            shift = r.fused_deg - base.fused_deg
            err = shift - np.array(w)
            rows.append(dict(t0=t0, case=name, inj=list(map(float, w)), shift=shift.tolist(), err=err.tolist(),
                             epi_y_shift=float(r.epi_deg[1] - base.epi_deg[1]), mot_y_shift=float(r.mot_deg[1] - base.mot_deg[1]),
                             verdict=E.verdict(good, r.fused_deg - base.fused_deg), sec=time.time() - t))
            print(f"[warp] t0={t0:.0f} {name:6s} inj={np.round(w, 3).tolist()} shift={np.round(shift, 3).tolist()} "
                  f"err={np.round(err, 3).tolist()} epi_y_shift={rows[-1]['epi_y_shift']:+.3f} "
                  f"mot_y_shift={rows[-1]['mot_y_shift']:+.3f} diff-verdict={rows[-1]['verdict']} {time.time() - t:.0f}s", flush=True)
    out = Path(__file__).parent / "results"
    (out / "warp_study.json").write_text(json.dumps(rows))
    err = np.array([r["err"] for r in rows])
    print("[warp] per-axis error of the shift (deg): bias", np.round(err.mean(0), 3).tolist(),
          "rmse", np.round(np.sqrt((err ** 2).mean(0)), 3).tolist(), f"n={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
