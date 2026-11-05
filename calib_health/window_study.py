"""Precision vs data length: sub-windows (10/20/30/60 s) of the cached NTNU eval stretches.

Injects {0, y+-0.1, y+-0.3, x+0.2, z+0.2} deg into the GOOD calibration. Reports the spread of the
estimate across windows (the estimator's repeatability) and the verdict rates.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import calib
import estimators as E
from extract_cache import get_stretch

CASES = {"zero": [0, 0, 0], "y+0.1": [0, .1, 0], "y-0.1": [0, -.1, 0], "y+0.3": [0, .3, 0], "y-0.3": [0, -.3, 0],
         "x+0.2": [.2, 0, 0], "z+0.2": [0, 0, .2]}


def main() -> None:
    good = calib.load_ntnu_water()
    rows = []
    for t0 in (105.0, 170.0, 235.0):
        sd = get_stretch("ntnu", t0, 60.0)
        prep = E.prepare(good, sd)
        for win in (10, 20, 30, 60):
            for w0 in range(0, 60, win):
                keys = np.where((sd.t_key_s >= w0 + sd.t_key_s[0]) & (sd.t_key_s < w0 + win + sd.t_key_s[0]))[0]
                if len(keys) < 5:
                    continue
                for name, inj in CASES.items():
                    r = E.estimate(prep, calib.inject(good, np.array(inj, float)), n_boot=10, keyframes=keys)
                    v = E.verdict(good, r.fused_deg)
                    rows.append(dict(stretch=t0, win=win, w0=w0, case=name, inj=inj, fused=r.fused_deg.tolist(),
                                     epi_y=float(r.epi_deg[1]), mot_y=float(r.mot_deg[1]), verdict=v, n_pairs=r.n_pairs))
                    print(f"win={win:2d}s t0={t0:.0f}+{w0:2d} {name:6s} fused={np.round(r.fused_deg, 3).tolist()} "
                          f"mot_y={r.mot_deg[1]:+.3f} pairs={r.n_pairs} {v}", flush=True)
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "window_study.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
