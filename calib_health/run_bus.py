"""Run the estimator once per stretch on bus_outside with its given calibration (no GT for the true error).

Usage: run_bus.py [t0,t0,...] [dur_s]
"""
from __future__ import annotations

import sys
import time

import numpy as np

import calib
import estimators as E
from extract_cache import get_stretch


def main() -> None:
    t0s = [float(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "0,70,140").split(",")]
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    rig = calib.load_bus()
    print(f"[bus] f={rig.f_px:.1f} px, B={rig.baseline_m:.4f} m; 0.1 deg about y = "
          f"{E.implied(rig, np.array([0, .1, 0]))['err_3m'] * 100:.1f} % depth error at 3 m", flush=True)
    for t0 in t0s:
        t = time.time()
        sd = get_stretch("bus", t0, dur)
        t_ex = time.time() - t
        prep = E.prepare(rig, sd)
        t = time.time()
        r = E.estimate(prep, rig, n_boot=30, seed=int(t0))
        im = E.implied(rig, r.fused_deg)
        print(f"[bus] t0={t0:.0f}-{t0 + dur:.0f}: matches={len(sd.frame)} mono={r.n_mono} pairs={r.n_pairs}/{len(sd.t_key_s)} "
              f"epi={np.round(r.epi_deg, 3).tolist()} mot={np.round(r.mot_deg, 3).tolist()} "
              f"fused={np.round(r.fused_deg, 3).tolist()} ci90_y=[{r.ci_deg[1, 0]:+.3f},{r.ci_deg[1, 1]:+.3f}] "
              f"se_epi_y={r.se_deg['epi'][1]:.3f} se_mot_y={r.se_deg['mot'][1]:.3f} consistent={r.consistent} "
              f"dy_rms={im['dy_rms_px']:.2f}px err3m={im['err_3m'] * 100:.1f}% verdict={E.verdict(rig, r.fused_deg)} "
              f"(extract {t_ex:.0f}s, estimate {time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
