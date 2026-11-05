"""Injected-error baseline on NTNU traj_06 (GOOD in-water calibration + known rotation errors).

Usage: baseline.py [--boot 30] [--eval 105,170,235] [--dev 40] [--dur 60]
Prints one line per case as it finishes, then summary tables. Writes results/baseline.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import calib
import estimators as E
from extract_cache import get_stretch

LEVELS = [0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.5, -0.5]
AXES = ["x", "y", "z"]
OUT = Path(__file__).parent / "results"


def cases(good: calib.Rig):
    out = [("zero", np.zeros(3))]
    for ai, a in enumerate(AXES):
        for lv in LEVELS[1:]:
            w = np.zeros(3)
            w[ai] = lv
            out.append((f"{a}{lv:+.1f}", w))
    out.append(("air", calib.air_vs_water_error_deg(good)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=30)
    ap.add_argument("--eval", default="105,170,235")
    ap.add_argument("--dev", type=float, default=40)
    ap.add_argument("--dur", type=float, default=60)
    a = ap.parse_args()
    good = calib.load_ntnu_water()
    rows, timing = [], {}
    stretches = [("dev", a.dev)] + [("eval", float(t)) for t in a.eval.split(",")]
    n_tot = len(stretches) * len(cases(good))
    done, t_start = 0, time.time()
    for role, t0 in stretches:
        t = time.time()
        sd = get_stretch("ntnu", t0, a.dur)
        timing[f"extract_{int(t0)}_s"] = time.time() - t
        t = time.time()
        prep = E.prepare(good, sd)
        timing[f"prepare_{int(t0)}_s"] = time.time() - t
        for name, w in cases(good):
            t = time.time()
            r = E.estimate(prep, calib.inject(good, w), n_boot=a.boot, seed=int(t0))
            dt = time.time() - t
            v = E.verdict(good, r.fused_deg)
            row = dict(role=role, t0=t0, case=name, inj=w.tolist(), epi=r.epi_deg.tolist(), mot=r.mot_deg.tolist(),
                       fused=r.fused_deg.tolist(), ci=r.ci_deg.tolist(), se={k: v_.tolist() for k, v_ in r.se_deg.items()},
                       consistent=r.consistent, verdict=v, n_mono=r.n_mono, n_pairs=r.n_pairs, dy_rms=r.dy_rms_px, sec=dt)
            rows.append(row)
            done += 1
            print(f"[{done}/{n_tot}] {role} t0={t0:.0f} {name:7s} inj={np.round(w, 2).tolist()} "
                  f"fused={np.round(r.fused_deg, 3).tolist()} mot_y={r.mot_deg[1]:+.3f} se_mot_y={r.se_deg['mot'][1]:.3f} "
                  f"ci_y=[{r.ci_deg[1, 0]:+.3f},{r.ci_deg[1, 1]:+.3f}] verdict={v} {dt:.1f}s", flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "baseline.json").write_text(json.dumps(dict(rows=rows, timing=timing, boot=a.boot), indent=0))
    print(f"[done] {time.time() - t_start:.0f} s total", flush=True)


if __name__ == "__main__":
    main()
