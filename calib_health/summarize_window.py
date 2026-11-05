"""Summary of results/window_study.json: repeatability and verdict rates vs window length."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import calib
import estimators as E


def main() -> None:
    rows = json.loads((Path(__file__).parent / "results" / "window_study.json").read_text())
    good = calib.load_ntnu_water()
    print("window  n_win | zero: y mean  y sd(across windows)  x sd  z sd  false-alarm(abs) | y+-0.1: flag(diff)  y+-0.3: flag(diff) | x+0.2 flag z+0.2 flag (diff)")
    for win in (10, 20, 30, 60):
        rs = [r for r in rows if r["win"] == win]
        z = [r for r in rs if r["case"] == "zero"]
        if not z:
            continue
        zy = np.array([r["fused"][1] for r in z])
        zx = np.array([r["fused"][0] for r in z])
        zz = np.array([r["fused"][2] for r in z])
        fa = np.mean([r["verdict"] != "OK" for r in z])
        # differential: shift relative to the zero estimate of the SAME window = exact by construction, so use
        # the pooled zero mean (of all windows of that length) as the reference instead
        ref = np.mean([r["fused"] for r in z], axis=0)

        def dflag(case_names):
            sel = [r for r in rs if r["case"] in case_names]
            return np.mean([E.verdict(good, np.array(r["fused"]) - ref) != "OK" for r in sel]), len(sel)
        f1, _ = dflag(("y+0.1", "y-0.1"))
        f3, _ = dflag(("y+0.3", "y-0.3"))
        fx, _ = dflag(("x+0.2",))
        fz, _ = dflag(("z+0.2",))
        zd = np.mean([E.verdict(good, np.array(r["fused"]) - ref) != "OK" for r in z])
        print(f"{win:3d} s  {len(z):3d}   | {zy.mean():+.3f}  {zy.std():.3f}  {zx.std():.3f}  {zz.std():.3f}  {fa:.2f} (diff-mode FA {zd:.2f}) | "
              f"{f1:.2f}  {f3:.2f} | {fx:.2f} {fz:.2f}")


if __name__ == "__main__":
    main()
