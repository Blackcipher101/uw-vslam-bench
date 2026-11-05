"""Summary tables from results/baseline.json (eval stretches only; the dev stretch is the reference)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

import calib
import estimators as E

AX = "xyz"


def main() -> None:
    d = json.loads((Path(__file__).parent / "results" / "baseline.json").read_text())
    good = calib.load_ntnu_water()
    rows = d["rows"]
    ev = [r for r in rows if r["role"] == "eval"]
    zero = {r["t0"]: np.array(r["fused"]) for r in ev if r["case"] == "zero"}
    dev0 = next(np.array(r["fused"]) for r in rows if r["role"] == "dev" and r["case"] == "zero")
    zero_epi = {r["t0"]: np.array(r["epi"]) for r in ev if r["case"] == "zero"}
    zero_mot = {r["t0"]: np.array(r["mot"]) for r in ev if r["case"] == "zero"}
    print("== Residual of the GOOD in-water calibration as seen by the estimator (inj = 0), deg [x y z]")
    for t0, z in zero.items():
        print(f"  eval stretch {t0:.0f}-{t0 + 60:.0f}: fused {np.round(z, 3).tolist()}  epi_y {zero_epi[t0][1]:+.3f}  mot_y {zero_mot[t0][1]:+.3f}")
    print(f"  dev stretch 40-100: fused {np.round(dev0, 3).tolist()}")

    print("\n== Per-case table (3 eval stretches): estimate on the injected axis")
    print("case     inj_deg  est_mean  est_sd  shift_mean(est-est0)  err_vs_inj(shift-inj)  verdicts OK/CHECK/RECAL  truth")
    per_axis = {a: [] for a in range(3)}
    for name in [r["case"] for r in rows if r["role"] == "dev"]:
        if name in ("zero",):
            continue
        rs = [r for r in ev if r["case"] == name]
        if not rs:
            continue
        inj = np.array(rs[0]["inj"])
        ax = int(np.argmax(np.abs(inj))) if name != "air" else 1
        est = np.array([r["fused"][ax] for r in rs])
        shift = np.array([r["fused"][ax] - zero[r["t0"]][ax] for r in rs])
        vc = Counter(r["verdict"] for r in rs)
        truth = E.verdict(good, inj)
        print(f"{name:7s}  {inj[ax]:+.3f}   {est.mean():+.3f}   {est.std():.3f}   {shift.mean():+.3f}                 {shift.mean() - inj[ax]:+.3f}"
              f"                 {vc.get('OK', 0)}/{vc.get('CHECK', 0)}/{vc.get('RECALIBRATE', 0)}            {truth}")
        if name != "air":
            per_axis[ax] += [(inj[ax], r["fused"][ax], r["fused"][ax] - zero[r["t0"]][ax]) for r in rs]
        else:
            for k in (0, 2):
                shift_k = np.array([r["fused"][k] - zero[r["t0"]][k] for r in rs])
                print(f"   air other axes: {AX[k]} shift {shift_k.mean():+.3f} (true {inj[k]:+.3f})")

    print("\n== Per-axis accuracy over all non-zero levels x eval stretches (deg)")
    print("axis  n   raw_bias  raw_rmse   diff_bias  diff_rmse   (raw = est-inj, diff = est-est0-inj; est0 = same-stretch zero-injection estimate)")
    for a in range(3):
        t = np.array(per_axis[a])
        raw, df = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
        print(f"{AX[a]}     {len(t)}  {raw.mean():+.3f}     {np.sqrt(np.mean(raw ** 2)):.3f}      {df.mean():+.3f}      {np.sqrt(np.mean(df ** 2)):.3f}")

    print("\n== Off-axis leakage: mean |shift| of the non-injected axes (deg)")
    for a in range(3):
        leak = []
        for r in ev:
            if r["case"] in ("zero", "air"):
                continue
            inj = np.array(r["inj"])
            if int(np.argmax(np.abs(inj))) != a:
                continue
            for k in range(3):
                if k != a:
                    leak.append(abs(r["fused"][k] - zero[r["t0"]][k]))
        print(f"  inject {AX[a]}: mean off-axis |shift| {np.mean(leak):.3f}, max {np.max(leak):.3f}")

    print("\n== Detection (verdict != OK = 'flag'); truth class from the injected value via the same thresholds")
    print("injected |level|  axis   n   flag_rate  RECAL_rate  verdict==truth   [absolute mode]   flag_rate_diff-mode(ref=dev zero)")
    for lv in (0.1, 0.2, 0.3, 0.5):
        for a in range(3):
            rs = [r for r in ev if r["case"] != "zero" and r["case"] != "air" and int(np.argmax(np.abs(r["inj"]))) == a
                  and abs(r["inj"][a]) == lv]
            flag = np.mean([r["verdict"] != "OK" for r in rs])
            rec = np.mean([r["verdict"] == "RECALIBRATE" for r in rs])
            match = np.mean([r["verdict"] == E.verdict(good, np.array(r["inj"])) for r in rs])
            dm = np.mean([E.verdict(good, np.array(r["fused"]) - dev0) != "OK" for r in rs])
            print(f"{lv:.1f}               {AX[a]}     {len(rs)}   {flag:.2f}       {rec:.2f}        {match:.2f}                             {dm:.2f}")
    z = [r for r in ev if r["case"] == "zero"]
    print(f"false alarm at 0 injected: absolute mode {np.mean([r['verdict'] != 'OK' for r in z]):.2f} "
          f"({Counter(r['verdict'] for r in z)}), differential mode "
          f"{np.mean([E.verdict(good, np.array(r['fused']) - dev0) != 'OK' for r in z]):.2f}")

    print("\n== y-axis: static epipolar (epi) vs motion intercept (mot) vs fused, shift vs same-stretch zero (deg)")
    for m in ("epi", "mot", "fused"):
        errs = []
        for r in ev:
            if r["case"] in ("zero", "air") or int(np.argmax(np.abs(r["inj"]))) != 1:
                continue
            z0 = next(x for x in ev if x["t0"] == r["t0"] and x["case"] == "zero")
            errs.append(r[m][1] - z0[m][1] - r["inj"][1])
        errs = np.array(errs)
        print(f"  {m:5s}: bias {errs.mean():+.3f}  rmse {np.sqrt(np.mean(errs ** 2)):.3f}  max|err| {np.abs(errs).max():.3f}  n={len(errs)}")
    se = [np.array(r["se"]["epi"])[1] for r in ev if r["case"] == "zero"]
    sm = [np.array(r["se"]["mot"])[1] for r in ev if r["case"] == "zero"]
    print(f"  bootstrap SE of y at zero: epi {np.mean(se):.3f}  mot {np.mean(sm):.3f}")
    print("\n== timing (s): " + ", ".join(f"{k}={v:.1f}" for k, v in d["timing"].items()) +
          f"; mean estimate per case {np.mean([r['sec'] for r in rows]):.1f} s (boot={d['boot']})")


if __name__ == "__main__":
    main()
