"""Sonar-vs-stereo depth baseline, time-offset sweep and extrinsic-perturbation sweep.

  python3 baseline.py --cache out/depth_cache.npz --traj <tum> [--extrinsic svin|<calib.json>] --out out/baseline_x.json
Residual = z_stereo - z_sonar (both z in the rectified left camera frame, metres), see sonar_cam/evalcore.py.
Inlier = |res| < 0.15 m + 5 % of z_sonar. spearman = rank correlation of z_sonar vs z_stereo over all pairs
(the key association statistic: ~0 means the sonar points carry no information about the stereo depth).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from sonar_cam.calib import tolerance
from sonar_cam.data import GT_COLMAP
from sonar_cam.evalcore import Dataset, Pairs, robust_stats
from sonar_cam.geometry import perturb
from project import load_extrinsic

BINS = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.6, 4.5]


def summarize(p: Pairs) -> dict:
    s = robust_stats(p.res)
    s["unique_pings"] = int(len(np.unique(p.ping_idx)))
    s["unique_frames"] = int(len(np.unique(p.img_idx)))
    s["inlier_frac"] = float(np.mean(np.abs(p.res) < tolerance(p.z_sonar))) if len(p.res) else float("nan")
    s["spearman"] = float(spearmanr(p.z_sonar, p.z_stereo)[0]) if len(p.res) > 10 else float("nan")
    s["rel_median"] = float(np.median(p.res / p.z_sonar)) if len(p.res) else float("nan")
    s["rel_mad"] = float(1.4826 * np.median(np.abs(p.res / p.z_sonar - np.median(p.res / p.z_sonar)))) if len(p.res) else float("nan")
    return s


def by_distance(p: Pairs) -> list[dict]:
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (p.z_sonar >= lo) & (p.z_sonar < hi)
        r = p.res[m]
        rel = r / p.z_sonar[m]
        out.append(dict(bin=f"{lo:.1f}-{hi:.1f}", n=int(m.sum()),
                        res_median=float(np.median(r)) if m.any() else np.nan,
                        res_mad=float(1.4826 * np.median(np.abs(r - np.median(r)))) if m.any() else np.nan,
                        rel_median=float(np.median(rel)) if m.any() else np.nan,
                        rel_mad=float(1.4826 * np.median(np.abs(rel - np.median(rel)))) if m.any() else np.nan))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--traj", default=GT_COLMAP)
    ap.add_argument("--extrinsic", default="svin")
    ap.add_argument("--window-s", type=float, default=4.2)
    ap.add_argument("--future-s", type=float, default=0.0)
    ap.add_argument("--min-range", type=float, default=0.0)
    ap.add_argument("--min-peak", type=int, default=10)
    ap.add_argument("--picker", choices=("svin", "contrast"), default="svin")
    ap.add_argument("--min-prom", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ds = Dataset(a.cache, a.traj, min_range_m=a.min_range, min_intensity=a.min_peak,
                 picker=a.picker, min_prominence=a.min_prom)
    T0, sign = load_extrinsic(a.extrinsic)
    kw = dict(window_s=a.window_s, future_s=a.future_s, heading_sign=sign)
    res = dict(args=vars(a), n_frames=int(len(ds.t_img)), n_valid_pings=int(np.isfinite(ds.range_m).sum()))
    base = ds.window_pairs(T0, 0.0, **kw)
    res["baseline"] = summarize(base)
    res["by_distance"] = by_distance(base)
    print("baseline", res["baseline"])
    for r in res["by_distance"]:
        print(r)
    res["time_sweep"] = []
    for dt in np.round(np.arange(-0.5, 0.5001, 0.1), 2):
        s = summarize(ds.window_pairs(T0, float(dt), **kw))
        s["dt_s"] = float(dt)
        res["time_sweep"].append(s)
        print("dt", dt, s)
    res["perturb"] = []
    for kind, vals, unit in (("rot", (-5, -2, -1, 1, 2, 5), np.deg2rad(1)), ("trans", (-0.10, -0.05, -0.02, 0.02, 0.05, 0.10), 1.0)):
        for ax in range(3):
            for v in vals:
                d = np.zeros(3)
                d[ax] = v * unit
                T = perturb(T0, d, (0, 0, 0)) if kind == "rot" else perturb(T0, (0, 0, 0), d)
                s = summarize(ds.window_pairs(T, 0.0, **kw))
                s.update(kind=kind, axis="xyz"[ax], value=v)
                res["perturb"].append(s)
                print(kind, "xyz"[ax], v, s)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
