"""Oracle check (NOT part of the tool): is the GOOD in-water calibration's w_y really ~0?

Uses mocap GT only to fix the per-pair translation length s_p (so alpha_p = f B / s_p is known) and
then fits the shared disparity intercept. Lever arm body->camera is unknown, so s_p carries a few
percent noise; pairs with large GT rotation are dropped.

Usage: validate_gt.py <t0_s> <dur_s>
"""
from __future__ import annotations
import os

import sys

import numpy as np

import bagio
import calib
import estimators as E
from extract_cache import get_stretch

MOCAP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results/ws1_sync/mocap/traj_06/traj_06_mocap.tum")


def main(t0: float, dur: float) -> None:
    g = calib.load_ntnu_water()
    sd = get_stretch("ntnu", t0, dur)
    prep = E.prepare(g, sd)
    gt = np.loadtxt(MOCAP)
    base = bagio.bag_start_s(bagio.NTNU_BAG)
    pos = lambda t: np.array([np.interp(t + base, gt[:, 0], gt[:, 1 + i]) for i in range(3)])
    q = lambda t: np.array([np.interp(t + base, gt[:, 0], gt[:, 4 + i]) for i in range(4)])
    f = g.f_px
    sel = np.where(prep.mono_ok)[0]
    vl, vr = E._rect(prep, g.R_lr, sel)
    xl, yl = vl[:, 0] / vl[:, 2], vl[:, 1] / vl[:, 2]
    d = f * (xl - vr[:, 0] / vr[:, 2])
    Rr = calib.rect_rotation(g.t_lr)
    xinv = 1.0 / (Rr[2] @ (prep.nl[sel] * prep.lam[sel, None]).T)
    pairs = prep.pair[sel]
    s_gt, rot = {}, {}
    for k in np.unique(pairs):
        ta, tb = sd.t_key_s[k], sd.t_gap_s[k]
        s_gt[k] = np.linalg.norm(pos(tb) - pos(ta))
        qa, qb = q(ta), q(tb)
        rot[k] = np.degrees(2 * np.arccos(np.clip(abs(qa @ qb / np.linalg.norm(qa) / np.linalg.norm(qb)), 0, 1)))
    ok_pair = {k for k in s_gt if rot[k] < 6.0 and s_gt[k] > 0.05}
    keep = np.array([p in ok_pair for p in pairs])
    alpha = np.array([f * g.baseline_m / s_gt.get(p, np.nan) for p in pairs])
    m = 1 + xl ** 2
    ci = (d - alpha * xinv) / m
    ci = ci[keep]
    # scale check: stereo-implied s_p vs GT
    ratios = []
    for k in ok_pair:
        kk = pairs == k
        a_st = np.median(d[kk] / xinv[kk])
        ratios.append(f * g.baseline_m / a_st / s_gt[k])
    med = np.median(ci)
    mad = 1.4826 * np.median(np.abs(ci - med))
    print(f"[oracle] stretch {t0}-{t0 + dur}: pairs {len(ok_pair)}, feats {keep.sum()}, "
          f"s_stereo/s_gt median {np.median(ratios):.3f} (IQR {np.percentile(ratios, 25):.3f}-{np.percentile(ratios, 75):.3f})")
    print(f"[oracle] intercept c: median {med:.2f} px (robust sd {mad:.2f}, se {mad / np.sqrt(keep.sum()):.3f}) "
          f"=> w_y = {np.rad2deg(-med / f):.3f} deg")
    for lo, hi in [(0, 15), (15, 25), (25, 40), (40, 200)]:
        k2 = keep & (d >= lo) & (d < hi)
        if k2.sum() > 20:
            r = (d - alpha * xinv)[k2]
            print(f"[oracle]   disparity {lo}-{hi} px (n={k2.sum()}): median resid {np.median(r):.2f} px")


if __name__ == "__main__":
    main(float(sys.argv[1]), float(sys.argv[2]))
