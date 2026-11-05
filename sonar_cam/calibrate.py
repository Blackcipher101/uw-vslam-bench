"""Pose-free sonar->camera extrinsic search (see sonar_cam/calib.py).

  python3 calibrate.py --cache out/depth_cache.npz --traj <tum> --out out/calib.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.spatial.transform import Rotation

from sonar_cam.calib import refine_extrinsic, rotvec_grid, score_rotations
from sonar_cam.data import GT_COLMAP
from sonar_cam.evalcore import Dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--traj", default=GT_COLMAP)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step-deg", type=float, default=6.0)
    ap.add_argument("--max-pairs", type=int, default=12000)
    ap.add_argument("--min-peak", type=int, default=10)
    ap.add_argument("--picker", choices=("svin", "contrast"), default="svin")
    ap.add_argument("--min-prom", type=float, default=30.0)
    ap.add_argument("--min-range", type=float, default=0.0)
    ap.add_argument("--null-shift-s", type=float, default=37.0)
    a = ap.parse_args()
    ds = Dataset(a.cache, a.traj, min_range_m=a.min_range, min_intensity=a.min_peak,
                 picker=a.picker, min_prominence=a.min_prom)
    pi, ii = ds.same_time_points()
    rng = np.random.default_rng(0)
    sel = rng.permutation(len(pi))[: a.max_pairs]
    pi, ii = pi[sel], ii[sel]
    print(f"pairs {len(pi)}", flush=True)
    grid = rotvec_grid(np.deg2rad(a.step_deg))
    t_so = ds.T_c0_So[:3, 3]
    out = dict(n_pairs=int(len(pi)), n_rot=int(len(grid)), step_deg=a.step_deg, svin_T_c0_So=ds.T_c0_So.tolist())
    svin_rv = Rotation.from_matrix(ds.T_c0_So[:3, :3]).as_rotvec()
    res = {}
    for name, shift in (("real", 0.0), ("null_shifted", a.null_shift_s)):
        for sign in (1.0, -1.0):
            t0 = time.time()
            # null: pair every ping with the stereo frame nearest to (t_ping + shift): breaks the true association
            if shift:
                p2, i2 = ds.same_time_points(dt_s=shift)
                s2 = rng.permutation(len(p2))[: a.max_pairs]
                p_, i_ = p2[s2], i2[s2]
            else:
                p_, i_ = pi, ii
            m, v = score_rotations(ds, p_, i_, grid, t_so, sign)
            k = np.argsort(-m)[:10]
            res[(name, sign)] = (m, v, grid, p_, i_)
            print(f"{name} sign={sign:+.0f}: best {m[k[0]]} matches ({v[k[0]]} in-image) of {len(p_)} pairs; "
                  f"median over grid {np.median(m):.0f}; {time.time() - t0:.0f}s", flush=True)
            out[f"{name}_sign{int(sign)}"] = dict(
                top=[dict(matches=int(m[j]), in_image=int(v[j]), rotvec=grid[j].tolist()) for j in k],
                grid_median=float(np.median(m)), grid_p99=float(np.percentile(m, 99)), n_pairs=int(len(p_)))
    # refine the best real candidate
    best = max(((res[("real", s)][0].max(), s) for s in (1.0, -1.0)))
    sign = best[1]
    m, v, g, p_, i_ = res[("real", sign)]
    from sonar_cam.geometry import perturb
    T0 = np.eye(4)
    T0[:3, :3] = Rotation.from_rotvec(g[np.argmax(m)]).as_matrix()
    T0[:3, 3] = t_so
    T, cost, x = refine_extrinsic(ds, p_, i_, T0, sign)
    out["refined"] = dict(heading_sign=sign, T_c0_So=T.tolist(), truncated_cost_m=cost,
                          svin_rotvec=svin_rv.tolist(), coarse_rotvec=Rotation.from_matrix(T0[:3, :3]).as_rotvec().tolist(),
                          refined_rotvec=Rotation.from_matrix(T[:3, :3]).as_rotvec().tolist(), t_c0_So=T[:3, 3].tolist())
    json.dump(out, open(a.out, "w"), indent=1)
    print("refined", json.dumps(out["refined"]))


if __name__ == "__main__":
    main()
