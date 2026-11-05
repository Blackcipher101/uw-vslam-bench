"""Pose-free sonar->camera extrinsic search against stereo depth (same-time pairing, no trajectory needed).

Each ping is paired with the stereo frame nearest in time (|dt| <= max_gap_s, vehicle motion in that time is
negligible), the sonar point is mapped with a candidate T_c0_So, projected into the rectified left image and
compared to the SGBM depth there. Score = number of in-image points whose stereo depth agrees within tolerance.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .evalcore import Dataset
from .geometry import perturb, sonar_points, transform


def rotvec_grid(step_rad: float, max_angle_rad: float = np.pi) -> np.ndarray:
    g = np.arange(-max_angle_rad, max_angle_rad + 1e-9, step_rad)
    X = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    return X[np.linalg.norm(X, axis=1) <= max_angle_rad]


def tolerance(z: np.ndarray, abs_m: float = 0.15, rel: float = 0.05) -> np.ndarray:
    return abs_m + rel * z


def score_rotations(ds: Dataset, ping_idx, img_idx, rotvecs, t_so, heading_sign=1.0, chunk=100):
    """(matches, in_image_valid) per candidate rotation; translation t_so (c0 frame, m) is held fixed."""
    P = sonar_points(ds.range_m[ping_idx], heading_sign * ds.head_deg[ping_idx]).T  # (3,N)
    img = np.asarray(img_idx)
    h, w = ds.depth.shape[1:]
    match = np.zeros(len(rotvecs), int)
    valid = np.zeros(len(rotvecs), int)
    for a in range(0, len(rotvecs), chunk):
        R = Rotation.from_rotvec(rotvecs[a:a + chunk]).as_matrix()          # R_c0_So
        Pc = np.einsum("cij,jn->cin", R, P) + t_so[None, :, None]
        Pr = np.einsum("ij,cjn->cin", ds.R1, Pc)
        with np.errstate(all="ignore"):
            z = Pr[:, 2]
            u = ds.f * Pr[:, 0] / z + ds.cx
            v = ds.f * Pr[:, 1] / z + ds.cy
        col = np.floor(u / ds.block)
        row = np.floor(v / ds.block)
        ok = (z > 0.05) & (col >= 0) & (col < w) & (row >= 0) & (row < h)
        col = np.where(ok, col, 0).astype(int)
        row = np.where(ok, row, 0).astype(int)
        zst = ds.depth[img[None, :], row, col].astype(float)
        ok &= np.isfinite(zst)
        agree = ok & (np.abs(zst - z) < tolerance(z))
        match[a:a + chunk] = agree.sum(1)
        valid[a:a + chunk] = ok.sum(1)
    return match, valid


def truncated_cost(ds: Dataset, P_so: np.ndarray, img: np.ndarray, T_c0_So: np.ndarray, trunc_m: float = 0.5) -> float:
    """Mean min(|z_stereo - z_sonar|, trunc) over all pairs; out-of-image / no-stereo pairs count as trunc_m."""
    z, uv = ds.project_rect(transform(T_c0_So, P_so))
    r = np.abs(ds.lookup(img, uv) - z)
    return float(np.mean(np.minimum(np.where(np.isfinite(r), r, trunc_m), trunc_m)))


def refine_extrinsic(ds: Dataset, ping_idx, img_idx, T0: np.ndarray, heading_sign=1.0, free_translation=True):
    """Nelder-Mead on truncated_cost (the stereo lookup is piecewise constant, so no gradients), 3 rounds with a
    shrinking simplex. Parametrisation: left-multiplied camera-frame correction (rotvec rad, metres)."""
    P = sonar_points(ds.range_m[ping_idx], heading_sign * ds.head_deg[ping_idx])
    img = np.asarray(img_idx)
    n = 6 if free_translation else 3

    def T_of(x):
        return perturb(T0, x[:3], x[3:] if free_translation else (0, 0, 0))

    best = np.zeros(n)
    for scale in (1.0, 0.4, 0.15):
        sim = np.vstack([best] + [best + np.eye(n)[k] * (np.deg2rad(2.0) if k < 3 else 0.03) * scale
                                  for k in range(n)])
        sol = minimize(lambda x: truncated_cost(ds, P, img, T_of(x)), best, method="Nelder-Mead",
                       options=dict(initial_simplex=sim, xatol=1e-4, fatol=1e-6, maxiter=400))
        best = sol.x
    return T_of(best), float(sol.fun), best
