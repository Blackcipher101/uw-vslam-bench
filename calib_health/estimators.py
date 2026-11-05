"""Stereo extrinsic health estimators.

Model: candidate rotation ``R_cand = Exp(w) R_true`` (w in left-frame axes, rad); the baseline
direction is trusted. In baseline-aligned rectified normalized coordinates (x', y') of the left
ray, to first order in w:

    dy = yl' - yr' =  w_x (1 + y'^2) - w_z x' - w_y x' y'          (vertical disparity, x f px)
    dd = d_cand - d_true = f [ w_z y' + w_x x' y' - w_y (1 + x'^2) ] (disparity bias, px)

Stage 1 (epipolar, static): robust LS of dy on the three basis functions -> w_x, w_z precisely;
  w_y enters only through the second-order x'y' term (weak, reported as method "epi_y").
Stage 2 (motion, scale-free): left-camera essential matrix between frames k and k+gap gives
  inverse depth x_inv = 1/Z up to a per-pair scale s_p. Measured disparity must satisfy
  d = alpha_p x_inv + c (1 + x'^2) with alpha_p = f B / s_p, so the shared intercept
  c = -f w_y is a disparity bias that no scale can absorb: a disparity bias makes the motion
  scale range-dependent, which is exactly a non-zero intercept. Solved by variable projection
  with effective-variance IRLS (errors in x_inv are propagated, otherwise OLS is attenuated).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

import calib
from calib import Rig, exp_so3, rect_rotation
from tracks import StretchData

SIGMA_PX = 0.3          # assumed KLT localisation noise (px), used only for weights
MIN_PAIR_FEATS = 25
MIN_PARALLAX_PX = 6.0   # median derotated left parallax over the pair
MIN_RAY_ANGLE_DEG = 8.0  # ray vs translation direction, avoids the epipole singularity


@dataclass
class Prepared:
    rig: Rig
    frame: np.ndarray
    nl: np.ndarray        # (N,3) left homogeneous normalized rays (x/z, y/z, 1)
    nr: np.ndarray        # (N,3)
    mono_ok: np.ndarray   # (N,) bool: has valid mono depth
    lam: np.ndarray       # (N,) left depth in unit-|t| scale (NaN if invalid)
    sig_x: np.ndarray     # (N,) std of 1/lam in the same units
    pair: np.ndarray      # (N,) keyframe id of the pair (== frame)
    n_frames: int
    n_pairs_used: int


def _pair_geometry(a: np.ndarray, b: np.ndarray, f: float):
    """Essential matrix on normalized points; returns (R, t, inlier mask) or None."""
    thr = 1.0 / f
    E, m = cv2.findEssentialMat(a[:, :2], b[:, :2], focal=1.0, pp=(0.0, 0.0), method=cv2.RANSAC,
                                prob=0.999, threshold=thr)
    if E is None or E.shape != (3, 3):
        return None
    n, R, t, m2 = cv2.recoverPose(E, a[:, :2], b[:, :2], focal=1.0, pp=(0.0, 0.0), mask=m.copy())
    if n < MIN_PAIR_FEATS:
        return None
    return R, t.ravel(), m2.ravel() > 0


def prepare(rig: Rig, sd: StretchData) -> Prepared:
    nl = np.hstack([rig.left.normalize(sd.pl), np.ones((len(sd.pl), 1))])
    nr = np.hstack([rig.right.normalize(sd.pr), np.ones((len(sd.pr), 1))])
    N = len(nl)
    lam = np.full(N, np.nan)
    sig_x = np.full(N, np.nan)
    used = 0
    f = rig.left.f
    sig_n = SIGMA_PX / f
    for k in np.unique(sd.frame):
        idx = np.where((sd.frame == k) & np.isfinite(sd.pl2[:, 0]))[0]
        if len(idx) < MIN_PAIR_FEATS:
            continue
        a = nl[idx]
        b = np.hstack([rig.left.normalize(sd.pl2[idx]), np.ones((len(idx), 1))])
        g = _pair_geometry(a, b, f)
        if g is None:
            continue
        R, t, inl = g
        t = t / np.linalg.norm(t)
        ia, ib, ii = a[inl], b[inl], idx[inl]
        ra = (R @ ia.T).T
        # derotated parallax in px: component of b not explained by R a, perpendicular to epipole
        par = np.linalg.norm(ib[:, :2] - ra[:, :2] / ra[:, 2:3], axis=1) * f
        if len(ii) < MIN_PAIR_FEATS or np.median(par) < MIN_PARALLAX_PX:
            continue
        # least-squares depths: lam * R a + t = mu * b  (unit |t|)
        A = np.stack([ra, -ib], axis=2)  # (n,3,2)
        AtA = np.einsum("nij,nik->njk", A, A)
        Atb = -np.einsum("nij,i->nj", A, t)
        x = np.linalg.solve(AtA + 1e-12 * np.eye(2), Atb[:, :, None])[:, :, 0]
        gap = np.linalg.norm(np.einsum("nij,nj->ni", A, x) + t, axis=1)
        ang = np.degrees(np.arccos(np.clip(np.abs(ia @ t) / np.linalg.norm(ia, axis=1), 0, 1)))
        ok = (x[:, 0] > 0) & (x[:, 1] > 0) & (gap < 6 * sig_n * x[:, 1]) & (ang > MIN_RAY_ANGLE_DEG)
        used += 1
        lam[ii[ok]] = x[ok, 0]
        sinpsi = np.sin(np.radians(ang[ok]))
        sig_x[ii[ok]] = np.sqrt(2.0) * sig_n / np.maximum(sinpsi, 1e-3)
    mono_ok = np.isfinite(lam)
    return Prepared(rig, sd.frame, nl, nr, mono_ok, lam, sig_x, sd.frame, len(sd.t_key_s), used)


def _rect(prep: Prepared, R_lr: np.ndarray, sel=None):
    Rr = rect_rotation(prep.rig.t_lr)
    nl, nr = (prep.nl, prep.nr) if sel is None else (prep.nl[sel], prep.nr[sel])
    vl = nl @ Rr.T
    vr = nr @ (Rr @ R_lr).T
    return vl, vr


def _tukey(r: np.ndarray, s: float, c: float = 4.685) -> np.ndarray:
    u = r / (c * s)
    return np.where(np.abs(u) < 1, (1 - u * u) ** 2, 0.0)


def _mad(r: np.ndarray, w: np.ndarray) -> float:
    keep = w > 0
    if keep.sum() < 5:
        return 1.0
    return 1.4826 * float(np.median(np.abs(r[keep])))


def epipolar_fit(prep: Prepared, R_cand: np.ndarray, wts: np.ndarray, free_y: bool = True, n_outer: int = 3):
    """Stage 1. Returns (w_rad (3,) error of R_cand, corrected R, rms residual px, cov or None)."""
    f = prep.rig.f_px
    R = R_cand.copy()
    w_tot = np.zeros(3)
    cov = None
    rms = np.nan
    for _ in range(n_outer):
        vl, vr = _rect(prep, R)
        xl, yl = vl[:, 0] / vl[:, 2], vl[:, 1] / vl[:, 2]
        yr = vr[:, 1] / vr[:, 2]
        dy = f * (yl - yr)
        B = f * np.stack([1 + yl ** 2, -xl, -xl * yl], axis=1)
        cols = [0, 1, 2] if free_y else [0, 1]  # basis order is (w_x, w_z, w_y)
        B = B[:, cols]
        rob = np.ones(len(dy))
        w = np.zeros(len(cols))
        for _ in range(12):
            ww = wts * rob
            if ww.sum() < 20:
                return w_tot, R, np.nan, None
            Wb = B * ww[:, None]
            w = np.linalg.solve(B.T @ Wb + 1e-9 * np.eye(len(cols)), Wb.T @ dy)
            res = dy - B @ w
            s = max(_mad(res, wts > 0), 0.05)
            rob = _tukey(res, s)
        ww = wts * rob
        cov = np.linalg.inv(B.T @ (B * ww[:, None])) * s ** 2
        rms = float(np.sqrt(np.sum(ww * res ** 2) / np.sum(ww)))
        step = np.zeros(3)
        step[[0, 2, 1][: len(cols)]] = w
        w_tot += step
        R = exp_so3(-step) @ R
    return w_tot, R, rms, cov


def motion_intercept(prep: Prepared, R_lr: np.ndarray, wts: np.ndarray, n_iter: int = 15):
    """Stage 2. Returns (c_px, se_px, n_feats, n_pairs): shared disparity intercept c = -f w_y."""
    sel = np.where(prep.mono_ok & (wts > 0))[0]
    if len(sel) < 40:
        return np.nan, np.nan, len(sel), 0
    f = prep.rig.f_px
    vl, vr = _rect(prep, R_lr, sel)
    xl, yl = vl[:, 0] / vl[:, 2], vl[:, 1] / vl[:, 2]
    d = f * (xl - vr[:, 0] / vr[:, 2])
    m = 1 + xl ** 2
    Rr = rect_rotation(prep.rig.t_lr)
    Zr = (Rr[2] @ (prep.nl[sel] * prep.lam[sel, None]).T)  # rectified-left depth, unit-|t| scale
    xinv = 1.0 / Zr
    sx = prep.sig_x[sel]
    pid, pidx = np.unique(prep.pair[sel], return_inverse=True)
    P = len(pid)
    w0 = wts[sel]
    rob = np.ones(len(sel))
    c, alpha = 0.0, np.ones(P)
    # init alpha per pair from the median ratio
    for p in range(P):
        k = pidx == p
        alpha[p] = np.median(d[k] / xinv[k])
    for _ in range(n_iter):
        sig2 = SIGMA_PX ** 2 + (alpha[pidx] * sx) ** 2
        w = w0 * rob / sig2
        Sxx = np.bincount(pidx, w * xinv ** 2, P)
        Sxd = np.bincount(pidx, w * xinv * d, P)
        Sxm = np.bincount(pidx, w * xinv * m, P)
        good = Sxx > 0
        ratio_d = np.where(good, Sxd / np.where(good, Sxx, 1), 0)[pidx]
        ratio_m = np.where(good, Sxm / np.where(good, Sxx, 1), 0)[pidx]
        dt = d - ratio_d * xinv
        mt = m - ratio_m * xinv
        den = np.sum(w * mt ** 2)
        if den <= 0:
            return np.nan, np.nan, len(sel), P
        c = float(np.sum(w * mt * dt) / den)
        alpha = np.where(good, (Sxd - c * Sxm) / np.where(good, Sxx, 1), alpha)
        res = (d - alpha[pidx] * xinv - c * m) * np.sqrt(1.0 / sig2)
        s = max(_mad(res, w0 * rob), 0.3)
        rob = _tukey(res, s)
    w = w0 * rob / (SIGMA_PX ** 2 + (alpha[pidx] * sx) ** 2)
    den = np.sum(w * mt ** 2)
    se = float(np.sqrt(1.0 / den)) * max(s, 1.0)
    return c, se, int((rob > 0).sum()), P


@dataclass
class Result:
    epi_deg: np.ndarray     # (3,) static epipolar 3-parameter fit (w_x, w_y, w_z)
    mot_deg: np.ndarray     # (3,) x,z from epipolar, y from the motion intercept
    fused_deg: np.ndarray   # (3,) y = inverse-variance fusion of the two, x,z refit with it
    se_deg: dict            # method -> (3,) bootstrap standard deviation
    ci_deg: np.ndarray      # (3,2) 90% block-bootstrap interval of fused_deg
    consistent: bool        # |y_epi - y_mot| < 2 sqrt(se_epi^2 + se_mot^2)
    n_feats: int
    n_mono: int
    n_pairs: int
    dy_rms_px: float


def _methods(prep: Prepared, rig_cand: Rig, wts: np.ndarray, n_alt: int = 2):
    """Returns (epi (3,), mot (3,), mono feats, pairs, rms) in radians, order (x, y, z)."""
    f = prep.rig.f_px
    epi, _, rms, _ = epipolar_fit(prep, rig_cand.R_lr, wts, free_y=True)
    wy, w_xz, nm, npairs = 0.0, np.zeros(3), 0, 0
    for _ in range(n_alt):
        R_noy = exp_so3(-np.array([0.0, wy, 0.0])) @ rig_cand.R_lr
        w_xz = epipolar_fit(prep, R_noy, wts, free_y=False)[0]
        R_xz = exp_so3(-np.array([w_xz[0], 0.0, w_xz[2]])) @ rig_cand.R_lr
        c, _, nm, npairs = motion_intercept(prep, R_xz, wts)
        if not np.isfinite(c):
            wy = np.nan
            break
        wy = -c / f
    return epi, np.array([w_xz[0], wy, w_xz[2]]), nm, npairs, rms


def _fuse(prep: Prepared, rig_cand: Rig, wts: np.ndarray, epi, mot, w_epi: float, w_mot: float) -> np.ndarray:
    ys = [(epi[1], w_epi), (mot[1], w_mot)]
    ys = [(v, w) for v, w in ys if np.isfinite(v)]
    wy = sum(v * w for v, w in ys) / sum(w for _, w in ys)
    R_noy = exp_so3(-np.array([0.0, wy, 0.0])) @ rig_cand.R_lr
    w_xz = epipolar_fit(prep, R_noy, wts, free_y=False)[0]
    return np.array([w_xz[0], wy, w_xz[2]])


def estimate(prep: Prepared, rig_cand: Rig, n_boot: int = 60, block: int = 4, seed: int = 0,
             keyframes: np.ndarray | None = None) -> Result:
    """Block bootstrap over keyframes (blocks of ``block`` consecutive keyframes) for the CIs.

    ``keyframes`` restricts the estimate to a subset (e.g. a shorter time window).
    """
    keys = np.unique(prep.frame) if keyframes is None else np.asarray(keyframes)
    nk = len(keys)
    base_w = np.isin(prep.frame, keys).astype(float)
    epi, mot, nm, npairs, rms = _methods(prep, rig_cand, base_w)
    rng = np.random.default_rng(seed)
    lut_size = prep.frame.max() + 1
    draws = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(nk - block + 1, 1), int(np.ceil(nk / block)))
        pick = np.concatenate([keys[s:s + block] for s in starts])
        lut = np.bincount(pick, minlength=lut_size).astype(float)
        draws.append((lut[prep.frame],) + _methods(prep, rig_cand, lut[prep.frame])[:2])
    E_ = np.array([d[1] for d in draws]) if draws else np.full((1, 3), np.nan)
    M_ = np.array([d[2] for d in draws]) if draws else np.full((1, 3), np.nan)
    se_e, se_m = np.nanstd(E_, axis=0), np.nanstd(M_, axis=0)
    w_e, w_m = 1.0 / max(se_e[1], 1e-6) ** 2, 1.0 / max(se_m[1], 1e-6) ** 2
    fused = _fuse(prep, rig_cand, base_w, epi, mot, w_e, w_m)
    F_ = np.array([_fuse(prep, rig_cand, d[0], d[1], d[2], w_e, w_m) for d in draws]) if draws else np.full((1, 3), np.nan)
    ci = np.nanpercentile(F_, [5, 95], axis=0).T
    n_in = int(base_w.sum())
    cons = bool(np.isfinite(mot[1]) and abs(epi[1] - mot[1]) < 2 * np.hypot(se_e[1], se_m[1]))
    deg = np.rad2deg
    return Result(deg(epi), deg(mot), deg(fused), {"epi": deg(se_e), "mot": deg(se_m), "fused": deg(np.nanstd(F_, axis=0))},
                  deg(ci), cons, n_in, nm, npairs, rms)


def implied(rig: Rig, w_deg: np.ndarray) -> dict:
    """Depth/disparity consequences of an estimated rotation error."""
    f, B = rig.f_px, rig.baseline_m
    w = np.deg2rad(w_deg)
    bias_px = f * abs(w[1])  # disparity bias at the image centre
    g = np.linspace(-0.5, 0.5, 11)
    X, Y = np.meshgrid(g, g)
    dy = f * (w[0] * (1 + Y ** 2) - w[2] * X - w[1] * X * Y)
    return dict(bias_px=bias_px, dy_rms_px=float(np.sqrt(np.mean(dy ** 2))),
                err_3m=bias_px * 3.0 / (f * B), err_8m=bias_px * 8.0 / (f * B))


T_CHECK_8M, T_RECAL_3M = 0.10, 0.10   # relative depth error that triggers CHECK (at 8 m) / RECAL (at 3 m)
T_CHECK_DY_PX, T_RECAL_DY_PX = 0.7, 1.5  # RMS vertical disparity over the central field


def verdict(rig: Rig, w_deg: np.ndarray) -> str:
    im = implied(rig, w_deg)
    if im["err_3m"] >= T_RECAL_3M or im["dy_rms_px"] >= T_RECAL_DY_PX:
        return "RECALIBRATE"
    if im["err_8m"] >= T_CHECK_8M or im["dy_rms_px"] >= T_CHECK_DY_PX:
        return "CHECK"
    return "OK"
