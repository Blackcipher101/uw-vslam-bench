"""Online-causal features from the run_bag --frame-log CSV.

Row i uses only rows <= i (windows are (t_i - w, t_i]); tests/test_health.py checks this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

WINDOWS_S = (1.0, 3.0, 5.0)
ROLL_SIGNALS = ("tracked", "stereo", "landmark_factors", "gated_frac", "img_mean", "img_std")
RAW_COLS = ("tracked", "stereo", "new_landmarks", "landmark_factors", "gated", "candidates", "vision_used",
            "vision_healthy", "consecutive_gated", "n_features", "img_mean", "img_std", "gated_frac")
MOTION_LAG_S = 0.3      # backward-difference baseline for velocity / angular rate / acceleration
MOTION_SIGNALS = ("ang_rate_dps", "speed_mps", "accel_mps2", "yaw_rate_abs_dps")
NO_RECOVERY_S = 120.0   # time_since_recovery value before the first recovery (cap for all values)
FRAME_LOG_COLS = ("t", "tracked", "stereo", "new_landmarks", "landmark_factors", "gated", "candidates",
                  "vision_used", "vision_healthy", "consecutive_gated", "recoveries", "dvl_factors", "dvl_gated",
                  "img_mean", "img_std", "n_features")


def load_frame_log(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in FRAME_LOG_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: frame log lacks columns {missing}")
    df = df.sort_values("t", kind="stable").drop_duplicates("t").reset_index(drop=True)
    return df


def _rolling(t: np.ndarray, x: np.ndarray, w: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal mean / min / least-squares slope (units per second) of x over (t_i - w, t_i]."""
    n = len(t)
    lo = np.searchsorted(t, t - w, side="right")
    tc = t - t[0]
    cs = [np.concatenate([[0.0], np.cumsum(a)]) for a in (x, tc, tc * x, tc * tc)]
    idx = np.arange(n) + 1
    cnt = (idx - lo).astype(float)
    sx, st, stx, stt = (c[idx] - c[lo] for c in cs)
    mean = sx / cnt
    var_t = stt - st * st / cnt
    cov = stx - st * sx / cnt
    slope = np.where((cnt >= 3) & (var_t > 1e-9), cov / np.where(var_t > 1e-9, var_t, 1.0), 0.0)
    mn = np.array([x[lo[i]:i + 1].min() for i in range(n)])
    return mean, mn, slope


def _rolling_max(t: np.ndarray, x: np.ndarray, w: float) -> np.ndarray:
    lo = np.searchsorted(t, t - w, side="right")
    return np.array([x[lo[i]:i + 1].max() for i in range(len(t))])


def motion_features(t: np.ndarray, est_tum: np.ndarray) -> dict[str, np.ndarray]:
    """Motion from the estimator's own trajectory, causal: frame i uses the newest estimate pose with
    stamp <= t_i + 20 ms and backward differences over ~MOTION_LAG_S. World-frame quantities (the VIO
    world is gravity aligned): angular rate = angle of R_i R_j^T / dt (deg/s), yaw rate = world-z
    component of its rotation vector, speed = |p_i - p_j| / dt, acceleration = |v_i - v_j| / (t_i - t_j).
    Frames before the first pose or the first lag get 0."""
    n = len(t)
    te = est_tum[:, 0]
    idx = np.searchsorted(te, t + 0.02, side="right") - 1
    have = idx >= 0
    idx = np.maximum(idx, 0)
    p = est_tum[idx, 1:4]
    R = Rotation.from_quat(est_tum[idx, 4:8])
    j = np.minimum(np.searchsorted(t, t - MOTION_LAG_S, side="left"), np.arange(n))
    dt = t - t[j]
    ok = have & have[j] & (dt > 1e-3)
    sdt = np.where(ok, dt, 1.0)
    v = np.where(ok[:, None], (p - p[j]) / sdt[:, None], 0.0)
    rv = (R * R[j].inv()).as_rotvec()
    ang = np.where(ok, np.degrees(np.linalg.norm(rv, axis=1)) / sdt, 0.0)
    yaw = np.where(ok, np.abs(np.degrees(rv[:, 2])) / sdt, 0.0)
    acc = np.where(ok, np.linalg.norm(v - v[j], axis=1) / sdt, 0.0)
    sig = {"ang_rate_dps": ang, "speed_mps": np.linalg.norm(v, axis=1), "accel_mps2": acc, "yaw_rate_abs_dps": yaw}
    out = {f"motion_{k}": x for k, x in sig.items()}
    for k, x in sig.items():
        for w in WINDOWS_S:
            out[f"motion_{k}_mean_{w:g}s"] = _rolling(t, x, w)[0]
            out[f"motion_{k}_max_{w:g}s"] = _rolling_max(t, x, w)
    return out


def build_features(df: pd.DataFrame, est_tum: np.ndarray | None = None) -> pd.DataFrame:
    """Feature table, one row per frame-log row, same order. All values finite. v0 = frame-log columns;
    with est_tum (the run's own TUM trajectory) the motion_* columns (v1) are appended."""
    t = df["t"].to_numpy(float)
    d = df.copy()
    d["gated_frac"] = d["gated"] / np.maximum(d["landmark_factors"] + d["gated"], 1)
    feats = {c: d[c].to_numpy(float) for c in RAW_COLS}
    for s in ROLL_SIGNALS:
        x = d[s].to_numpy(float)
        for w in WINDOWS_S:
            m, mn, sl = _rolling(t, x, w)
            feats[f"{s}_mean_{w:g}s"] = m
            feats[f"{s}_min_{w:g}s"] = mn
            feats[f"{s}_slope_{w:g}s"] = sl
    rec = np.zeros(len(t), bool)
    rec[1:] = np.diff(d["recoveries"].to_numpy(float)) > 0
    last = np.maximum.accumulate(np.where(rec, t, -np.inf))
    feats["time_since_recovery_s"] = np.minimum(t - last, NO_RECOVERY_S)  # -inf -> capped
    if est_tum is not None:
        feats.update(motion_features(t, est_tum))
    out = pd.DataFrame(feats)
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError("non-finite feature values (NaN/Inf in the frame log)")
    return out
