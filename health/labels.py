"""Drift labels for the localization-health model (definitions: see README.md).

Everything here is NON-causal on purpose: it uses ground truth and future frames to label.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate as ev  # noqa: E402  (hand_eye, move_gt, to_mats, load_tum)

WINDOW_S = 2.0          # relative-motion window W
EST_TOL_S = 0.02        # frame stamp <-> estimate stamp association
GT_TOL_S = 0.02         # frame stamp <-> GT stamp association (as evaluate.py)
WIN_END_TOL_S = 0.06    # window end must land within this of t + W (frames are ~80 ms apart)
GT_GAP_S = 0.25         # GT / estimate stamp gap above this inside a window invalidates it
MIN_VALID_FRAC = 0.8    # fraction of frames in the horizon with a valid window needed for a negative label


@dataclass
class Drift:
    """Per-frame drift of the window [t_i, t_i + W]; NaN where the window is invalid."""
    t: np.ndarray
    trans_mps: np.ndarray    # |translation error of the relative motion| / W, metres per second
    rot_dps: np.ndarray      # rotation error angle / W, degrees per second
    speed_mps: np.ndarray    # GT path length in the window / W
    valid: np.ndarray        # bool


def _nearest(stamps: np.ndarray, t: np.ndarray, tol: float) -> np.ndarray:
    """Index of the nearest stamp within tol, else -1 (stamps sorted)."""
    j = np.clip(np.searchsorted(stamps, t), 1, len(stamps) - 1)
    j = np.where(np.abs(stamps[j - 1] - t) < np.abs(stamps[j] - t), j - 1, j)
    return np.where(np.abs(stamps[j] - t) <= tol, j, -1)


def hand_eye_x(gt_tum: np.ndarray, est_tum: np.ndarray) -> np.ndarray:
    """T_body_cam from the whole run (evaluate.hand_eye, first_s=0)."""
    return ev.hand_eye(gt_tum, est_tum, 0.0)[0]


def window_drift(frame_t: np.ndarray, est_tum: np.ndarray, gt_tum: np.ndarray,
                 X: np.ndarray | None, window_s: float = WINDOW_S) -> Drift:
    """Drift of est vs GT relative motion over [t_i, t_i + window_s] for every frame time.

    err = inv(dGT) dEST with d* = inv(P_i) P_k in the camera frame, so no world alignment is
    needed, only the body-to-camera X (GT is moved onto the camera: T_world_body @ X). Windows
    whose end frame is missing, or that contain a GT / estimate stamp gap > GT_GAP_S, are invalid."""
    n = len(frame_t)
    out = Drift(frame_t, np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool))
    if len(est_tum) < 2 or len(gt_tum) < 2:
        return out
    gt = ev.move_gt(gt_tum, X) if X is not None else gt_tum
    te, tg = est_tum[:, 0], gt[:, 0]
    E, G = ev.to_mats(est_tum), ev.to_mats(gt)
    i_est = _nearest(te, frame_t, EST_TOL_S)
    k_frame = _nearest(frame_t, frame_t + window_s, WIN_END_TOL_S)
    for i in range(n):
        k = k_frame[i]
        if k < 0 or i_est[i] < 0 or i_est[k] < 0:
            continue
        ei, ek = i_est[i], i_est[k]
        gi, gk = _nearest(tg, np.array([frame_t[i], frame_t[k]]), GT_TOL_S)
        if gi < 0 or gk < 0 or gk <= gi:
            continue
        if np.max(np.diff(tg[gi:gk + 1])) > GT_GAP_S or np.max(np.diff(te[ei:ek + 1])) > GT_GAP_S:
            continue
        dt = frame_t[k] - frame_t[i]
        d_gt = np.linalg.inv(G[gi]) @ G[gk]
        d_est = np.linalg.inv(E[ei]) @ E[ek]
        err = np.linalg.inv(d_gt) @ d_est
        out.trans_mps[i] = np.linalg.norm(err[:3, 3]) / dt
        out.rot_dps[i] = ev.rot_angle_deg(err[:3, :3]) / dt
        out.speed_mps[i] = np.sum(np.linalg.norm(np.diff(G[gi:gk + 1, :3, 3], axis=0), axis=1)) / dt
        out.valid[i] = True
    return out


@dataclass
class Labels:
    y: np.ndarray            # float: 1 / 0 / NaN (undefined)
    exceed: np.ndarray       # bool per frame: window starting here exceeds the threshold
    recovery: np.ndarray     # bool per frame: a gate recovery happened at this frame
    onset: np.ndarray        # bool per frame: exceed | recovery (event candidates)


def make_labels(frame_t: np.ndarray, drift: Drift, recoveries: np.ndarray, horizon_s: float,
                thr_trans_mps: float, thr_rot_dps: float | None = None,
                use_recovery: bool = True) -> Labels:
    """y_i = 1 iff some frame j with t_i <= t_j <= t_i + H has a window that exceeds the
    threshold (translation > thr_trans_mps, or rotation > thr_rot_dps if given) or, with
    use_recovery, a gate recovery at j with t_i < t_j <= t_i + H.
    y_i = 0 iff not 1 and >= MIN_VALID_FRAC of the frames in [t_i, t_i+H] have a valid window and
    the run lasts to t_i + H; otherwise NaN."""
    n = len(frame_t)
    with np.errstate(invalid="ignore"):
        exceed = drift.valid & (drift.trans_mps > thr_trans_mps)
        if thr_rot_dps is not None:
            exceed |= drift.valid & (drift.rot_dps > thr_rot_dps)
    rec = np.zeros(n, bool)
    rec[1:] = np.diff(recoveries) > 0
    onset = exceed | rec if use_recovery else exceed.copy()
    hi = np.searchsorted(frame_t, frame_t + horizon_s, side="right")  # exclusive end index
    lo = np.arange(n)
    c_ex = np.concatenate([[0], np.cumsum(exceed)])
    c_rec = np.concatenate([[0], np.cumsum(rec)])
    c_val = np.concatenate([[0], np.cumsum(drift.valid)])
    pos = (c_ex[hi] - c_ex[lo]) > 0
    if use_recovery:
        pos |= (c_rec[hi] - c_rec[lo + 1]) > 0
    cnt = np.maximum(hi - lo, 1)
    covered = ((c_val[hi] - c_val[lo]) / cnt >= MIN_VALID_FRAC) & (frame_t + horizon_s <= frame_t[-1])
    y = np.where(pos, 1.0, np.where(covered, 0.0, np.nan))
    return Labels(y, exceed, rec, onset)


def event_onsets(frame_t: np.ndarray, onset: np.ndarray, cluster_gap_s: float) -> np.ndarray:
    """Event onset times: onset-frame stamps clustered; a new event starts when the gap to the
    previous onset frame exceeds cluster_gap_s."""
    ts = frame_t[onset]
    if len(ts) == 0:
        return ts
    return ts[np.concatenate([[True], np.diff(ts) > cluster_gap_s])]
