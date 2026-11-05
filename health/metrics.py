"""Frame-level and event-level metrics for a per-frame risk score."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ALARM_MERGE_S = 1.0     # flagged frames closer than this form one alarm episode


@dataclass
class SeqEval:
    """One run: frame stamps, risk score, label y (NaN = undefined) and event onset stamps."""
    t: np.ndarray
    score: np.ndarray
    y: np.ndarray
    events: np.ndarray
    horizon_s: float


def frame_metrics(seqs: list[SeqEval]) -> dict:
    y = np.concatenate([s.y for s in seqs])
    sc = np.concatenate([s.score for s in seqs])
    ok = np.isfinite(y)
    y, sc = y[ok], sc[ok]
    out = {"n": int(len(y)), "pos_rate": float(y.mean()) if len(y) else float("nan"),
           "auroc": float("nan"), "auprc": float("nan")}
    if len(y) and 0 < y.sum() < len(y):
        out["auroc"] = float(roc_auc_score(y, sc))
        out["auprc"] = float(average_precision_score(y, sc))
    return out


def _episodes(t: np.ndarray, flag: np.ndarray) -> list[tuple[float, float]]:
    ts = t[flag]
    if len(ts) == 0:
        return []
    cut = np.nonzero(np.diff(ts) > ALARM_MERGE_S)[0]
    starts = np.concatenate([[0], cut + 1])
    ends = np.concatenate([cut, [len(ts) - 1]])
    return [(ts[a], ts[b]) for a, b in zip(starts, ends)]


def alarm_stats(seqs: list[SeqEval], thr: float) -> dict:
    """Alarm = score >= thr. False alarm = alarm episode containing no y == 1 frame and at least one
    y == 0 frame; rate per minute of defined-negative time. An event is detected if an alarm frame lies
    in [onset - H, onset]; lead = onset - first such alarm (<= H); the rest are missed."""
    fa = neg_s = 0.0
    n_ev = det = 0
    leads: list[float] = []
    for s in seqs:
        flag = s.score >= thr
        dt = float(np.median(np.diff(s.t))) if len(s.t) > 1 else 0.0
        neg_s += float(np.sum(s.y == 0)) * dt
        for a, b in _episodes(s.t, flag):
            seg = s.y[np.searchsorted(s.t, a, "left"):np.searchsorted(s.t, b, "right")]
            if np.any(seg == 1):
                continue
            if np.any(seg == 0):
                fa += 1
        for on in s.events:
            n_ev += 1
            m = flag & (s.t >= on - s.horizon_s) & (s.t <= on)
            if m.any():
                det += 1
                leads.append(float(on - s.t[m][0]))
    frame_tp = frame_pos = 0
    for s in seqs:
        pos = s.y == 1
        frame_pos += int(pos.sum())
        frame_tp += int(np.sum(pos & (s.score >= thr)))
    return {"thr": float(thr), "fa_per_min": fa / (neg_s / 60) if neg_s > 0 else float("nan"),
            "events": n_ev, "detected": det, "missed": n_ev - det,
            "event_recall": det / n_ev if n_ev else float("nan"),
            "median_lead_s": float(np.median(leads)) if leads else float("nan"),
            "frame_recall": frame_tp / frame_pos if frame_pos else float("nan")}


def choose_threshold(seqs: list[SeqEval], target_fa_per_min: float, n_grid: int = 300) -> float:
    """Smallest threshold whose false-alarm rate on these runs is <= target (most sensitive allowed)."""
    sc = np.concatenate([s.score for s in seqs])
    u = np.unique(sc)
    cand = u if len(u) <= n_grid else np.unique(np.quantile(sc, np.linspace(0, 1, n_grid)))
    lo, hi = 0, len(cand)  # false-alarm rate is (near-)monotone in the threshold: bisect
    while lo < hi:
        mid = (lo + hi) // 2
        r = alarm_stats(seqs, float(cand[mid]))["fa_per_min"]
        if np.isfinite(r) and r <= target_fa_per_min:
            hi = mid
        else:
            lo = mid + 1
    return float(cand[lo]) if lo < len(cand) else float(np.nextafter(cand[-1], np.inf))  # else never alarm
