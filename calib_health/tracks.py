"""Calibration-independent feature extraction: left<->right matches and left temporal tracks.

Nothing here uses the stereo extrinsic, so one extraction serves every candidate calibration.
Output (``StretchData``): per keyframe k, stereo matches (raw pixels pl, pr) and, for those
features, the left pixel ``pl2`` at frame k+gap (NaN where the temporal track failed).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

LK = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01))


@dataclass
class StretchData:
    frame: np.ndarray  # (N,) keyframe index of each feature
    pl: np.ndarray     # (N,2) left pixel at keyframe
    pr: np.ndarray     # (N,2) right pixel at keyframe (stereo match)
    pl2: np.ndarray    # (N,2) left pixel at keyframe+gap, NaN if untracked
    t_key_s: np.ndarray  # (n_keyframes,) left stamp of the keyframe, s since bag start
    t_gap_s: np.ndarray  # (n_keyframes,) left stamp of keyframe+gap (NaN if beyond the stretch)
    size: tuple[int, int]  # (width, height)

    def save(self, path: Path) -> None:
        np.savez_compressed(path, frame=self.frame, pl=self.pl, pr=self.pr, pl2=self.pl2,
                            t_key_s=self.t_key_s, t_gap_s=self.t_gap_s, size=np.array(self.size))

    @staticmethod
    def load(path: Path) -> "StretchData":
        z = np.load(path)
        return StretchData(z["frame"], z["pl"], z["pr"], z["pl2"], z["t_key_s"], z["t_gap_s"], tuple(int(v) for v in z["size"]))


def _lk(a, b, p, init=None):
    q, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None if init is None else init.copy(), **LK,
                                        flags=0 if init is None else cv2.OPTFLOW_USE_INITIAL_FLOW)
    return q, st.ravel().astype(bool)


def stereo_matches(l_img: np.ndarray, r_img: np.ndarray, max_corners: int, border: int,
                   fb_px: float, max_abs_dy_px: float, max_disp_px: float, min_disp_px: float = 0.5):
    """Left corners tracked into the right image (forward-backward checked, loose geometry gate)."""
    h, w = l_img.shape
    mask = np.zeros_like(l_img)
    mask[border:h - border, border:w - border] = 255
    c = cv2.goodFeaturesToTrack(l_img, max_corners, 0.01, 8, mask=mask, blockSize=7)
    if c is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    p = c.reshape(-1, 1, 2).astype(np.float32)
    q, ok = _lk(l_img, r_img, p)
    back, ok2 = _lk(r_img, l_img, q, p)
    fb = np.linalg.norm((back - p).reshape(-1, 2), axis=1)
    d = (p - q).reshape(-1, 2)  # left u minus right u; v difference is the vertical residual
    inb = (q[:, 0, 0] > border) & (q[:, 0, 0] < w - border) & (q[:, 0, 1] > border) & (q[:, 0, 1] < h - border)
    keep = ok & ok2 & (fb < fb_px) & inb & (d[:, 0] > min_disp_px) & (d[:, 0] < max_disp_px) & (np.abs(d[:, 1]) < max_abs_dy_px)
    return p.reshape(-1, 2)[keep], q.reshape(-1, 2)[keep]


def extract(frames: Iterable[tuple[float, np.ndarray, np.ndarray]], key_every: int = 6, gap: int = 6,
            max_corners: int = 800, border: int = 20, fb_px: float = 0.3, max_abs_dy_px: float = 12.0,
            max_disp_px: float = 100.0, clahe_clip: float = 2.0, temporal_fb_px: float = 0.5,
            min_disp_px: float = 0.5) -> StretchData:
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    left_hist: dict[int, np.ndarray] = {}
    t_gap: dict[int, float] = {}
    pending: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    key_idx: dict[int, int] = {}
    out_f, out_pl, out_pr, out_pl2, t_key = [], [], [], [], []
    size = (0, 0)
    for i, (t, l_img, r_img) in enumerate(frames):
        size = (l_img.shape[1], l_img.shape[0])
        l_img, r_img = clahe.apply(l_img), clahe.apply(r_img)
        left_hist[i] = l_img
        if i - gap in pending:
            k = i - gap
            t_gap[key_idx[k]] = t
            pl, pr = pending.pop(k)
            q, ok = _lk(left_hist[k], l_img, pl.reshape(-1, 1, 2).astype(np.float32))
            back, ok2 = _lk(l_img, left_hist[k], q, pl.reshape(-1, 1, 2).astype(np.float32))
            fb = np.linalg.norm((back - pl.reshape(-1, 1, 2)).reshape(-1, 2), axis=1)
            good = ok & ok2 & (fb < temporal_fb_px)
            pl2 = np.where(good[:, None], q.reshape(-1, 2), np.nan)
            out_f.append(np.full(len(pl), key_idx[k]))
            out_pl.append(pl); out_pr.append(pr); out_pl2.append(pl2)
        if i % key_every == 0:
            pl, pr = stereo_matches(l_img, r_img, max_corners, border, fb_px, max_abs_dy_px, max_disp_px,
                                    min_disp_px)
            pending[i] = (pl, pr)
            key_idx[i] = len(t_key)
            t_key.append(t)
        for k in [k for k in left_hist if k < i - gap]:
            del left_hist[k]
    cat = lambda a, shape: np.concatenate(a) if a else np.zeros(shape)
    return StretchData(cat(out_f, (0,)).astype(int), cat(out_pl, (0, 2)), cat(out_pr, (0, 2)),
                       cat(out_pl2, (0, 2)), np.asarray(t_key),
                       np.array([t_gap.get(j, np.nan) for j in range(len(t_key))]), size)
