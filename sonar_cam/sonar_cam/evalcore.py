"""Shared evaluation core: sonar points (sliding window or same-time) vs the stereo depth cache.

Residual convention: ``res = z_stereo - z_sonar`` where both are the z coordinate (metres) in the *rectified left*
frame at the pixel the sonar point projects to; ``z_sonar`` is the sonar point's own rectified z.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Bag, Trajectory, load_camchain, load_svin_rig, sonar_in_cam0
from .geometry import ping_range, ping_range_contrast, sonar_points, transform, invert


@dataclass
class Pairs:
    ping_idx: np.ndarray   # index into the ping arrays
    img_idx: np.ndarray    # index into the cache frames
    z_sonar: np.ndarray
    z_stereo: np.ndarray
    range_m: np.ndarray    # sonar range of the ping
    uv: np.ndarray         # (N,2) working-resolution pixel of the sonar point in the rectified left image
    age_s: np.ndarray      # t_img - (t_ping + dt)

    @property
    def res(self) -> np.ndarray:
        return self.z_stereo - self.z_sonar


class Dataset:
    def __init__(self, cache_path: str, traj_path: str, min_range_m: float = 0.0, min_intensity: int = 10,
                 pings=None, picker: str = "svin", min_prominence: float = 30.0):
        c = np.load(cache_path)
        self.t_img = c["t"]
        self.depth = c["depth"]
        self.block = int(c["block"])
        self.f, self.cx, self.cy = float(c["f"]), float(c["cx"]), float(c["cy"])
        self.R1 = c["R1"]
        self.baseline_m = float(c["baseline_m"])
        pings = pings if pings is not None else Bag().pings()
        self.t_ping = np.array([p.t for p in pings])
        self.head_deg = np.array([p.head_deg for p in pings])
        if picker == "svin":
            rp = [ping_range(p.intensity, p.max_range_m, min_intensity=min_intensity, min_accept_m=min_range_m)
                  for p in pings]
        elif picker == "contrast":
            rp = [ping_range_contrast(p.intensity, p.max_range_m, min_prominence=min_prominence,
                                      min_accept_m=max(min_range_m, 0.3)) for p in pings]
        else:
            raise ValueError(f"unknown picker {picker!r}")
        self.range_m = np.array([r[0] for r in rp])
        self.peak = np.array([r[1] for r in rp])
        self.traj = Trajectory(traj_path)
        self.T_c0_So = sonar_in_cam0(load_svin_rig())
        self.size_work = (self.depth.shape[2] * self.block, self.depth.shape[1] * self.block)

    def lookup(self, img_idx: np.ndarray, uv: np.ndarray) -> np.ndarray:
        """Stereo depth at working-resolution pixels (NaN outside image / invalid)."""
        col = np.floor(np.nan_to_num(uv[:, 0], nan=-1.0) / self.block).astype(int)
        row = np.floor(np.nan_to_num(uv[:, 1], nan=-1.0) / self.block).astype(int)
        h, w = self.depth.shape[1:]
        ok = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        z = np.full(len(uv), np.nan)
        z[ok] = self.depth[img_idx[ok], row[ok], col[ok]]
        return z

    def project_rect(self, P_c0: np.ndarray):
        Pr = P_c0 @ self.R1.T
        with np.errstate(all="ignore"):
            z = np.where(Pr[:, 2] > 0.05, Pr[:, 2], np.nan)
            uv = np.stack([self.f * Pr[:, 0] / z + self.cx, self.f * Pr[:, 1] / z + self.cy], axis=1)
        return z, uv

    def window_pairs(self, T_c0_So: np.ndarray, dt_s: float = 0.0, window_s: float = 4.2,
                     future_s: float = 0.0, heading_sign: float = 1.0) -> Pairs:
        """Sonar points from pings with true time t_ping+dt in (t_img-window, t_img+future], placed in the world
        with the interpolated T_w_c0 at that time, re-projected to each cached image using that image's pose."""
        good = np.isfinite(self.range_m)
        t_true = self.t_ping + dt_s
        good &= self.traj.valid(t_true)
        idx = np.nonzero(good)[0]
        order = idx[np.argsort(t_true[idx])]
        tt = t_true[order]
        T_w_c_ping = self.traj.poses(tt)
        P_so = sonar_points(self.range_m[order], heading_sign * self.head_deg[order])
        P_w = transform(T_w_c_ping, transform(T_c0_So, P_so))
        img_ok = self.traj.valid(self.t_img)
        T_w_c_img = np.zeros((len(self.t_img), 4, 4))
        T_w_c_img[img_ok] = self.traj.poses(self.t_img[img_ok])
        lo = np.searchsorted(tt, self.t_img - window_s)
        hi = np.searchsorted(tt, self.t_img + future_s, side="right")
        rec = {k: [] for k in ("p", "i", "zs", "zst", "r", "uv", "age")}
        for i in np.nonzero(img_ok & (hi > lo))[0]:
            sl = slice(lo[i], hi[i])
            Pc = transform(invert(T_w_c_img[i]), P_w[sl])
            z, uv = self.project_rect(Pc)
            ii = np.full(len(z), i)
            zst = self.lookup(ii, uv)
            m = np.isfinite(zst) & np.isfinite(z)
            rec["p"].append(order[sl][m]); rec["i"].append(ii[m]); rec["zs"].append(z[m]); rec["zst"].append(zst[m])
            rec["r"].append(self.range_m[order[sl][m]]); rec["uv"].append(uv[m])
            rec["age"].append(self.t_img[i] - tt[sl][m])
        cat = lambda k, shape=(0,): np.concatenate(rec[k]) if rec[k] else np.zeros(shape)
        return Pairs(cat("p").astype(int), cat("i").astype(int), cat("zs"), cat("zst"), cat("r"),
                     np.concatenate(rec["uv"]) if rec["uv"] else np.zeros((0, 2)), cat("age"))

    def same_time_points(self, dt_s: float = 0.0, max_gap_s: float = 0.04):
        """Pose-free pairing: each valid ping with the cached image nearest in (t_ping+dt); returns ping idx, img idx."""
        tp = self.t_ping + dt_s
        j = np.clip(np.searchsorted(self.t_img, tp), 1, len(self.t_img) - 1)
        j = np.where(np.abs(self.t_img[j - 1] - tp) < np.abs(self.t_img[j] - tp), j - 1, j)
        ok = np.isfinite(self.range_m) & (np.abs(self.t_img[j] - tp) <= max_gap_s)
        return np.nonzero(ok)[0], j[ok]


def robust_stats(res: np.ndarray) -> dict:
    if len(res) == 0:
        return dict(n=0, median=np.nan, mad=np.nan)
    med = float(np.median(res))
    return dict(n=int(len(res)), median=med, mad=float(1.4826 * np.median(np.abs(res - med))))
