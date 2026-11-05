"""Stereo rectification + SGBM depth for the bus_outside rig (radtan pinhole, camchain extrinsic)."""
from __future__ import annotations

import cv2
import numpy as np

from .data import Bag, TOPIC_LEFT, TOPIC_RIGHT, decode_compressed_image, load_camchain


class StereoRig:
    """Rectifies a left/right pair and computes depth in the *rectified left* frame.

    ``R1`` maps a c0 (left camera) point into the rectified left frame; ``f``/``cx``/``cy`` are the rectified
    intrinsics at working resolution (``scale`` x the native 1600x1200), ``baseline_m`` the stereo baseline.
    """

    def __init__(self, scale: float = 0.5, num_disp: int = 208, block: int = 7, clahe: bool = False):
        cc = load_camchain()
        self.cc = cc
        c0, c1 = cc["cam0"], cc["cam1"]
        self.size = c0["size"]
        R, t = cc["T_c1_c0"][:3, :3], cc["T_c1_c0"][:3, 3]
        R1, R2, P1, P2, *_ = cv2.stereoRectify(c0["K"], c0["dist"], c1["K"], c1["dist"], self.size, R, t,
                                               flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        self.R1 = R1
        self.scale = scale
        self.work_size = (int(round(self.size[0] * scale)), int(round(self.size[1] * scale)))
        self.f = P1[0, 0] * scale
        self.cx = P1[0, 2] * scale
        self.cy = P1[1, 2] * scale
        self.baseline_m = float(-P2[0, 3] / P2[0, 0])
        self.maps = [cv2.initUndistortRectifyMap(c["K"], c["dist"], Ri, Pi, self.size, cv2.CV_16SC2)
                     for c, Ri, Pi in ((c0, R1, P1), (c1, R2, P2))]
        self.num_disp = num_disp
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=num_disp, blockSize=block, P1=8 * block * block,
            P2=32 * block * block, disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100,
            speckleRange=2, preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) if clahe else None

    def _prep(self, jpg: bytes, k: int) -> np.ndarray:
        g = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_GRAYSCALE)
        g = cv2.remap(g, *self.maps[k], cv2.INTER_LINEAR)
        g = cv2.resize(g, self.work_size, interpolation=cv2.INTER_AREA)
        return self.clahe.apply(g) if self.clahe is not None else g

    def depth(self, jpg_left: bytes, jpg_right: bytes) -> tuple[np.ndarray, np.ndarray]:
        """(depth_z_m float32 HxW with NaN where invalid, rectified left gray uint8) at working resolution."""
        gl, gr = self._prep(jpg_left, 0), self._prep(jpg_right, 1)
        d = self.sgbm.compute(gl, gr).astype(np.float32) / 16.0
        z = np.full(d.shape, np.nan, np.float32)
        ok = d >= 1.0
        z[ok] = self.f * self.baseline_m / d[ok]
        return z, gl

    def to_rect_px(self, P_c0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """c0-frame points (N,3) -> rectified-frame points and (u,v) pixels at working resolution."""
        Pr = P_c0 @ self.R1.T
        z = np.where(Pr[:, 2] > 1e-6, Pr[:, 2], np.nan)
        uv = np.stack([self.f * Pr[:, 0] / z + self.cx, self.f * Pr[:, 1] / z + self.cy], axis=1)
        return Pr, uv


def stereo_pairs(bag: Bag, max_dt_s: float = 0.005) -> list[tuple[int, int, float]]:
    """(left_rowid, right_rowid, left_stamp) matched by nearest header stamp."""
    from .data import decode_image_stamp
    lr, rr = bag.rowids(TOPIC_LEFT), bag.rowids(TOPIC_RIGHT)
    tl = np.array([decode_image_stamp(bag.blob(r)) for r in lr])
    tr = np.array([decode_image_stamp(bag.blob(r)) for r in rr])
    out = []
    for i, t in enumerate(tl):
        j = int(np.argmin(np.abs(tr - t)))
        if abs(tr[j] - t) <= max_dt_s:
            out.append((int(lr[i]), int(rr[j]), float(t)))
    return out
