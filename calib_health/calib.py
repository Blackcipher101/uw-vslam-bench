"""Stereo rig model, calibration loaders, error injection and virtual rectification.

Frames: OpenCV optical frames (x right, y down, z forward). ``x_l = R_lr x_r + t_lr``:
``R_lr, t_lr`` map right-camera points into the left-camera frame; the right camera centre sits
at ``t_lr`` (about +x) in the left frame. Units: metres, radians internally, degrees at the API
edge. A "rotation error" ``w`` (rad, left-frame axes) means ``R_cand = Exp(w) @ R_true``.
"""
from __future__ import annotations
import os

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

try:
    import yaml

    def _load_yaml(p: Path) -> dict:
        return yaml.safe_load(p.read_text())
except ImportError:  # the rosbags venv ships ruamel instead of PyYAML
    from ruamel.yaml import YAML

    def _load_yaml(p: Path) -> dict:
        return YAML(typ="safe").load(p.read_text())

_DATA = Path(os.path.expanduser(os.environ.get("UWB_DATA", "~/svin_data")))  # dataset root (env UWB_DATA)
NTNU_CAL = _DATA / "ntnu/calibration"
WATER_CHAIN = NTNU_CAL / "stereo_intrinsics_under_water/camchain-ariel_stereo_underwater.yaml"
AIR_IMUCAM = NTNU_CAL / "stereo_extrinsics_above_water/camchain-imucam-ariel_stereo_extrinsics_above_water.yaml"
BUS_CHAIN = _DATA / "bus_outside/camchain_bus_outside.yaml"


@dataclass(frozen=True)
class Cam:
    K: np.ndarray
    D: np.ndarray
    fisheye: bool
    size: tuple[int, int]  # (width, height)

    @property
    def f(self) -> float:
        return float(0.5 * (self.K[0, 0] + self.K[1, 1]))

    def normalize(self, pts: np.ndarray) -> np.ndarray:
        """Pixels (N,2) -> normalized image coordinates (x/z, y/z), (N,2)."""
        p = np.ascontiguousarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        if self.fisheye:
            out = cv2.fisheye.undistortPoints(p, self.K, self.D.reshape(-1, 1))
        else:
            out = cv2.undistortPoints(p, self.K, self.D)
        return out.reshape(-1, 2)

    def project(self, xn: np.ndarray) -> np.ndarray:
        """Normalized coordinates (N,2) -> pixels (N,2) (inverse of normalize)."""
        xyz = np.hstack([xn, np.ones((len(xn), 1))]).reshape(-1, 1, 3)
        if self.fisheye:
            out = cv2.fisheye.projectPoints(xyz, np.zeros(3), np.zeros(3), self.K, self.D.reshape(-1, 1))[0]
        else:
            out = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), self.K, self.D)[0]
        return out.reshape(-1, 2)


@dataclass(frozen=True)
class Rig:
    name: str
    left: Cam
    right: Cam
    R_lr: np.ndarray
    t_lr: np.ndarray

    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(self.t_lr))

    @property
    def f_px(self) -> float:
        return self.left.f


def exp_so3(w: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(w, dtype=np.float64).reshape(3, 1))[0]


def log_so3(R: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(R, dtype=np.float64))[0].ravel()


def inject(rig: Rig, w_deg: np.ndarray) -> Rig:
    """Candidate rig with ``R_cand = Exp(w) R_true`` (w in degrees, left-frame x, y, z)."""
    R = exp_so3(np.deg2rad(np.asarray(w_deg, dtype=float))) @ rig.R_lr
    return replace(rig, R_lr=R)


def rotation_error_deg(R_cand: np.ndarray, R_true: np.ndarray) -> np.ndarray:
    """w such that R_cand = Exp(w) R_true, in degrees."""
    return np.rad2deg(log_so3(R_cand @ R_true.T))


def _cam_from_dict(d: dict, fisheye: bool) -> Cam:
    fx, fy, cx, cy = d["intrinsics"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    return Cam(K, np.asarray(d["distortion_coeffs"], dtype=float), fisheye, tuple(int(v) for v in d["resolution"]))


def load_ntnu_water() -> Rig:
    """In-water intrinsics + in-water extrinsic (the GOOD one). Left = cam1, right = cam0."""
    cc = _load_yaml(WATER_CHAIN)
    T = np.asarray(cc["cam1"]["T_cn_cnm1"], dtype=float)  # cam0 -> cam1 = right -> left
    return Rig("ntnu_water", _cam_from_dict(cc["cam1"], True), _cam_from_dict(cc["cam0"], True), T[:3, :3], T[:3, 3])


def load_ntnu_air_extrinsic(good: Rig) -> Rig:
    """GOOD rig with the stereo extrinsic taken from the in-air imu-cam chain (the BAD one)."""
    cc = _load_yaml(AIR_IMUCAM)
    T = np.asarray(cc["cam1"]["T_cn_cnm1"], dtype=float)
    return replace(good, name="ntnu_air_extrinsic", R_lr=T[:3, :3], t_lr=T[:3, 3])


def air_vs_water_error_deg(good: Rig) -> np.ndarray:
    return rotation_error_deg(load_ntnu_air_extrinsic(good).R_lr, good.R_lr)


def load_bus() -> Rig:
    """bus_outside: radtan; cam0 is the left camera, T_cn_cnm1 maps cam0 -> cam1 (left -> right)."""
    cc = _load_yaml(BUS_CHAIN)
    T_rl = np.asarray(cc["cam1"]["T_cn_cnm1"], dtype=float)
    T_lr = np.linalg.inv(T_rl)
    return Rig("bus_outside", _cam_from_dict(cc["cam0"], False), _cam_from_dict(cc["cam1"], False), T_lr[:3, :3], T_lr[:3, 3])


def rect_rotation(t_lr: np.ndarray) -> np.ndarray:
    """Rotation taking left-frame vectors into a frame whose x axis is the baseline."""
    e1 = t_lr / np.linalg.norm(t_lr)
    e2 = np.cross([0.0, 0.0, 1.0], e1)
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    return np.vstack([e1, e2, e3])


def rectified_coords(rig: Rig, pl: np.ndarray, pr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Matched raw pixels -> virtual-rectified normalized coordinates (xl', yl'), (xr', yr').

    Both rays are rotated into the baseline-aligned frame defined by ``rig.t_lr``. For a
    perfect calibration yl' == yr' and ``f (xl' - xr') = f B / Z``.
    """
    Rr = rect_rotation(rig.t_lr)
    nl = rig.left.normalize(pl)
    nr = rig.right.normalize(pr)
    vl = (Rr @ np.hstack([nl, np.ones((len(nl), 1))]).T).T
    vr = (Rr @ rig.R_lr @ np.hstack([nr, np.ones((len(nr), 1))]).T).T
    return vl[:, :2] / vl[:, 2:3], vr[:, :2] / vr[:, 2:3]
