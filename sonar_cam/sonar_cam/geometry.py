"""Sonar ping -> point in sonar frame -> world -> camera pixel.

Frames: ``world`` (trajectory frame), ``c0`` (left camera, optical: x right, y down, z forward), ``So`` (sonar).
Naming: T_w_c0 maps c0 points to world; T_c0_So maps sonar points to c0.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation

# SVIn Subscriber::sonarCallback (commented block in okvis_ros/src/Subscriber.cpp:359-379): peak = argmax of
# intensity over bins [0, n-100), accepted if peak intensity > 10 and range < 4.5 m,
# range = (idx + 1) * max_range / n.
SVIN_TAIL_BINS = 100
SVIN_MIN_INTENSITY = 10
SVIN_MAX_RANGE_M = 4.5


def ping_range(intensity: np.ndarray, max_range_m: float, min_intensity: int = SVIN_MIN_INTENSITY,
               tail_bins: int = SVIN_TAIL_BINS, max_accept_m: float = SVIN_MAX_RANGE_M,
               min_accept_m: float = 0.0) -> tuple[float, int]:
    """SVIn-style range from one ping: (range_m or NaN, peak intensity)."""
    n = len(intensity)
    idx = int(np.argmax(intensity[: n - tail_bins]))
    peak = int(intensity[idx])
    r = (idx + 1) * max_range_m / n
    if peak <= min_intensity or not (min_accept_m <= r < max_accept_m):
        return float("nan"), peak
    return r, peak


def ping_range_contrast(intensity: np.ndarray, max_range_m: float, min_prominence: float = 30.0,
                        baseline_bins: int = 31, tail_bins: int = SVIN_TAIL_BINS,
                        min_accept_m: float = 0.3, max_accept_m: float = SVIN_MAX_RANGE_M) -> tuple[float, float]:
    """Alternative picker (not in SVIn): argmax of (intensity - running median over baseline_bins) in
    [0, n-tail), accepted if the prominence >= min_prominence. The raw-argmax SVIn rule is biased towards the
    end of the search window because the mean intensity grows with range (time-varying gain): 20 % of the raw
    argmax picks land in the last 10 bins of the window.  Returns (range_m or NaN, prominence)."""
    n = len(intensity)
    x = intensity.astype(float)
    prom = x - median_filter(x, size=baseline_bins, mode="nearest")
    idx = int(np.argmax(prom[: n - tail_bins]))
    r = (idx + 1) * max_range_m / n
    if prom[idx] < min_prominence or not (min_accept_m <= r < max_accept_m):
        return float("nan"), float(prom[idx])
    return r, float(prom[idx])


def sonar_points(range_m: np.ndarray, head_deg: np.ndarray) -> np.ndarray:
    """(N,3) in the sonar frame: (r cos h, r sin h, 0), h = head_position in radians (SonarError.cpp:144-150)."""
    h = np.deg2rad(np.asarray(head_deg, float))
    r = np.asarray(range_m, float)
    return np.stack([r * np.cos(h), r * np.sin(h), np.zeros_like(r)], axis=1)


def transform(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Apply T (4x4, or (N,4,4) one per point) to points (N,3)."""
    if T.ndim == 2:
        return P @ T[:3, :3].T + T[:3, 3]
    return np.einsum("nij,nj->ni", T[:, :3, :3], P) + T[:, :3, 3]


def invert(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def perturb(T_c0_So: np.ndarray, rotvec_rad=(0, 0, 0), trans_m=(0, 0, 0)) -> np.ndarray:
    """Left-multiplied correction expressed in the camera frame (rotation about the camera origin, then shift)."""
    D = np.eye(4)
    D[:3, :3] = Rotation.from_rotvec(rotvec_rad).as_matrix()
    D[:3, 3] = trans_m
    return D @ T_c0_So


def sonar_world_points(T_w_c0_at_ping: np.ndarray, T_c0_So: np.ndarray, range_m, head_deg) -> np.ndarray:
    """World-frame sonar points using the camera pose at each ping time ((N,4,4))."""
    P_so = sonar_points(range_m, head_deg)
    P_c0 = transform(T_c0_So, P_so)
    return transform(T_w_c0_at_ping, P_c0)


def world_to_cam(T_w_c0_at_img: np.ndarray, P_w: np.ndarray) -> np.ndarray:
    return transform(invert(T_w_c0_at_img), P_w)


def project(P_c0: np.ndarray, K: np.ndarray, dist: np.ndarray, size: tuple[int, int],
            min_z_m: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame points with the radtan model -> (uv (N,2), in_image (N,) bool, z>min_z required)."""
    P = np.asarray(P_c0, float).reshape(-1, 3)
    front = P[:, 2] > min_z_m
    uv = np.full((len(P), 2), np.nan)
    if front.any():
        uv[front] = cv2.projectPoints(P[front].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)
    w, h = size
    ok = front & (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
    return uv, ok
