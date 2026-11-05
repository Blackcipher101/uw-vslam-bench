"""Access to the AFRL stereo-VI underwater dataset (ROS 2 .db3 bags + Kalibr calibration).

Frames: Kalibr convention, cameras are OpenCV optical frames (x right, y down,
z forward). ``T_a_b`` maps points expressed in frame b into frame a.
Timestamps: bag header stamps in ns; the ground truth uses the left-image stamps.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import rosbag2_py
import yaml
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage, Imu

NS = 1_000_000_000
LEFT_TOPIC = "/slave1/image_raw/compressed"
RIGHT_TOPIC = "/slave2/image_raw/compressed"
IMU_TOPIC = "/imu/imu"
# The two IDS cameras are slaved; on bus_outside the right stamp trails the left
# by a constant ~1.07 ms, so pair by nearest stamp within this window.
STEREO_PAIR_TOL_NS = 5_000_000


@dataclass(frozen=True)
class CameraCalib:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    radtan: tuple[float, float, float, float]  # Kalibr order: k1, k2, p1, p2
    T_cam_imu: np.ndarray

    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def D(self) -> np.ndarray:
        """OpenCV distortion vector (k1, k2, p1, p2)."""
        return np.asarray(self.radtan, dtype=float)


@dataclass(frozen=True)
class ImuNoise:
    gyroscope_noise_density: float
    gyroscope_random_walk: float
    accelerometer_noise_density: float
    accelerometer_random_walk: float
    rate_hz: float


@dataclass(frozen=True)
class Calibration:
    cam0: CameraCalib
    cam1: CameraCalib
    T_cam1_cam0: np.ndarray
    timeshift_cam_imu_s: float  # Kalibr: t_imu = t_cam + timeshift
    imu: ImuNoise


def _camera(d: dict) -> CameraCalib:
    if d["camera_model"] != "pinhole" or d["distortion_model"] != "radtan":
        raise ValueError(f"unsupported camera {d['camera_model']}/{d['distortion_model']}")
    fx, fy, cx, cy = d["intrinsics"]
    w, h = d["resolution"]
    return CameraCalib(int(w), int(h), fx, fy, cx, cy, tuple(d["distortion_coeffs"]),
                       np.asarray(d["T_cam_imu"], dtype=float))


def load_calibration(camchain: Path, imu_yaml: Path) -> Calibration:
    cc = yaml.safe_load(camchain.read_text())
    iy = yaml.safe_load(imu_yaml.read_text())
    cam0, cam1 = _camera(cc["cam0"]), _camera(cc["cam1"])
    T_cam1_cam0 = np.asarray(cc["cam1"]["T_cn_cnm1"], dtype=float)
    # The chain and the per-camera IMU extrinsics must agree; a large mismatch means
    # the files are from different calibrations.
    T_chain = cam1.T_cam_imu @ np.linalg.inv(cam0.T_cam_imu)
    dt = np.linalg.norm(T_chain[:3, 3] - T_cam1_cam0[:3, 3])
    if dt > 0.01:
        raise ValueError(f"camchain inconsistent: cam1<-cam0 differs by {dt:.3f} m")
    shift = float(cc["cam0"].get("timeshift_cam_imu", 0.0))
    imu = ImuNoise(iy["gyroscope_noise_density"], iy["gyroscope_random_walk"],
                   iy["accelerometer_noise_density"], iy["accelerometer_random_walk"],
                   float(iy["update_rate"]))
    return Calibration(cam0, cam1, T_cam1_cam0, shift, imu)


def _reader(bag: Path, topics: list[str]) -> rosbag2_py.SequentialReader:
    r = rosbag2_py.SequentialReader()
    storage = "sqlite3" if bag.suffix == ".db3" else "mcap"
    r.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=storage),
           rosbag2_py.ConverterOptions("", ""))
    r.set_filter(rosbag2_py.StorageFilter(topics=topics))
    return r


def _stamp(msg) -> int:
    return msg.header.stamp.sec * NS + msg.header.stamp.nanosec


def read_imu(bag: Path, timeshift_cam_imu_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t_ns in camera time, acc m/s^2, gyro rad/s), sorted, finite, strictly increasing."""
    r = _reader(bag, [IMU_TOPIC])
    ts, acc, gyro = [], [], []
    while r.has_next():
        _, data, _ = r.read_next()
        m = deserialize_message(data, Imu)
        a, g = m.linear_acceleration, m.angular_velocity
        v = (a.x, a.y, a.z, g.x, g.y, g.z)
        if not all(np.isfinite(v)):
            continue
        ts.append(_stamp(m) - int(round(timeshift_cam_imu_s * NS)))
        acc.append(v[:3])
        gyro.append(v[3:])
    t = np.asarray(ts, dtype=np.int64)
    order = np.argsort(t, kind="stable")
    t, acc_a, gyro_a = t[order], np.asarray(acc)[order], np.asarray(gyro)[order]
    keep = np.concatenate([[True], np.diff(t) > 0]) if len(t) else np.zeros(0, bool)
    return t[keep], acc_a[keep], gyro_a[keep]


def _decode_gray(msg: CompressedImage) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"failed to decode image at t={_stamp(msg)} ({msg.format})")
    return img


def iter_stereo(bag: Path, decode_right: bool = True) -> Iterator[tuple[int, np.ndarray, np.ndarray | None]]:
    """Yield (t_left_ns, left_gray, right_gray) for pairs within STEREO_PAIR_TOL_NS.

    Unpaired frames are dropped; the caller sees the gap in the timestamps.
    """
    r = _reader(bag, [LEFT_TOPIC, RIGHT_TOPIC])
    pend: dict[str, dict[int, CompressedImage]] = {"l": {}, "r": {}}
    while r.has_next():
        topic, data, _ = r.read_next()
        m = deserialize_message(data, CompressedImage)
        t = _stamp(m)
        side, other = ("l", "r") if topic == LEFT_TOPIC else ("r", "l")
        match = min(pend[other], key=lambda k: abs(k - t), default=None)
        if match is None or abs(match - t) > STEREO_PAIR_TOL_NS:
            pend[side][t] = m
            for k in [k for k in pend[side] if t - k > NS // 2]:
                del pend[side][k]
            continue
        o = pend[other].pop(match)
        lm, rm = (m, o) if side == "l" else (o, m)
        yield _stamp(lm), _decode_gray(lm), (_decode_gray(rm) if decode_right else None)
