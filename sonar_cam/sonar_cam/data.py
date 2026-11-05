"""Data access: bag (sqlite3 + hand-written CDR decode), camchain, SVIn sonar extrinsic, TUM trajectories.

All transforms are 4x4 float64 ``T_a_b`` = pose of frame b expressed in frame a (maps b-points to a-points).
All times are float seconds in the bag (ROS1-origin epoch) clock; header stamps are used, not bag receive time.
"""
from __future__ import annotations
import os

import re
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

_DATA = os.path.expanduser(os.environ.get("UWB_DATA", "~/svin_data"))  # dataset root (env UWB_DATA)
BAG = f"{_DATA}/bus_outside/bus_outside.db3"
CAMCHAIN = f"{_DATA}/bus_outside/camchain_bus_outside.yaml"
SVIN_RIG_V1 = os.path.expanduser(os.environ.get("UWB_SVIN_CONFIG", "~/svin_ws/src/SVIn/config")) + "/config_stereorig_v1.yaml"
GT_COLMAP = f"{_DATA}/bus_outside/bus_outside.txt"
VIO_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results/bus_outside/uwct/r2_off.tum")

TOPIC_LEFT = "/slave1/image_raw/compressed"   # cam0 in camchain
TOPIC_RIGHT = "/slave2/image_raw/compressed"  # cam1
TOPIC_SONAR = "/imagenex831l/range"           # imagenex831l/msg/ProcessedRange


@dataclass
class Ping:
    t: float              # header stamp [s]
    head_deg: float       # head_position [deg], full 360 rotation in 0.9 deg steps
    max_range_m: float    # max_range [m] (6.0 in this bag)
    intensity: np.ndarray  # uint8[250] echo intensity, bin i covers range ((i+1) * max_range / n)


def _align(off: int, a: int) -> int:
    return (off + a - 1) // a * a


def _read_header(d: bytes):
    """CDR (little endian) std_msgs/Header at offset 4 -> (stamp_s, frame_id, end_offset)."""
    sec, nsec, slen = struct.unpack_from("<IIi", d, 4)
    frame = d[16:16 + slen - 1].decode()
    return sec + nsec * 1e-9, frame, 16 + slen


def decode_ping(d: bytes) -> Ping:
    """Decode an imagenex831l/msg/ProcessedRange CDR blob.

    Layout (verified on all 55923 messages, see README): Header; uint8[] intensity (len 250);
    4 bytes unidentified; float32 head_position [deg]; uint32 (=1, unidentified);
    float32 max_range [m]; remaining 5 words unidentified and unused.
    The msg definition is not available offline; field names are the ones SVIn's subscriber uses
    (head_position, max_range, intensity).
    """
    t, _, off = _read_header(d)
    off = _align(off, 4)
    n = struct.unpack_from("<I", d, off)[0]
    off += 4
    inten = np.frombuffer(d, np.uint8, n, off).copy()
    off = _align(off + n, 4) + 4          # skip unidentified 4 bytes
    head, _, max_range = struct.unpack_from("<fIf", d, off)
    return Ping(t, float(head), float(max_range), inten)


def decode_image_stamp(d: bytes) -> float:
    return _read_header(d)[0]


def decode_compressed_image(d: bytes) -> bytes:
    """sensor_msgs/CompressedImage: Header, string format, uint8[] data -> JPEG bytes."""
    _, _, off = _read_header(d)
    off = _align(off, 4)
    flen = struct.unpack_from("<I", d, off)[0]
    off = _align(off + 4 + flen, 4)
    n = struct.unpack_from("<I", d, off)[0]
    return d[off + 4: off + 4 + n]


class Bag:
    """Read-only access to the sqlite3 bag; rowids per topic are indexed once."""

    def __init__(self, path: str = BAG):
        self.con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.topics = {n: i for i, n in self.con.execute("select id,name from topics")}
        self._rows: dict[str, np.ndarray] = {}

    def rowids(self, topic: str) -> np.ndarray:
        if topic not in self._rows:
            q = "select id from messages where topic_id=? order by timestamp"
            self._rows[topic] = np.array([r[0] for r in self.con.execute(q, (self.topics[topic],))])
        return self._rows[topic]

    def blob(self, rowid: int) -> bytes:
        return self.con.execute("select data from messages where id=?", (int(rowid),)).fetchone()[0]

    def iter_blobs(self, topic: str):
        q = "select data from messages where topic_id=? order by timestamp"
        for (d,) in self.con.execute(q, (self.topics[topic],)):
            yield d

    def pings(self) -> list[Ping]:
        return [decode_ping(d) for d in self.iter_blobs(TOPIC_SONAR)]

    def image_stamps(self, topic: str) -> np.ndarray:
        return np.array([decode_image_stamp(self.blob(r)) for r in self.rowids(topic)])


def load_camchain(path: str = CAMCHAIN) -> dict:
    """cam0/cam1 intrinsics K, distortion (radtan k1 k2 p1 p2), size and T_cam1_cam0 (stereo extrinsic)."""
    cc = yaml.safe_load(open(path))
    out = {}
    for k in ("cam0", "cam1"):
        fx, fy, cx, cy = cc[k]["intrinsics"]
        out[k] = dict(K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]),
                      dist=np.array(cc[k]["distortion_coeffs"], float),
                      size=tuple(cc[k]["resolution"]),
                      T_cam_imu=np.array(cc[k]["T_cam_imu"], float),
                      topic=cc[k]["rostopic"])
    out["T_c1_c0"] = np.array(cc["cam1"]["T_cn_cnm1"], float)
    return out


def load_svin_rig(path: str = SVIN_RIG_V1) -> dict:
    """Parse T_SC (cam0, cam1) and sonar T_SSo from an SVIn config (OpenCV-yaml flavour, not plain YAML)."""
    txt = open(path).read()
    mats = [np.array([float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", m)]).reshape(4, 4)
            for m in re.findall(r"T_SC:\s*\[([^\]]*)\]", txt)]
    so = re.search(r"T_SSo:\s*\[([^\]]*)\]", txt).group(1)
    T_SSo = np.array([float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", so)]).reshape(4, 4)
    return dict(T_SC0=mats[0], T_SC1=mats[1], T_SSo=T_SSo)


def sonar_in_cam0(rig: dict) -> np.ndarray:
    """T_c0_So = inv(T_SC0) @ T_SSo (IMU S frame is the SVIn body frame)."""
    return np.linalg.inv(rig["T_SC0"]) @ rig["T_SSo"]


class Trajectory:
    """TUM trajectory (t x y z qx qy qz qw) = T_world_cam0 (left camera). Linear/slerp interpolation."""

    def __init__(self, path: str):
        a = np.loadtxt(path, comments="#")
        a = a[np.argsort(a[:, 0])]
        self.t = a[:, 0]
        self.p = a[:, 1:4]
        self.rot = Rotation.from_quat(a[:, 4:8])
        self._slerp = Slerp(self.t, self.rot)

    def valid(self, t: np.ndarray, max_gap_s: float = 0.5) -> np.ndarray:
        """True where t lies in the trajectory span and the neighbouring samples are <= max_gap_s apart."""
        t = np.asarray(t, float)
        inside = (t >= self.t[0]) & (t <= self.t[-1])
        i = np.clip(np.searchsorted(self.t, t), 1, len(self.t) - 1)
        return inside & ((self.t[i] - self.t[i - 1]) <= max_gap_s)

    def poses(self, t: np.ndarray) -> np.ndarray:
        """(N,4,4) T_world_cam0 at times t (must be inside the span; callers filter with valid())."""
        t = np.clip(np.asarray(t, float), self.t[0], self.t[-1])
        i = np.clip(np.searchsorted(self.t, t), 1, len(self.t) - 1)
        w = ((t - self.t[i - 1]) / (self.t[i] - self.t[i - 1]))[:, None]
        T = np.tile(np.eye(4), (len(t), 1, 1))
        T[:, :3, :3] = self._slerp(t).as_matrix()
        T[:, :3, 3] = self.p[i - 1] * (1 - w) + self.p[i] * w
        return T
