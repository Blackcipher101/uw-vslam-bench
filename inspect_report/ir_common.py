"""Shared IO, calibration, pose and timing helpers for the inspection-report baseline.

Clock domain: image header stamps (seconds, float) are compared directly with TUM
timestamps; both must come from the same clock. Poses are the LEFT (cam0) camera in
the VIO world frame (TUM: t x y z qx qy qz qw).
"""
from __future__ import annotations

import json
import sqlite3
import struct
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import yaml

POSE_MAX_GAP_S = 0.5  # nearest TUM sample further than this from an image stamp => no pose
PAIR_TOL_S = 0.005  # stereo pair stamp tolerance


# ----------------------------------------------------------------------------- timing
class StageTimer:
    """Appends one JSON line per timed event to <out>/timings.jsonl, flushed immediately."""

    def __init__(self, out_dir: Path | str, cuda: bool = False):
        self.path = Path(out_dir) / "timings.jsonl"
        self.cuda = cuda

    def _sync(self) -> None:
        if self.cuda:
            import torch

            torch.cuda.synchronize()

    @contextmanager
    def time(self, stage: str, **extra):
        self._sync()
        t0 = time.perf_counter()
        yield
        self._sync()
        dt = time.perf_counter() - t0
        rec = {"stage": stage, "seconds": dt, **extra}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()


def load_timings(out_dir: Path | str) -> dict[str, dict]:
    """Aggregate timings.jsonl into {stage: {n, total_s, mean_s, median_s, p95_s}}."""
    p = Path(out_dir) / "timings.jsonl"
    by: dict[str, list[float]] = {}
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            by.setdefault(r["stage"], []).append(r["seconds"])
    return {
        k: {
            "n": len(v),
            "total_s": float(np.sum(v)),
            "mean_s": float(np.mean(v)),
            "median_s": float(np.median(v)),
            "p95_s": float(np.percentile(v, 95)),
        }
        for k, v in by.items()
    }


# ------------------------------------------------------------------------ calibration
def load_camchain(path: Path | str) -> dict:
    """Kalibr camchain -> dict with K, dist, model, size, T_c1_c0 (maps cam0 points into cam1), topics."""
    cc = yaml.safe_load(open(path))
    out = {}
    for name in ("cam0", "cam1"):
        c = cc[name]
        fx, fy, cx, cy = c["intrinsics"]
        out[name] = {
            "K": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]),
            "dist": np.array(c["distortion_coeffs"], dtype=np.float64),
            "model": c["distortion_model"],
            "size": tuple(c["resolution"]),  # (w, h)
            "topic": c.get("rostopic"),
        }
    out["T_c1_c0"] = np.array(cc["cam1"]["T_cn_cnm1"], dtype=np.float64)
    return out


# ------------------------------------------------------------------------------ poses
def load_tum(path: Path | str) -> np.ndarray:
    """Return Nx8 array [t, x, y, z, qx, qy, qz, qw], sorted by time. Rejects non-finite rows."""
    a = np.loadtxt(path, ndmin=2)
    if a.shape[1] != 8:
        raise ValueError(f"{path}: expected 8 columns (TUM), got {a.shape[1]}")
    a = a[np.all(np.isfinite(a), axis=1)]
    return a[np.argsort(a[:, 0], kind="stable")]


def quat_to_R(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def interp_pose(tum: np.ndarray, t: float, max_gap_s: float = POSE_MAX_GAP_S) -> np.ndarray | None:
    """4x4 T_world_cam at time t (linear position, normalised-lerp rotation), or None if no
    TUM sample within max_gap_s on the relevant side."""
    ts = tum[:, 0]
    i = int(np.searchsorted(ts, t))
    if i == 0 or i == len(ts):
        j = 0 if i == 0 else len(ts) - 1
        if abs(ts[j] - t) > max_gap_s:
            return None
        lo = hi = j
        a = 0.0
    else:
        lo, hi = i - 1, i
        if min(t - ts[lo], ts[hi] - t) > max_gap_s:
            return None
        a = (t - ts[lo]) / (ts[hi] - ts[lo]) if ts[hi] > ts[lo] else 0.0
    p = (1 - a) * tum[lo, 1:4] + a * tum[hi, 1:4]
    q0, q1 = tum[lo, 4:8], tum[hi, 4:8]
    if np.dot(q0, q1) < 0:
        q1 = -q1
    q = (1 - a) * q0 + a * q1
    if not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-9:
        return None
    T = np.eye(4)
    T[:3, :3] = quat_to_R(q)
    T[:3, 3] = p
    return T


def path_length_m(tum: np.ndarray, t0: float, t1: float) -> float:
    m = (tum[:, 0] >= t0) & (tum[:, 0] <= t1)
    xyz = tum[m, 1:4]
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()) if len(xyz) > 1 else 0.0


# ---------------------------------------------------------------------------- bag IO
def _stamp_from_cdr(data: bytes) -> float:
    """Header stamp of any ROS 2 message whose first field is std_msgs/Header (CDR little endian)."""
    sec, nsec = struct.unpack_from("<iI", data, 4)
    return sec + nsec * 1e-9


def raw_messages(bag: Path | str, topics: list[str]) -> Iterator[tuple[str, float, bytes, str]]:
    """Yield (topic, header_stamp_s, cdr_bytes, msgtype) in bag order.

    A *.db3 file is read directly with sqlite (the AFRL bag has no metadata.yaml); a
    directory is opened with rosbags (sqlite3 or mcap rosbag2).
    """
    bag = Path(bag)
    if bag.suffix == ".db3":
        con = sqlite3.connect(f"file:{bag}?mode=ro", uri=True)
        tid = {r[1]: (r[0], r[2]) for r in con.execute("select id,name,type from topics")}
        missing = [t for t in topics if t not in tid]
        if missing:
            raise KeyError(f"topics {missing} not in {bag}; have {sorted(tid)}")
        ids = {tid[t][0]: (t, tid[t][1]) for t in topics}
        q = f"select topic_id,data from messages where topic_id in ({','.join(map(str, ids))}) order by timestamp"
        for topic_id, data in con.execute(q):
            data = bytes(data)
            t, typ = ids[topic_id]
            yield t, _stamp_from_cdr(data), data, typ
        con.close()
    else:
        from rosbags.highlevel import AnyReader

        with AnyReader([bag]) as r:
            conns = [c for c in r.connections if c.topic in topics]
            missing = set(topics) - {c.topic for c in conns}
            if missing:
                raise KeyError(f"topics {sorted(missing)} not in {bag}")
            for c, _, data in r.messages(connections=conns):
                yield c.topic, _stamp_from_cdr(bytes(data)), bytes(data), c.msgtype


_TYPESTORE = None


def decode_image(data: bytes, msgtype: str) -> np.ndarray:
    """Decode a CompressedImage or raw Image CDR blob to BGR uint8 (mono is replicated)."""
    import cv2

    global _TYPESTORE
    if _TYPESTORE is None:
        from rosbags.typesys import Stores, get_typestore

        _TYPESTORE = get_typestore(Stores.ROS2_JAZZY)
    msg = _TYPESTORE.deserialize_cdr(data, msgtype)
    if msgtype.endswith("CompressedImage"):
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cannot decode compressed image ({msg.format})")
        return img
    enc = msg.encoding
    buf = np.asarray(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    if enc in ("mono8", "8UC1"):
        g = buf[:, : msg.width]
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    if enc in ("bgr8", "rgb8"):
        im = buf[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
        return im if enc == "bgr8" else cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    raise ValueError(f"unsupported image encoding {enc}")


def pair_stamps(left: list[float], right: list[float], tol_s: float = PAIR_TOL_S) -> list[tuple[int, int]]:
    """Index pairs (i_left, j_right) with |dt| <= tol_s, nearest match, each right used once."""
    r = np.asarray(right)
    used: set[int] = set()
    pairs = []
    if len(r) == 0:
        return pairs
    order = np.argsort(r)
    rs = r[order]
    for i, t in enumerate(left):
        k = int(np.searchsorted(rs, t))
        cands = [c for c in (k - 1, k) if 0 <= c < len(rs)]
        c = min(cands, key=lambda c: abs(rs[c] - t))
        j = int(order[c])
        if abs(rs[c] - t) <= tol_s and j not in used:
            used.add(j)
            pairs.append((i, j))
    return pairs


def select_keyframes(
    stamps: list[float],
    positions: list[np.ndarray | None],
    min_dt_s: float = 2.0,
    max_dt_s: float = 4.0,
    min_dist_m: float = 0.3,
) -> list[int]:
    """Indices of keyframes: first frame; then the first frame with dt>=min_dt_s since the last
    keyframe AND (travelled >= min_dist_m, or dt>=max_dt_s, or pose unknown)."""
    sel: list[int] = []
    for i, t in enumerate(stamps):
        if not sel:
            sel.append(i)
            continue
        j = sel[-1]
        dt = t - stamps[j]
        if dt < min_dt_s:
            continue
        pi, pj = positions[i], positions[j]
        moved = pi is None or pj is None or float(np.linalg.norm(pi - pj)) >= min_dist_m
        if moved or dt >= max_dt_s:
            sel.append(i)
    return sel
