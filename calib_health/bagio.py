"""Stereo frame iterators for the two bags used here (rosbags library; no ROS install needed).

Yields (t_s, left_gray, right_gray); ``t_s`` is the left header stamp in seconds since the first
stamp read from the bag. Unpaired frames (|dt| > tol) are dropped.
"""
from __future__ import annotations
import os

from pathlib import Path
from typing import Iterator

import sqlite3

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

_DATA = Path(os.path.expanduser(os.environ.get("UWB_DATA", "~/svin_data")))  # dataset root (env UWB_DATA)
NTNU_BAG = _DATA / "ntnu/traj_06/traj_06_ros2"
NTNU_LEFT, NTNU_RIGHT = "/alphasense_driver_ros/cam1", "/alphasense_driver_ros/cam0"
NTNU_PRESSURE = "/mavros/imu/static_pressure"
BUS_BAG = _DATA / "bus_outside/bus_outside.db3"
BUS_LEFT, BUS_RIGHT = "/slave1/image_raw/compressed", "/slave2/image_raw/compressed"
PAIR_TOL_S = 0.003


def _stamp(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def _raw_gray(msg) -> np.ndarray:
    if msg.encoding != "mono8":
        raise ValueError(f"expected mono8, got {msg.encoding}")
    return np.asarray(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, : msg.width].copy()


def _jpeg_gray(msg) -> np.ndarray:
    img = cv2.imdecode(np.asarray(msg.data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("failed to decode compressed image")
    return img


def _iter_db3(path: Path, topics: tuple[str, str]) -> Iterator[tuple[str, float, object]]:
    """Messages (topic, stamp_s, msg) of a bare .db3 (no metadata.yaml), in bag order."""
    ts = get_typestore(Stores.ROS2_HUMBLE)
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    ids = {n: (i, t) for i, n, t in con.execute("select id, name, type from topics") if n in topics}
    q = "select topic_id, data from messages where topic_id in (%s) order by timestamp" % ",".join(str(i) for i, _ in ids.values())
    by_id = {i: (n, t) for n, (i, t) in ids.items()}
    for tid, data in con.execute(q):
        n, mt = by_id[tid]
        m = ts.deserialize_cdr(bytes(data), mt)
        yield n, _stamp(m), m


def iter_pairs(bag: Path, left: str, right: str, t0_s: float, t1_s: float, compressed: bool,
               bag_t0_s: float | None = None) -> Iterator[tuple[float, np.ndarray, np.ndarray]]:
    dec = _jpeg_gray if compressed else _raw_gray
    pend: dict[str, dict[float, np.ndarray]] = {left: {}, right: {}}

    def source():
        if bag.suffix == ".db3":
            yield from _iter_db3(bag, (left, right))
            return
        with AnyReader([bag]) as rd:
            base = rd.start_time * 1e-9 if bag_t0_s is None else bag_t0_s
            conns = [c for c in rd.connections if c.topic in (left, right)]
            for conn, _, raw in rd.messages(connections=conns, start=int((base + t0_s - 1.0) * 1e9),
                                            stop=int((base + t1_s + 1.0) * 1e9)):
                m = rd.deserialize(raw, conn.msgtype)
                yield conn.topic, _stamp(m), m

    base = None
    for topic, t, msg in source():
        if base is None:
            base = (bag_t0_s if bag_t0_s is not None else (bag_start_s(bag) if bag.suffix != ".db3" else t))
        if t - base > t1_s + 1.0:
            break
        if t - base < t0_s - 1.0:
            continue
        me, other = topic, (right if topic == left else left)
        m = min(pend[other], key=lambda k: abs(k - t), default=None)
        if m is None or abs(m - t) > PAIR_TOL_S:
            pend[me][t] = dec(msg)
            for k in [k for k in pend[me] if t - k > 0.5]:
                del pend[me][k]
            continue
        o = pend[other].pop(m)
        img = dec(msg)
        l_img, r_img, tl = (img, o, t) if me == left else (o, img, m)
        if t0_s <= tl - base <= t1_s:
            yield tl - base, l_img, r_img


def bag_start_s(bag: Path) -> float:
    with AnyReader([bag]) as rd:
        return rd.start_time * 1e-9


def read_pressure(bag: Path, t0_s: float, t1_s: float) -> tuple[np.ndarray, np.ndarray]:
    """(t_s since bag start, pressure Pa) within [t0_s, t1_s]."""
    ts, pa = [], []
    with AnyReader([bag]) as rd:
        base = rd.start_time * 1e-9
        conns = [c for c in rd.connections if c.topic == NTNU_PRESSURE]
        for conn, _, raw in rd.messages(connections=conns):
            m = rd.deserialize(raw, conn.msgtype)
            t = _stamp(m) - base
            if t0_s <= t <= t1_s:
                ts.append(t)
                pa.append(m.fluid_pressure)
    return np.asarray(ts), np.asarray(pa)
