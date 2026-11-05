"""Write a copy of a ROS 2 bag with physics-based haze removal (see dehaze.py) on both cameras.

Pass 1 fits the water model once from left frames sampled across the whole bag (with stereo
depth). Pass 2 rewrites every stereo pair through that fixed model. Output images are greyscale,
re-encoded like the source (CompressedImage -> JPEG "mono8; jpeg compressed mono8", Image -> mono8),
exactly as in enhance_bag.py, so its control bag stays the right comparison. Other selected topics
are copied byte-for-byte. Unpaired images are dropped (they cannot get depth).

Usage (lightglue venv for scipy/cv2/rosbags):
  ~/.venvs/lightglue/bin/python scripts/dehaze_bag.py <src_bag> <dst_dir> --mode backscatter \
      --intrinsics camchain.yaml --imucam camchain-imucam.yaml --left cam0 --right cam1 \
      --left-topic T --right-topic T --copy /imu ...
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from dehaze import LIGHTCONE_MAX_GAIN, LightConeModel, StereoDepth, WaterModel, load_rig
from denoise import Denoiser, estimate_sigma
from probe_enhance import decode

PAIR_TOL_NS = 5_000_000
FIT_EVERY = 100   # pairs between fitting samples in pass 1


def pairs(reader: AnyReader, left: str, right: str, extra: set[str]):
    """Yield ("pair", (lconn, lstamp, lmsg), (rconn, rstamp, rmsg)) or ("copy", conn, stamp, raw)."""
    conns = [c for c in reader.connections if c.topic in {left, right} | extra]
    pend: dict[str, tuple] = {}
    for conn, stamp, raw in reader.messages(connections=conns):
        if conn.topic not in (left, right):
            yield ("copy", conn, stamp, raw)
            continue
        msg = reader.deserialize(raw, conn.msgtype)
        t = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        side, other = ("l", "r") if conn.topic == left else ("r", "l")
        if other in pend and abs(pend[other][0] - t) < PAIR_TOL_NS:
            o = pend.pop(other)[1]
            mine = (conn, stamp, msg)
            yield ("pair", mine, o) if side == "l" else ("pair", o, mine)
        else:
            pend[side] = (t, (conn, stamp, msg))


def encode(ts, msg, grey: np.ndarray, quality: int) -> tuple[bytes, str]:
    if hasattr(msg, "format"):
        ok, buf = cv2.imencode(".jpg", grey, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        new = ts.types["sensor_msgs/msg/CompressedImage"](header=msg.header, format="mono8; jpeg compressed mono8",
                                                          data=np.frombuffer(buf, np.uint8))
        return ts.serialize_cdr(new, "sensor_msgs/msg/CompressedImage"), "sensor_msgs/msg/CompressedImage"
    new = ts.types["sensor_msgs/msg/Image"](header=msg.header, height=grey.shape[0], width=grey.shape[1],
                                            encoding="mono8", is_bigendian=0, step=grey.shape[1],
                                            data=np.ascontiguousarray(grey).reshape(-1))
    return ts.serialize_cdr(new, "sensor_msgs/msg/Image"), "sensor_msgs/msg/Image"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    # lightcone-cap: lamp correction with the gain capped at LIGHTCONE_MAX_GAIN; lightcone-ffdnet: the
    # same, then FFDNet with noise map = sensor sigma x per-pixel gain (see denoise.py).
    ap.add_argument("--mode", choices=("backscatter", "full", "lightcone", "lightcone-cap", "lightcone-ffdnet"),
                    required=True)
    ap.add_argument("--intrinsics", type=Path, required=True)
    ap.add_argument("--imucam", type=Path, required=True)
    ap.add_argument("--left", required=True, help="Kalibr camera name of the left camera")
    ap.add_argument("--right", required=True)
    ap.add_argument("--left-topic", required=True)
    ap.add_argument("--right-topic", required=True)
    ap.add_argument("--copy", nargs="*", default=[])
    ap.add_argument("--jpeg-quality", type=int, default=95)
    a = ap.parse_args()
    if a.dst.exists():
        raise SystemExit(f"{a.dst} exists; remove it first")
    stereo = StereoDepth(*load_rig(a.intrinsics, a.imucam, a.left, a.right))
    print(f"stereo depth at {stereo.size}, f={stereo.f:.1f}px baseline={stereo.baseline:.3f}m", flush=True)

    # Pass 1: fit the water once.
    samples, n, tic = [], 0, time.perf_counter()
    with AnyReader([a.src]) as r:
        for kind, *rest in pairs(r, a.left_topic, a.right_topic, set()):
            if kind != "pair":
                continue
            n += 1
            if n % FIT_EVERY:
                continue
            (_, _, lm), (_, _, rm) = rest
            lb, rb = decode(lm), decode(rm)
            if a.mode.startswith("lightcone"):
                samples.append(lb)
                continue
            zl, _ = stereo(cv2.cvtColor(lb, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rb, cv2.COLOR_BGR2GRAY))
            samples.append((lb, zl))
    lamp = a.mode.startswith("lightcone")
    model = LightConeModel.fit(samples) if lamp else WaterModel.fit(samples, a.mode)
    max_gain = LIGHTCONE_MAX_GAIN if a.mode in ("lightcone-cap", "lightcone-ffdnet") else None
    denoiser, sigma_raw, gain_grey = None, 0.0, None
    if a.mode == "lightcone-ffdnet":
        sigma_raw = float(np.median([estimate_sigma(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)) for im in samples]))
        gain_grey = model.gain(max_gain).mean(axis=2)
        denoiser = Denoiser()
        print(f"sensor noise sigma {sigma_raw:.2f} (8-bit); noise map {sigma_raw * gain_grey.min():.1f}-"
              f"{sigma_raw * gain_grey.max():.1f}", flush=True)
    print(f"fitted on {len(samples)} frames of {n} pairs ({time.perf_counter() - tic:.0f}s): {model.describe()}", flush=True)
    a.dst.parent.mkdir(parents=True, exist_ok=True)

    # Pass 2: rewrite every pair with the fixed model.
    ts = get_typestore(Stores.ROS2_JAZZY)
    written, tic = 0, time.perf_counter()
    with AnyReader([a.src]) as r, Writer(a.dst, version=9) as w:
        out = {}

        def conn_for(conn, msgtype):
            key = (conn.topic, msgtype)
            if key not in out:
                out[key] = w.add_connection(conn.topic, msgtype, typestore=ts)
            return out[key]

        for kind, *rest in pairs(r, a.left_topic, a.right_topic, set(a.copy)):
            if kind == "copy":
                conn, stamp, raw = rest
                w.write(conn_for(conn, conn.msgtype), stamp, raw)
                continue
            (lc, ls, lm), (rc, rs, rm) = rest
            lb, rb = decode(lm), decode(rm)
            if lamp:
                zl = zr = None  # one fixed field per sequence; the right camera shares the left's lamp
            else:
                zl, zr = stereo(cv2.cvtColor(lb, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rb, cv2.COLOR_BGR2GRAY))
            for conn, stamp, msg, bgr, z in ((lc, ls, lm, lb, zl), (rc, rs, rm, rb, zr)):
                out_bgr = model.apply(bgr, max_gain) if lamp else model.apply(bgr, z)
                grey = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2GRAY)
                if denoiser is not None:
                    smap = cv2.resize(gain_grey, (grey.shape[1], grey.shape[0])) * sigma_raw
                    grey = denoiser(grey, smap)
                data, mtype = encode(ts, msg, grey, a.jpeg_quality)
                w.write(conn_for(conn, mtype), stamp, data)
            written += 1
            if written % 500 == 0:
                print(f"{written} pairs ({written / (time.perf_counter() - tic):.1f} pairs/s)", flush=True)
    if not lamp:
        (a.dst.parent / f"{a.dst.name}_water_model.json").write_text(json.dumps(
            {"mode": a.mode, "b_inf": model.b_inf.tolist(), "beta_b": model.beta_b.tolist(),
             "beta_d": model.beta_d.tolist(), "gain": model.gain, "fit_frames": len(samples)}, indent=1))
    print(f"done: {written} pairs -> {a.dst}")


if __name__ == "__main__":
    main()
