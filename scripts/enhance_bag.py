"""Write a copy of a ROS 2 bag whose stereo images went through the enhancement pipeline.

Pipeline per image (same for every dataset): decode -> Enhancer(mode) -> greyscale -> re-encode in
the source's message type (CompressedImage: JPEG of the grey image at --jpeg-quality, format
"mono8; jpeg compressed mono8"; Image: mono8). Headers and bag timestamps are kept. Other selected
topics are copied byte-for-byte. Use --mode none to build the control bag: identical decode /
greyscale / re-encode path without enhancement, so any difference is due to enhancement only.

Usage (lightglue venv):
  ~/.venvs/lightglue/bin/python scripts/enhance_bag.py <src_bag> <dst_dir> --mode funie-native \
      --images /cam_l/image /cam_r/image --copy /imu ...
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from enhance import MODES, Enhancer
from probe_enhance import decode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path, help="output bag directory (must not exist)")
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--images", nargs="+", required=True, help="image topics to process")
    ap.add_argument("--copy", nargs="*", default=[], help="topics copied unchanged")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.dst.exists():
        raise SystemExit(f"{a.dst} exists; remove it first")

    ts = get_typestore(Stores.ROS2_JAZZY)
    enh = Enhancer(a.mode, a.device)
    wanted = set(a.images) | set(a.copy)
    n_img, tic = 0, time.perf_counter()
    with AnyReader([a.src]) as r, Writer(a.dst, version=9) as w:
        conns = [c for c in r.connections if c.topic in wanted]
        missing = wanted - {c.topic for c in conns}
        if missing:
            raise SystemExit(f"source bag lacks topics {sorted(missing)}")
        out = {c.id: w.add_connection(c.topic, c.msgtype, typestore=ts) for c in conns}
        for conn, stamp, raw in r.messages(connections=conns):
            if conn.topic not in a.images:
                w.write(out[conn.id], stamp, raw)
                continue
            msg = r.deserialize(raw, conn.msgtype)
            grey = cv2.cvtColor(enh(decode(msg)), cv2.COLOR_BGR2GRAY)
            if hasattr(msg, "format"):
                ok, buf = cv2.imencode(".jpg", grey, [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_quality])
                if not ok:
                    raise RuntimeError(f"JPEG encode failed at {stamp}")
                new = ts.types["sensor_msgs/msg/CompressedImage"](
                    header=msg.header, format="mono8; jpeg compressed mono8", data=np.frombuffer(buf, np.uint8))
                data = ts.serialize_cdr(new, "sensor_msgs/msg/CompressedImage")
            else:
                new = ts.types["sensor_msgs/msg/Image"](
                    header=msg.header, height=grey.shape[0], width=grey.shape[1], encoding="mono8",
                    is_bigendian=0, step=grey.shape[1], data=np.ascontiguousarray(grey).reshape(-1))
                data = ts.serialize_cdr(new, "sensor_msgs/msg/Image")
            w.write(out[conn.id], stamp, data)
            n_img += 1
            if n_img % 1000 == 0:
                print(f"{n_img} images ({n_img / (time.perf_counter() - tic):.1f} img/s)", flush=True)
    print(f"done: {n_img} images through mode={a.mode} -> {a.dst}")


if __name__ == "__main__":
    main()
