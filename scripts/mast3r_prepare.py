"""Export the left camera of an AFRL bag as a MASt3R-SLAM image folder.

MASt3R-SLAM's generic loader (``RGBFiles``) reads ``*.png`` in natural order and assigns
fake timestamps ``index / 30``. This writes the frames plus ``stamps.txt`` (one bag stamp
in ns per exported frame, same order) so ``mast3r_to_tum.py`` can map its trajectory back
to bag time, and ``intrinsics.yaml`` in the format its ``--calib`` flag expects
(width, height, [fx, fy, cx, cy, k1, k2, p1, p2]) for the exported resolution.

The folder must not contain a path component named tum/euroc/eth3d/7-scenes/realsense/
webcam, or MASt3R-SLAM picks a different loader.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage

from dataset import LEFT_TOPIC, _decode_gray, _reader, _stamp, load_calibration

RESERVED_DIRS = {"tum", "euroc", "eth3d", "7-scenes", "realsense", "webcam"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path)
    ap.add_argument("camchain", type=Path)
    ap.add_argument("imu_yaml", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--scale", type=float, default=0.5, help="image resize factor (MASt3R works at 512 px)")
    ap.add_argument("--stride", type=int, default=2, help="keep every N-th left frame")
    args = ap.parse_args()
    if not (0 < args.scale <= 1) or args.stride < 1:
        raise SystemExit("--scale must be in (0, 1] and --stride >= 1")
    bad = RESERVED_DIRS & set(args.out_dir.resolve().parts)
    if bad:
        raise SystemExit(f"out_dir contains reserved component(s) {bad}; MASt3R-SLAM would misread it")

    cam = load_calibration(args.camchain, args.imu_yaml).cam0
    img_dir = args.out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    if any(img_dir.glob("*.png")):
        raise SystemExit(f"{img_dir} already has frames; remove it first")

    w, h = round(cam.width * args.scale), round(cam.height * args.scale)
    stamps: list[int] = []
    r = _reader(args.bag, [LEFT_TOPIC])
    n_seen, last_t = 0, -1
    while r.has_next():
        _, data, _ = r.read_next()
        m = deserialize_message(data, CompressedImage)
        t = _stamp(m)
        if t <= last_t:  # stamps must be strictly increasing to map back unambiguously
            print(f"skip out-of-order/duplicate frame t={t} (prev {last_t})", flush=True)
            continue
        last_t = t
        n_seen += 1
        if (n_seen - 1) % args.stride:
            continue
        img = _decode_gray(m)
        if img.shape != (cam.height, cam.width):
            raise RuntimeError(f"frame t={t} is {img.shape[::-1]}, calibration says {cam.width}x{cam.height}")
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(img_dir / f"{len(stamps):06d}.png"), img)
        stamps.append(t)
        if len(stamps) % 500 == 0:
            print(f"exported {len(stamps)} frames (t={t})", flush=True)

    (args.out_dir / "stamps.txt").write_text("".join(f"{t}\n" for t in stamps))
    s = args.scale
    calib = [cam.fx * s, cam.fy * s, cam.cx * s, cam.cy * s, *cam.radtan]  # radtan is scale-free
    (args.out_dir / "intrinsics.yaml").write_text(
        f"width: {w}\nheight: {h}\n# fx, fy, cx, cy, k1, k2, p1, p2 (cam0, scaled by {s})\n"
        f"calibration: [{', '.join(f'{v:.9g}' for v in calib)}]\n")
    print(f"done: {len(stamps)} of {n_seen} frames, {w}x{h}, -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
