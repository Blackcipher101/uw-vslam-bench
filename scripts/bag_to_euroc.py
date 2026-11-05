"""Export a stereo + IMU ROS 2 bag to the EuRoC/ASL folder layout (ORB-SLAM3, Basalt).

  <out>/mav0/cam0/data/<t_ns>.png + data.csv   (left camera)
  <out>/mav0/cam1/data/<t_ns>.png + data.csv   (right camera)
  <out>/mav0/imu0/data.csv                     (t_ns, w_xyz [rad/s], a_xyz [m/s^2])
  <out>/times.txt                              (left stamps, ns; ORB-SLAM3 example input)

Images are written as raw grey PNGs (no processing). Stereo pairs are matched within 5 ms and both
images are stamped with the LEFT header stamp, as EuRoC tools expect synchronised pairs.

Usage: python bag_to_euroc.py <bag> <out_dir> --left T --right T --imu T
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader

PAIR_TOL_NS = 5_000_000


def grey(msg) -> np.ndarray:
    if hasattr(msg, "format"):
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError("failed to decode compressed image")
        return img
    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    if msg.encoding == "mono8":
        return np.ascontiguousarray(a[:, :msg.width])
    img = a[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if msg.encoding == "rgb8" else cv2.COLOR_BGR2GRAY)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--imu", required=True)
    a = ap.parse_args()
    root = a.out / "mav0"
    for d in ("cam0/data", "cam1/data", "imu0"):
        (root / d).mkdir(parents=True, exist_ok=True)
    pend: dict[str, tuple[int, object]] = {}
    stamps, n_imu = [], 0
    with AnyReader([a.bag]) as r, open(root / "imu0" / "data.csv", "w") as fi:
        fi.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
                 "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
        conns = [c for c in r.connections if c.topic in (a.left, a.right, a.imu)]
        if len({c.topic for c in conns}) != 3:
            raise SystemExit("bag lacks one of the requested topics")
        last_imu = -1
        for conn, _, raw in r.messages(connections=conns):
            m = r.deserialize(raw, conn.msgtype)
            t = m.header.stamp.sec * 10**9 + m.header.stamp.nanosec
            if conn.topic == a.imu:
                if t <= last_imu:
                    continue  # EuRoC readers require strictly increasing IMU stamps
                last_imu = t
                w, acc = m.angular_velocity, m.linear_acceleration
                fi.write(f"{t},{w.x:.9f},{w.y:.9f},{w.z:.9f},{acc.x:.9f},{acc.y:.9f},{acc.z:.9f}\n")
                n_imu += 1
                continue
            side, other = ("l", "r") if conn.topic == a.left else ("r", "l")
            if other in pend and abs(pend[other][0] - t) < PAIR_TOL_NS:
                to, mo = pend.pop(other)
                tl, ml, mr = (t, m, mo) if side == "l" else (to, mo, m)
                cv2.imwrite(str(root / "cam0" / "data" / f"{tl}.png"), grey(ml))
                cv2.imwrite(str(root / "cam1" / "data" / f"{tl}.png"), grey(mr))
                stamps.append(tl)
            else:
                pend[side] = (t, m)
    stamps.sort()
    for cam in ("cam0", "cam1"):
        with open(root / cam / "data.csv", "w") as f:
            f.write("#timestamp [ns],filename\n")
            f.writelines(f"{t},{t}.png\n" for t in stamps)
    (a.out / "times.txt").write_text("".join(f"{t}\n" for t in stamps))
    print(f"exported {len(stamps)} stereo pairs, {n_imu} IMU samples -> {a.out}")


if __name__ == "__main__":
    main()
