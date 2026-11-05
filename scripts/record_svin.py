"""Record SVIn trajectories during a bag replay and write them as TUM files (cam0 frame).

/okvis_odometry  nav_msgs/Odometry, T_W_S when the config has trackedBodyFrame: S
                 (run_svin.sh sets that); converted to cam0 with T_SC0 from the config.
/loop_closure_path nav_msgs/Path of keyframe poses after pose-graph optimisation. These come
                 from okvis keyframe_pose, which is already the cam0 pose, so no conversion.

Writes on SIGINT/SIGTERM: <out>/<tag>_odom.tum and <out>/<tag>_loop.tum.
Usage: record_svin.py <svin_config.yaml> <out_dir> <tag>
"""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

import numpy as np
import rclpy
import yaml
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.node import Node
from rclpy.parameter import Parameter

from geometry import mat_to_quat, quat_to_mat


def load_T_SC0(config: Path) -> np.ndarray:
    text = config.read_text().split("\n", 1)[1]  # drop the OpenCV '%YAML:1.0' directive
    cfg = yaml.safe_load(text)
    t = np.asarray(cfg["cameras"][0]["T_SC"], dtype=float).reshape(4, 4)
    if not np.allclose(t[:3, :3] @ t[:3, :3].T, np.eye(3), atol=1e-3):
        raise ValueError(f"{config}: cameras[0].T_SC rotation is not orthonormal")
    return t


class Recorder(Node):
    def __init__(self, T_SC0: np.ndarray) -> None:
        super().__init__("svin_bench_recorder", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.T_SC0 = T_SC0
        self.odom: list[list[float]] = []
        self.loop: list[list[float]] = []
        self.create_subscription(Odometry, "/okvis_odometry", self.on_odom, 200)
        self.create_subscription(PathMsg, "/loop_closure_path", self.on_path, 10)

    def on_odom(self, m: Odometry) -> None:
        p, q = m.pose.pose.position, m.pose.pose.orientation
        T_WS = np.eye(4)
        T_WS[:3, :3] = quat_to_mat(q.x, q.y, q.z, q.w)
        T_WS[:3, 3] = [p.x, p.y, p.z]
        T = T_WS @ self.T_SC0
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self.odom.append([t, *T[:3, 3], *mat_to_quat(T[:3, :3])])

    def on_path(self, m: PathMsg) -> None:
        # Each message is the full optimised keyframe path; keep the latest.
        self.loop = [[ps.header.stamp.sec + ps.header.stamp.nanosec * 1e-9,
                      ps.pose.position.x, ps.pose.position.y, ps.pose.position.z,
                      ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w]
                     for ps in m.poses]


def write_tum(rows: list[list[float]], path: Path) -> None:
    a = np.asarray(rows, dtype=float).reshape(-1, 8)
    a = a[np.argsort(a[:, 0], kind="stable")] if len(a) else a
    np.savetxt(path, a, fmt="%.9f")
    print(f"wrote {len(a)} poses to {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("tag")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Recorder(load_T_SC0(a.config))
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    while not stop["flag"]:
        rclpy.spin_once(node, timeout_sec=0.2)
    write_tum(node.odom, a.out / f"{a.tag}_odom.tum")
    write_tum(node.loop, a.out / f"{a.tag}_loop.tum")
    node.destroy_node()
    rclpy.try_shutdown()
    sys.exit(0)


if __name__ == "__main__":
    main()
