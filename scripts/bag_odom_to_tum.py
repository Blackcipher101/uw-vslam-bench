"""Extract a pose topic (nav_msgs/Odometry or geometry_msgs/PoseStamped) from a bag as TUM.

Stamps are the message header stamps (s). The pose is whatever the topic carries (frame_id <-
child_frame_id for Odometry); the frames are printed so the evaluator can be told what to expect.

Usage: ~/.venvs/rosbags/bin/python scripts/bag_odom_to_tum.py <bag_dir_or_file> <topic> <out.tum>
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rosbags.highlevel import AnyReader


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bag", type=Path)
    ap.add_argument("topic")
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    rows, frames = [], set()
    with AnyReader([a.bag]) as r:
        conns = [c for c in r.connections if c.topic == a.topic]
        if not conns:
            raise SystemExit(f"topic {a.topic} not in {a.bag}")
        for conn, _, raw in r.messages(connections=conns):
            m = r.deserialize(raw, conn.msgtype)
            pose = m.pose.pose if hasattr(m.pose, "pose") else m.pose
            frames.add((m.header.frame_id, getattr(m, "child_frame_id", "")))
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            p, q = pose.position, pose.orientation
            rows.append((t, p.x, p.y, p.z, q.x, q.y, q.z, q.w))
    rows.sort()
    with a.out.open("w") as f:
        for row in rows:
            f.write(" ".join(f"{v:.9f}" for v in row) + "\n")
    print(f"wrote {len(rows)} poses to {a.out}; frames (parent, child): {sorted(frames)}")


if __name__ == "__main__":
    main()
