"""Check what cuVSLAM's exported tracks look like before wiring them into OKVIS.

For the first N stereo frames: project each exported landmark (rig = cam0 frame) into
cam0 with and without the radtan model and compare to the cam0 observation with the
same id. The smaller residual tells which pixel convention the observations use.
Also reports observations per frame and track lifetimes.

Usage: run via run_cuvslam.sh's environment:
  CUVSLAM_ENTRY=probe_tracks.py run_cuvslam.sh <bag.db3> <calib_dir> [--frames 400] [--scale 0.5]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import cuvslam
from dataset import iter_stereo, load_calibration
from run_cuvslam import build_rig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", type=Path)
    ap.add_argument("calib_dir", type=Path)
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--scale", type=float, default=0.5)
    a = ap.parse_args()
    cal = load_calibration(next(a.calib_dir.glob("camchain_*.yaml")), a.calib_dir / "imu.yaml")
    cfg = cuvslam.Tracker.OdometryConfig(
        odometry_mode=cuvslam.Tracker.OdometryMode.Multicamera, rectified_stereo_camera=False,
        async_sba=False, use_gpu=True, enable_observations_export=True, enable_landmarks_export=True)
    tracker = cuvslam.Tracker(build_rig(cal, a.scale, False), cfg)
    c = cal.cam0
    K = c.K().copy()
    K[:2] *= a.scale
    K[0, 2], K[1, 2] = (c.cx + 0.5) * a.scale - 0.5, (c.cy + 0.5) * a.scale - 0.5
    first_seen, last_seen = {}, {}
    n_obs0, n_obs1, n_lm = [], [], []
    res_dist, res_pin = [], []
    for i, (t, left, right) in enumerate(iter_stereo(a.bag)):
        if i >= a.frames:
            break
        left = cv2.resize(left, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
        tracker.track(int(t), [left, right])
        o0 = tracker.get_last_observations(0)
        o1 = tracker.get_last_observations(1)
        lms = tracker.get_last_landmarks()
        n_obs0.append(len(o0)); n_obs1.append(len(o1)); n_lm.append(len(lms))
        for o in o0:
            first_seen.setdefault(o.id, i); last_seen[o.id] = i
        obs = {o.id: (o.u, o.v) for o in o0}
        pts, uv = [], []
        for lm in lms:
            if lm.id in obs and lm.coords[2] > 0.1:
                pts.append(lm.coords); uv.append(obs[lm.id])
        if len(pts) >= 5:
            pts = np.asarray(pts, np.float64); uv = np.asarray(uv)
            pd, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, c.D())
            pp, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, np.zeros(4))
            res_dist.extend(np.linalg.norm(pd[:, 0] - uv, axis=1))
            res_pin.extend(np.linalg.norm(pp[:, 0] - uv, axis=1))
    life = np.array([last_seen[k] - first_seen[k] + 1 for k in first_seen])
    print(f"frames {len(n_obs0)}  obs/frame cam0 median {np.median(n_obs0):.0f} cam1 {np.median(n_obs1):.0f}  "
          f"landmarks/frame median {np.median(n_lm):.0f}")
    print(f"tracks {len(life)}  lifetime frames median {np.median(life):.0f} p90 {np.percentile(life, 90):.0f} "
          f"max {life.max()}")
    for name, r in (("radtan (raw pixels)", res_dist), ("pinhole (undistorted)", res_pin)):
        r = np.asarray(r)
        print(f"reprojection vs obs, {name}: median {np.median(r):.2f} px  p90 {np.percentile(r, 90):.2f} px  n {len(r)}")


if __name__ == "__main__":
    main()
