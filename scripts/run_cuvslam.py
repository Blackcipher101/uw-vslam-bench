"""Run PyCuVSLAM offline on an AFRL underwater stereo-VI bag.

Rig frame = cam0 (left camera, OpenCV optical). Output trajectories are
T_world_cam0 in TUM format (t x y z qx qy qz qw), stamped with left-image time.
Images are radtan-distorted; cuVSLAM handles that natively via its Brown model.

Usage: run_cuvslam.sh <bag.db3> <calib_dir> <out_dir> [--modes stereo,stereo_imu] [--scale 0.5]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import cuvslam
from dataset import NS, Calibration, CameraCalib, iter_stereo, load_calibration, read_imu
from geometry import mat_to_quat


def to_pose(t: np.ndarray) -> cuvslam.Pose:
    return cuvslam.Pose(rotation=mat_to_quat(t[:3, :3]), translation=t[:3, 3].tolist())


def make_camera(c: CameraCalib, scale: float, rig_from_cam: np.ndarray) -> cuvslam.Camera:
    k1, k2, p1, p2 = c.radtan
    # Distortion acts on normalised coordinates, so it is unchanged by resizing;
    # only the pixel intrinsics scale. Pixel centres: (u + 0.5) * s - 0.5.
    return cuvslam.Camera(
        size=[round(c.width * scale), round(c.height * scale)],
        principal=[(c.cx + 0.5) * scale - 0.5, (c.cy + 0.5) * scale - 0.5],
        focal=[c.fx * scale, c.fy * scale],
        rig_from_camera=to_pose(rig_from_cam),
        distortion=cuvslam.Distortion(cuvslam.Distortion.Model.Brown, [k1, k2, 0.0, p1, p2]))


def build_rig(cal: Calibration, scale: float, with_imu: bool) -> cuvslam.Rig:
    cams = [make_camera(cal.cam0, scale, np.eye(4)),
            make_camera(cal.cam1, scale, np.linalg.inv(cal.T_cam1_cam0))]
    imus = []
    if with_imu:
        n = cal.imu
        imus.append(cuvslam.ImuCalibration(
            rig_from_imu=to_pose(cal.cam0.T_cam_imu), frequency=n.rate_hz,
            gyroscope_noise_density=n.gyroscope_noise_density,
            gyroscope_random_walk=n.gyroscope_random_walk,
            accelerometer_noise_density=n.accelerometer_noise_density,
            accelerometer_random_walk=n.accelerometer_random_walk))
    return cuvslam.Rig(cams, imus)


def run_mode(bag: Path, cal: Calibration, imu: tuple, mode: str, scale: float, slam: bool,
             out: Path, log) -> dict:
    with_imu = mode == "stereo_imu"
    cfg = cuvslam.Tracker.OdometryConfig(
        odometry_mode=(cuvslam.Tracker.OdometryMode.Inertial if with_imu
                       else cuvslam.Tracker.OdometryMode.Multicamera),
        rectified_stereo_camera=False,
        async_sba=False,  # deterministic offline replay
        use_gpu=True)
    slam_cfg = cuvslam.Tracker.SlamConfig(sync_mode=True, use_gpu=True) if slam else None
    tracker = cuvslam.Tracker(build_rig(cal, scale, with_imu), cfg, slam_cfg)

    imu_t, imu_acc, imu_gyro = imu
    imu_i = 0
    rows, lat_ms = [], []
    n_frames = n_lost = 0
    t_start = time.perf_counter()
    for t, left, right in iter_stereo(bag):
        if scale != 1.0:
            left = cv2.resize(left, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if with_imu:
            while imu_i < len(imu_t) and imu_t[imu_i] <= t:
                tracker.register_imu_measurement(0, cuvslam.ImuMeasurement(
                    timestamp_ns=int(imu_t[imu_i]),
                    linear_accelerations=imu_acc[imu_i].tolist(),
                    angular_velocities=imu_gyro[imu_i].tolist()))
                imu_i += 1
        t0 = time.perf_counter()
        est, _ = tracker.track(int(t), [left, right])
        lat_ms.append((time.perf_counter() - t0) * 1e3)
        n_frames += 1
        if est.world_from_rig is None:
            n_lost += 1
        else:
            p = est.world_from_rig.pose
            rows.append([t / NS, *np.asarray(p.translation), *np.asarray(p.rotation)])
        if n_frames % 500 == 0:
            el = time.perf_counter() - t_start
            print(f"[{mode}] {n_frames} frames, lost {n_lost}, track {np.median(lat_ms):.1f} ms median, "
                  f"{n_frames / el:.1f} fps wall", file=log, flush=True)

    tag = f"cuvslam_{mode}"
    np.savetxt(out / f"{tag}_odom.tum", np.asarray(rows), fmt="%.9f")
    res = {"method": "cuvslam", "mode": mode, "scale": scale, "frames": n_frames,
           "lost_frames": n_lost, "track_ms_mean": float(np.mean(lat_ms)),
           "track_ms_p95": float(np.percentile(lat_ms, 95)), "track_ms_max": float(np.max(lat_ms)),
           "files": [f"{tag}_odom.tum"]}
    if slam:
        poses = tracker.get_all_slam_poses()
        sm = np.asarray([[ps.timestamp_ns / NS, *np.asarray(ps.pose.translation),
                          *np.asarray(ps.pose.rotation)] for ps in poses])
        if len(sm):
            sm = sm[np.argsort(sm[:, 0])]
            np.savetxt(out / f"{tag}_slam.tum", sm, fmt="%.9f")
            res["files"].append(f"{tag}_slam.tum")
        res["loop_closures"] = len(tracker.slam.get_loop_closure_poses())
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", type=Path)
    ap.add_argument("calib_dir", type=Path, help="dir with camchain_*.yaml and imu.yaml")
    ap.add_argument("out", type=Path)
    ap.add_argument("--modes", default="stereo,stereo_imu")
    ap.add_argument("--scale", type=float, default=0.5, help="image resize factor (1600x1200 native)")
    ap.add_argument("--no-slam", action="store_true")
    a = ap.parse_args()
    if not 0.1 <= a.scale <= 1.0:
        sys.exit("--scale must be in [0.1, 1.0]")
    a.out.mkdir(parents=True, exist_ok=True)
    camchain = next(a.calib_dir.glob("camchain_*.yaml"))
    cal = load_calibration(camchain, a.calib_dir / "imu.yaml")
    cuvslam.warm_up_gpu()
    imu = read_imu(a.bag, cal.timeshift_cam_imu_s)
    print(f"cuvslam {cuvslam.__version__}; {len(imu[0])} imu samples; calib {camchain.name}", flush=True)
    results = []
    for mode in a.modes.split(","):
        res = run_mode(a.bag, cal, imu, mode, a.scale, not a.no_slam, a.out, sys.stdout)
        print(json.dumps(res), flush=True)
        results.append(res)
    (a.out / "cuvslam_summary.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
