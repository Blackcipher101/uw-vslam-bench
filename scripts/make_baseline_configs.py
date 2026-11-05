"""Write ORB-SLAM3 (stereo-inertial) and Basalt configs from the same Kalibr files uw_ct_vio uses.

Pinhole + equidistant (Kannala-Brandt) cameras only (ORB-SLAM3 "KannalaBrandt8", Basalt "kb4").
The physically left camera becomes camera 1 / camera 0 of the baselines.

Usage: python make_baseline_configs.py <intrinsics.yaml> <imucam.yaml> <imu.yaml> <out_dir> \
           --left cam1 --right cam0 --fps 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def mat_yaml(name: str, T: np.ndarray) -> str:
    data = ", ".join(f"{v:.12g}" for v in T.ravel())
    return f"{name}: !!opencv-matrix\n  rows: 4\n  cols: 4\n  dt: f\n  data: [{data}]\n"


def pose_json(T: np.ndarray) -> dict:
    q = Rotation.from_matrix(T[:3, :3]).as_quat()  # x y z w
    return {"px": T[0, 3], "py": T[1, 3], "pz": T[2, 3], "qx": q[0], "qy": q[1], "qz": q[2], "qw": q[3]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("intrinsics", type=Path)
    ap.add_argument("imucam", type=Path)
    ap.add_argument("imu", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--basalt-template", type=Path,
                    help="Basalt calib json to copy the 'vignette' entry from (cereal requires it; VIO ignores it)")
    a = ap.parse_args()
    ci, ce, iy = (yaml.safe_load(p.read_text()) for p in (a.intrinsics, a.imucam, a.imu))
    for c in (a.left, a.right):
        if ci[c]["distortion_model"] != "equidistant":
            raise SystemExit(f"{c}: only equidistant cameras are supported here")
    T_ci = {c: np.array(ce[c]["T_cam_imu"], float) for c in (a.left, a.right)}
    T_ic = {c: np.linalg.inv(T) for c, T in T_ci.items()}           # camera pose in the IMU frame
    T_l_r = T_ci[a.left] @ T_ic[a.right]                             # right camera pose in left frame
    if T_l_r[0, 3] <= 0:
        raise SystemExit("right camera is not to the right of the left one")
    w, h = ci[a.left]["resolution"]
    a.out.mkdir(parents=True, exist_ok=True)

    # ORB-SLAM3, settings format 1.0 (as Examples/Stereo-Inertial/TUM-VI.yaml)
    lines = ['%YAML:1.0', 'File.version: "1.0"', 'Camera.type: "KannalaBrandt8"']
    for i, c in ((1, a.left), (2, a.right)):
        fx, fy, cx, cy = ci[c]["intrinsics"]
        k = ci[c]["distortion_coeffs"]
        lines += [f"Camera{i}.fx: {fx}", f"Camera{i}.fy: {fy}", f"Camera{i}.cx: {cx}", f"Camera{i}.cy: {cy}"]
        lines += [f"Camera{i}.k{j + 1}: {k[j]}" for j in range(4)]
    s = "\n".join(lines) + "\n" + mat_yaml("Stereo.T_c1_c2", T_l_r)
    s += "".join(f"Camera{i}.overlappingBegin: 0\nCamera{i}.overlappingEnd: {w - 1}\n" for i in (1, 2))
    s += (f"Camera.width: {w}\nCamera.height: {h}\nCamera.fps: {int(round(a.fps))}\nCamera.RGB: 0\nStereo.ThDepth: 40.0\n")
    s += mat_yaml("IMU.T_b_c1", T_ic[a.left])
    s += (f"IMU.NoiseGyro: {iy['gyroscope_noise_density']}\nIMU.NoiseAcc: {iy['accelerometer_noise_density']}\n"
          f"IMU.GyroWalk: {iy['gyroscope_random_walk']}\nIMU.AccWalk: {iy['accelerometer_random_walk']}\n"
          f"IMU.Frequency: {float(iy['update_rate'])}\n")
    s += ("ORBextractor.nFeatures: 1000\nORBextractor.scaleFactor: 1.2\nORBextractor.nLevels: 8\n"
          "ORBextractor.iniThFAST: 20\nORBextractor.minThFAST: 7\n"
          "Viewer.KeyFrameSize: 0.05\nViewer.KeyFrameLineWidth: 1.0\nViewer.GraphLineWidth: 0.9\n"
          "Viewer.PointSize: 2.0\nViewer.CameraSize: 0.08\nViewer.CameraLineWidth: 3.0\n"
          "Viewer.ViewpointX: 0.0\nViewer.ViewpointY: -0.7\nViewer.ViewpointZ: -3.5\nViewer.ViewpointF: 500.0\n")
    (a.out / "orbslam3_stereo_inertial.yaml").write_text(s)

    # Basalt calibration (as data/tumvi_512_ds_calib.json, kb4 cameras)
    intr = []
    for c in (a.left, a.right):
        fx, fy, cx, cy = ci[c]["intrinsics"]
        k = ci[c]["distortion_coeffs"]
        intr.append({"camera_type": "kb4", "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
                                                          "k1": k[0], "k2": k[1], "k3": k[2], "k4": k[3]}})
    calib = {"value0": {
        "T_imu_cam": [pose_json(T_ic[a.left]), pose_json(T_ic[a.right])],
        "intrinsics": intr, "resolution": [[w, h], [w, h]],
        "calib_accel_bias": [0.0] * 9, "calib_gyro_bias": [0.0] * 12,
        "imu_update_rate": float(iy["update_rate"]),
        "accel_noise_std": [iy["accelerometer_noise_density"]] * 3,
        "gyro_noise_std": [iy["gyroscope_noise_density"]] * 3,
        "accel_bias_std": [iy["accelerometer_random_walk"]] * 3,
        "gyro_bias_std": [iy["gyroscope_random_walk"]] * 3,
        "T_mocap_world": pose_json(np.eye(4)), "T_imu_marker": pose_json(np.eye(4)),
        "mocap_time_offset_ns": 0, "mocap_to_imu_offset_ns": 0, "cam_time_offset_ns": 0}}
    if a.basalt_template:
        calib["value0"]["vignette"] = json.loads(a.basalt_template.read_text())["value0"]["vignette"]
    (a.out / "basalt_calib.json").write_text(json.dumps(calib, indent=1))
    print(f"wrote {a.out}/orbslam3_stereo_inertial.yaml and basalt_calib.json "
          f"(baseline {np.linalg.norm(T_l_r[:3, 3]):.4f} m, {w}x{h} @ {a.fps} Hz)")


if __name__ == "__main__":
    main()
