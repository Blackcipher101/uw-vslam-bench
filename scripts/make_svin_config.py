"""Build an SVIn config for a Kalibr-calibrated stereo-IMU rig from a template config.

Takes camera intrinsics/distortion from one camchain (e.g. calibrated under water) and camera-IMU
extrinsics + time shift from a camchain-imucam (e.g. calibrated in air: the rigid geometry does
not change with the medium), and IMU noise from a Kalibr imu.yaml. Everything outside the camera,
IMU and rate fields is kept from the template.

Usage:
  python3 scripts/make_svin_config.py <template.yaml> <intrinsics_camchain.yaml> \
      <imucam_camchain.yaml> <imu.yaml> <out.yaml> --left cam1 --right cam0 --camera-rate 20
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import yaml

DISTORTION = {"radtan": "radialtangential", "equidistant": "equidistant"}
BASELINE_TOL_M = 0.005


def fmt(v: float) -> str:
    return f"{v:.12g}"


def camera_block(intr: dict, T_cam_imu: np.ndarray) -> str:
    if intr["camera_model"] != "pinhole" or intr["distortion_model"] not in DISTORTION:
        raise SystemExit(f"unsupported camera {intr['camera_model']}/{intr['distortion_model']}")
    T_SC = np.linalg.inv(T_cam_imu)
    rows = ",\n         ".join(", ".join(fmt(x) for x in r) for r in T_SC)
    fx, fy, cx, cy = intr["intrinsics"]
    w, h = intr["resolution"]
    return (f"    - {{T_SC:\n       [ {rows}],\n"
            f"       image_dimension: [{w}, {h}],\n"
            f"       distortion_coefficients: [{', '.join(fmt(x) for x in intr['distortion_coeffs'])}],\n"
            f"       distortion_type: {DISTORTION[intr['distortion_model']]},\n"
            f"       focal_length: [{fmt(fx)}, {fmt(fy)}],\n"
            f"       principal_point: [{fmt(cx)}, {fmt(cy)}]}}\n")


def set_key(text: str, key: str, value: str) -> str:
    new, n = re.subn(rf"^(\s*{key}:)[^\n#]*", rf"\g<1> {value} ", text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit(f"template has no '{key}:' line")
    return new


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("template", type=Path)
    ap.add_argument("intrinsics", type=Path)
    ap.add_argument("imucam", type=Path)
    ap.add_argument("imu", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--left", default="cam0", help="Kalibr camera that is physically left")
    ap.add_argument("--right", default="cam1")
    ap.add_argument("--camera-rate", type=int, required=True)
    ap.add_argument("--resize-factor", type=float, default=1.0)
    a = ap.parse_args()

    intr = yaml.safe_load(a.intrinsics.read_text())
    ext = yaml.safe_load(a.imucam.read_text())
    imu = yaml.safe_load(a.imu.read_text())
    T_ci = {c: np.array(ext[c]["T_cam_imu"], dtype=float) for c in (a.left, a.right)}

    # The two calibrations must describe the same rig: compare stereo baselines.
    T_rl = T_ci[a.right] @ np.linalg.inv(T_ci[a.left])
    b_ext = float(np.linalg.norm(T_rl[:3, 3]))
    cams = sorted(intr)  # cam0, cam1
    T_intr = np.array(intr[cams[1]]["T_cn_cnm1"], dtype=float)
    b_intr = float(np.linalg.norm(T_intr[:3, 3]))
    if abs(b_ext - b_intr) > BASELINE_TOL_M:
        raise SystemExit(f"baseline mismatch: extrinsics {b_ext:.4f} m vs intrinsics chain {b_intr:.4f} m")
    # Left must be left: in the left camera's frame the right camera sits at +x.
    if (np.linalg.inv(T_rl))[0, 3] <= 0:
        raise SystemExit(f"{a.right} is not to the right of {a.left}; swap --left/--right")

    shifts = {ext[c].get("timeshift_cam_imu", 0.0) for c in (a.left, a.right)}
    shift = float(np.mean(list(shifts)))
    text = a.template.read_text()
    start, end = text.index("cameras:\n"), text.index("camera_params:")
    cams_txt = "cameras:\n" + camera_block(intr[a.left], T_ci[a.left]) + "\n" + \
        camera_block(intr[a.right], T_ci[a.right]) + "\n"
    text = text[:start] + cams_txt + text[end:]
    text = set_key(text, "camera_rate", str(a.camera_rate))
    text = set_key(text, "sigma_g_c", fmt(imu["gyroscope_noise_density"]))
    text = set_key(text, "sigma_a_c", fmt(imu["accelerometer_noise_density"]))
    text = set_key(text, "sigma_gw_c", fmt(imu["gyroscope_random_walk"]))
    text = set_key(text, "sigma_aw_c", fmt(imu["accelerometer_random_walk"]))
    text = set_key(text, "imu_rate", str(int(round(imu["update_rate"]))))
    # SVIn uses t_image - imageDelay; Kalibr defines t_imu = t_cam + timeshift_cam_imu.
    text = set_key(text, "imageDelay", fmt(-shift))
    text = set_key(text, "resizeFactor", repr(float(a.resize_factor)))  # must parse as a real
    text = re.sub(r"(T_BS:\s*\[)[^\]]*\]",
                  r"\g<1>1.0, 0.0, 0.0, 0.0,\n         0.0, 1.0, 0.0, 0.0,\n         0.0, 0.0, 1.0, 0.0,\n"
                  r"         0.0, 0.0, 0.0, 1.0]", text, count=1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text)
    print(f"wrote {a.out}: left={a.left} right={a.right} baseline {b_ext:.4f} m (intrinsics chain "
          f"{b_intr:.4f} m), imageDelay {-shift:.6f} s, imu {imu['update_rate']} Hz")


if __name__ == "__main__":
    main()
