"""Measure SuperPoint + LightGlue as a candidate SVIn frontend on an AFRL bag.

Preprocessing matches SVIn's okvis_ros Subscriber for config_stereorig_v1 (resize by --scale,
INTER_LINEAR, then CLAHE), so the numbers compare with probe_stereo.py / the cuVSLAM frontend.
Per frame it reports:
  kp      SuperPoint keypoints in cam0
  stereo  cam0<->cam1 LightGlue matches passing the rectified epipolar test (|dy| < --epi-px,
          disparity in (0, --max-disp-px])
  temp    cam0(t-1)<->cam0(t) matches, and how many survive essential-matrix RANSAC
  track   length (frames) of the feature tracks chained through temporal inliers
  ms      GPU time for extract(cam0, cam1) + stereo match + temporal match

Usage (bash, ROS sourced for rosbag2_py):
  PYTHONPATH=scripts:$PYTHONPATH ~/.venvs/lightglue/bin/python scripts/probe_lightglue.py \
      <bag.db3> <calib_dir> --start-s 0 --frames 300
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint

from dataset import NS, Calibration, iter_stereo, load_calibration



def to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(img).to(device, torch.float32)[None, None] / 255.0


def match(m: LightGlue, f0: dict, f1: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (idx0, idx1) of LightGlue matches."""
    out = m({"image0": f0, "image1": f1})
    mm = out["matches"][0].cpu().numpy()
    return mm[:, 0], mm[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path)
    ap.add_argument("calib_dir", type=Path)
    ap.add_argument("--start-s", type=float, default=0.0, help="skip this many seconds of the bag")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--clahe-clip", type=float, default=1.0)
    ap.add_argument("--clahe-tiles", type=int, default=2)
    ap.add_argument("--max-kp", type=int, default=1024)
    ap.add_argument("--epi-px", type=float, default=2.0)
    ap.add_argument("--max-disp-px", type=float, default=200.0)
    ap.add_argument("--ransac-px", type=float, default=1.0)
    a = ap.parse_args()

    cal: Calibration = load_calibration(next(a.calib_dir.glob("camchain_*.yaml")), a.calib_dir / "imu.yaml")
    s = a.scale
    K0 = cal.cam0.K() * np.array([[s], [s], [1.0]])
    K1 = cal.cam1.K() * np.array([[s], [s], [1.0]])
    D0, D1 = cal.cam0.D(), cal.cam1.D()
    size = (round(cal.cam0.width * s), round(cal.cam0.height * s))
    T = cal.T_cam1_cam0
    R1, R2, P1, P2, *_ = cv2.stereoRectify(K0, D0, K1, D1, size, np.ascontiguousarray(T[:3, :3]), T[:3, 3].reshape(3, 1).copy(), alpha=0)
    clahe = cv2.createCLAHE(clipLimit=a.clahe_clip, tileGridSize=(a.clahe_tiles, a.clahe_tiles))

    dev = torch.device("cuda")
    ext = SuperPoint(max_num_keypoints=a.max_kp).eval().to(dev)
    lg = LightGlue(features="superpoint").eval().to(dev)

    def prep(img: np.ndarray) -> np.ndarray:
        return clahe.apply(cv2.resize(img, size, interpolation=cv2.INTER_LINEAR))

    rows, track_len_done = [], []
    prev, prev_tracks, t0 = None, None, None
    with torch.inference_mode():
        for t, left, right in iter_stereo(a.bag):
            t0 = t if t0 is None else t0
            if (t - t0) / NS < a.start_s:
                continue
            if len(rows) >= a.frames:
                break
            l, r = prep(left), prep(right)
            torch.cuda.synchronize()
            tic = time.perf_counter()
            f0 = ext.extract(to_tensor(l, dev))
            f1 = ext.extract(to_tensor(r, dev))
            i0, i1 = match(lg, f0, f1)
            if prev is not None:
                j_prev, j_cur = match(lg, prev, f0)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - tic) * 1e3

            k0 = f0["keypoints"][0].cpu().numpy()
            k1 = f1["keypoints"][0].cpu().numpy()
            # Stereo: rectify both ends and apply the epipolar / disparity test.
            ok = np.zeros(0, bool)
            if len(i0):  # cv2.undistortPoints returns None for an empty input
                u0 = cv2.undistortPoints(k0[i0].reshape(-1, 1, 2), K0, D0, R=R1, P=P1).reshape(-1, 2)
                u1 = cv2.undistortPoints(k1[i1].reshape(-1, 1, 2), K1, D1, R=R2, P=P2).reshape(-1, 2)
                disp = u0[:, 0] - u1[:, 0]
                ok = (np.abs(u0[:, 1] - u1[:, 1]) < a.epi_px) & (disp > 0) & (disp <= a.max_disp_px)

            # Temporal: essential-matrix RANSAC on undistorted pixels of cam0(t-1) -> cam0(t).
            n_temp = n_inl = 0
            tracks = np.ones(len(k0), dtype=int)  # track length ending at each cam0 keypoint
            if prev is not None and len(j_cur) >= 8:
                kp_prev = prev["keypoints"][0].cpu().numpy()
                a_ = cv2.undistortPoints(kp_prev[j_prev].reshape(-1, 1, 2), K0, D0, P=K0).reshape(-1, 2)
                b_ = cv2.undistortPoints(k0[j_cur].reshape(-1, 1, 2), K0, D0, P=K0).reshape(-1, 2)
                _, mask = cv2.findEssentialMat(a_, b_, K0, cv2.RANSAC, 0.999, a.ransac_px)
                inl = mask.ravel().astype(bool) if mask is not None else np.zeros(len(j_cur), bool)
                n_temp, n_inl = len(j_cur), int(inl.sum())
                tracks[j_cur[inl]] = prev_tracks[j_prev[inl]] + 1
                ended = np.ones(len(prev_tracks), bool)
                ended[j_prev[inl]] = False
                track_len_done.extend(prev_tracks[ended].tolist())
            rows.append((len(k0), int(ok.sum()), n_temp, n_inl, ms))
            prev, prev_tracks = f0, tracks
            n = len(rows)
            print(f"{n:4d} t={(t - t0) / NS:7.2f}s kp={len(k0)} stereo={int(ok.sum())}/{len(i0)} "
                  f"temp={n_inl}/{n_temp} ms={ms:.1f}", flush=True)
            if n % 50 == 0:
                r_ = np.array(rows)
                print(f"-- {n} frames: median kp={np.median(r_[:, 0]):.0f} stereo={np.median(r_[:, 1]):.0f} "
                      f"temp_inl={np.median(r_[1:, 3]):.0f} ms={np.median(r_[:, 4]):.1f}", flush=True)

    if prev_tracks is not None:
        track_len_done.extend(prev_tracks.tolist())
    r_ = np.array(rows)
    tl = np.array(track_len_done)
    tl = tl[tl > 1]
    print("== summary")
    print(f"frames {len(r_)} from t={a.start_s:.0f}s  scale={s} max_kp={a.max_kp}")
    for name, col in (("keypoints", 0), ("stereo", 1), ("temporal_matches", 2), ("temporal_inliers", 3)):
        v = r_[1:, col] if col >= 2 else r_[:, col]
        print(f"{name:17s} median {np.median(v):6.0f}  p10 {np.percentile(v, 10):6.0f}  min {v.min():6.0f}")
    print(f"track length (>1) median {np.median(tl):.0f}  p90 {np.percentile(tl, 90):.0f}  max {tl.max()}"
          if len(tl) else "no tracks")
    print(f"latency ms median {np.median(r_[1:, 4]):.1f}  p95 {np.percentile(r_[1:, 4], 95):.1f}  "
          f"max {r_[1:, 4].max():.1f}   gpu peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
