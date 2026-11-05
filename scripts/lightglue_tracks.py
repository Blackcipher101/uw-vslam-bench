"""Precompute SuperPoint + LightGlue feature tracks for SVIn's `track_file` frontend.

For every stereo pair of the bag (paired like SVIn's stereo_sync, by left stamp) this:
  1. preprocesses exactly as okvis_ros's Subscriber for config_stereorig_v1 (resize by --scale,
     INTER_LINEAR, CLAHE), so pixels are in the image SVIn's frontend sees;
  2. extracts SuperPoint in cam0 and cam1;
  3. matches cam0(t-1) -> cam0(t) with LightGlue, keeps essential-matrix RANSAC inliers and
     carries their track ids forward (other keypoints start new tracks);
  4. matches cam0(t) -> cam1(t), keeps matches passing the rectified epipolar test, and
     triangulates them (rectified disparity) into the cam0 frame.

Output (little-endian binary, read by okvis TrackFileReader):
  file header : b"SVTRK1\\0\\0", uint32 version=1, uint32 reserved
  per frame   : int64 t_ns (left image header stamp), uint32 n
  per track   : uint64 id, float32 u0 v0 u1 v1 x y z, uint8 flags (bit0 has_uv1, bit1 has_point)
Pixels are distorted (raw) coordinates in the preprocessed image; x y z in metres, cam0 frame.

Usage (bash, ROS sourced for rosbag2_py):
  PYTHONPATH=scripts:$PYTHONPATH ~/.venvs/lightglue/bin/python scripts/lightglue_tracks.py \\
      <bag.db3> <calib_dir> <out.svtrk> [--max-kp 1024]
"""
from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint

from dataset import NS, Calibration, iter_stereo, load_calibration


MAGIC = b"SVTRK1\0\0"
VERSION = 1
TRACK = np.dtype([("id", "<u8"), ("u0", "<f4"), ("v0", "<f4"), ("u1", "<f4"), ("v1", "<f4"),
                  ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("flags", "u1")])
HAS_UV1, HAS_POINT = 1, 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path)
    ap.add_argument("calib_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--clahe-clip", type=float, default=1.0)
    ap.add_argument("--clahe-tiles", type=int, default=2)
    ap.add_argument("--max-kp", type=int, default=1024)
    ap.add_argument("--epi-px", type=float, default=2.0)
    ap.add_argument("--min-disp-px", type=float, default=0.5)
    ap.add_argument("--max-disp-px", type=float, default=200.0)
    ap.add_argument("--ransac-px", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole bag")
    a = ap.parse_args()

    cal: Calibration = load_calibration(next(a.calib_dir.glob("camchain_*.yaml")), a.calib_dir / "imu.yaml")
    s = a.scale
    # Intrinsics scaled as okvis does (c * s), matching VioParametersReader.
    K0 = cal.cam0.K() * np.array([[s], [s], [1.0]])
    K1 = cal.cam1.K() * np.array([[s], [s], [1.0]])
    D0, D1 = cal.cam0.D(), cal.cam1.D()
    size = (round(cal.cam0.width * s), round(cal.cam0.height * s))
    T10 = cal.T_cam1_cam0  # maps cam0 points into cam1
    R1, R2, P1, P2, *_ = cv2.stereoRectify(K0, D0, K1, D1, size, np.ascontiguousarray(T10[:3, :3]),
                                           T10[:3, 3].reshape(3, 1).copy(), alpha=0)
    f, cx, cy = P1[0, 0], P1[0, 2], P1[1, 2]
    baseline_m = -P2[0, 3] / P2[0, 0]
    if not baseline_m > 0:
        raise SystemExit(f"unexpected rectified baseline {baseline_m} m (cam1 must be right of cam0)")
    rvec10, _ = cv2.Rodrigues(np.ascontiguousarray(T10[:3, :3]))
    clahe = cv2.createCLAHE(clipLimit=a.clahe_clip, tileGridSize=(a.clahe_tiles, a.clahe_tiles))

    dev = torch.device("cuda")
    ext = SuperPoint(max_num_keypoints=a.max_kp).eval().to(dev)
    lg = LightGlue(features="superpoint").eval().to(dev)

    def prep(img: np.ndarray) -> np.ndarray:
        return clahe.apply(cv2.resize(img, size, interpolation=cv2.INTER_LINEAR))

    def extract(img: np.ndarray) -> dict:
        return ext.extract(torch.from_numpy(img).to(dev, torch.float32)[None, None] / 255.0)

    def match(fa: dict, fb: dict) -> tuple[np.ndarray, np.ndarray]:
        mm = lg({"image0": fa, "image1": fb})["matches"][0].cpu().numpy()
        return mm[:, 0], mm[:, 1]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    prev_f, prev_ids, next_id = None, None, 1
    n_frames, last_t, tic0 = 0, -1, time.perf_counter()
    stats = {"kp": [], "stereo": [], "tracked": [], "reproj": [], "ms": []}
    with open(a.out, "wb") as fo, torch.inference_mode():
        fo.write(MAGIC + struct.pack("<II", VERSION, 0))
        for t, left, right in iter_stereo(a.bag):
            if a.max_frames and n_frames >= a.max_frames:
                break
            if t <= last_t:
                print(f"skip non-increasing stamp {t} (prev {last_t})", flush=True)
                continue
            last_t = t
            tic = time.perf_counter()
            f0, f1 = extract(prep(left)), extract(prep(right))
            k0 = f0["keypoints"][0].cpu().numpy().astype(np.float64)
            k1 = f1["keypoints"][0].cpu().numpy().astype(np.float64)
            rec = np.zeros(len(k0), TRACK)
            rec["u0"], rec["v0"] = k0[:, 0], k0[:, 1]

            # Temporal: carry ids along RANSAC-inlier matches from the previous left frame.
            ids = np.zeros(len(k0), np.uint64)
            if prev_f is not None and len(k0) and len(prev_ids):
                jp, jc = match(prev_f, f0)
                if len(jc) >= 8:
                    kp_prev = prev_f["keypoints"][0].cpu().numpy().astype(np.float64)
                    pa = cv2.undistortPoints(kp_prev[jp].reshape(-1, 1, 2), K0, D0, P=K0).reshape(-1, 2)
                    pb = cv2.undistortPoints(k0[jc].reshape(-1, 1, 2), K0, D0, P=K0).reshape(-1, 2)
                    _, mask = cv2.findEssentialMat(pa, pb, K0, cv2.RANSAC, 0.999, a.ransac_px)
                    if mask is not None:
                        inl = mask.ravel().astype(bool)
                        ids[jc[inl]] = prev_ids[jp[inl]]
            n_tracked = int((ids > 0).sum())
            fresh = ids == 0
            ids[fresh] = np.arange(next_id, next_id + fresh.sum(), dtype=np.uint64)
            next_id += int(fresh.sum())
            rec["id"] = ids

            # Stereo: epipolar test in the rectified pair, then depth from disparity.
            n_stereo = 0
            if len(k0) and len(k1):
                i0, i1 = match(f0, f1)
                if len(i0):
                    r0 = cv2.undistortPoints(k0[i0].reshape(-1, 1, 2), K0, D0, R=R1, P=P1).reshape(-1, 2)
                    r1 = cv2.undistortPoints(k1[i1].reshape(-1, 1, 2), K1, D1, R=R2, P=P2).reshape(-1, 2)
                    disp = r0[:, 0] - r1[:, 0]
                    ok = ((np.abs(r0[:, 1] - r1[:, 1]) < a.epi_px) & (disp >= a.min_disp_px)
                          & (disp <= a.max_disp_px))
                    i0, i1, r0, disp = i0[ok], i1[ok], r0[ok], disp[ok]
                    z = f * baseline_m / disp
                    p_rect = np.stack([(r0[:, 0] - cx) * z / f, (r0[:, 1] - cy) * z / f, z], 1)
                    p_c0 = p_rect @ R1  # R1 maps cam0 -> rectified, so R1^T maps back (row form)
                    rec["u1"][i0], rec["v1"][i0] = k1[i1, 0], k1[i1, 1]
                    rec["x"][i0], rec["y"][i0], rec["z"][i0] = p_c0[:, 0], p_c0[:, 1], p_c0[:, 2]
                    rec["flags"][i0] = HAS_UV1 | HAS_POINT
                    n_stereo = len(i0)
                    if n_stereo:  # self-check: the 3D point must reproject onto the cam1 pixel
                        proj, _ = cv2.projectPoints(p_c0, rvec10, T10[:3, 3], K1, D1)
                        stats["reproj"].append(float(np.median(np.linalg.norm(proj.reshape(-1, 2) - k1[i1], axis=1))))

            fo.write(struct.pack("<qI", t, len(rec)))
            fo.write(rec.tobytes())
            prev_f, prev_ids = f0, ids
            n_frames += 1
            ms = (time.perf_counter() - tic) * 1e3
            for k, v in (("kp", len(k0)), ("stereo", n_stereo), ("tracked", n_tracked), ("ms", ms)):
                stats[k].append(v)
            if n_frames % 100 == 0:
                med = {k: np.median(v[-100:]) for k, v in stats.items() if v}
                el = time.perf_counter() - tic0
                print(f"{n_frames} frames t={t / NS:.2f} | last100 median kp {med['kp']:.0f} stereo "
                      f"{med['stereo']:.0f} tracked {med['tracked']:.0f} reproj {med.get('reproj', float('nan')):.3f}px "
                      f"{med['ms']:.0f} ms | {n_frames / el:.1f} fps", flush=True)

    print("== summary")
    print(f"frames {n_frames}, tracks created {next_id - 1}, file {a.out} ({a.out.stat().st_size / 2**20:.0f} MiB)")
    for k in ("kp", "stereo", "tracked"):
        v = np.array(stats[k])
        print(f"{k:8s} median {np.median(v):6.0f}  p10 {np.percentile(v, 10):6.0f}  min {v.min():6.0f}")
    r = np.array(stats["reproj"])
    print(f"stereo 3D->cam1 reprojection, per-frame median: median {np.median(r):.3f}px  max {r.max():.3f}px")
    print(f"latency ms median {np.median(stats['ms']):.1f}  p95 {np.percentile(stats['ms'], 95):.1f}; "
          f"gpu peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
