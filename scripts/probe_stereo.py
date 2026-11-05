"""Diagnose cam0->cam1 matching of cuVSLAM landmarks (the C++ CuvslamTracker::findStereo logic).

Preprocessing matches SVIn's okvis_ros Subscriber for config_stereorig_v1: resize by --scale
(INTER_LINEAR) then CLAHE(clip, tiles). Intrinsics scaled as okvis does (c * s).
For every landmark with a cam0 observation, reports which acceptance test it fails, the
prediction error |LK - projection|, the forward-backward error, and the disparity.

Usage: CUVSLAM_ENTRY=probe_stereo.py run_cuvslam.sh <bag.db3> <calib_dir> [--frames 200]
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import cuvslam
from dataset import Calibration, iter_stereo, load_calibration
from run_cuvslam import to_pose


def okvis_camera(c, scale: float, rig_from_cam: np.ndarray) -> cuvslam.Camera:
    k1, k2, p1, p2 = c.radtan
    return cuvslam.Camera(size=[round(c.width * scale), round(c.height * scale)],
                          principal=[c.cx * scale, c.cy * scale], focal=[c.fx * scale, c.fy * scale],
                          rig_from_camera=to_pose(rig_from_cam),
                          distortion=cuvslam.Distortion(cuvslam.Distortion.Model.Brown, [k1, k2, 0.0, p1, p2]))


def K_of(c, scale: float) -> np.ndarray:
    return np.array([[c.fx * scale, 0, c.cx * scale], [0, c.fy * scale, c.cy * scale], [0, 0, 1.0]])


def lk(a, b, p, init, win, levels):
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01)
    q, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p, init.copy(), winSize=(win, win), maxLevel=levels,
                                        criteria=crit, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    return q, st.ravel().astype(bool)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", type=Path)
    ap.add_argument("calib_dir", type=Path)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--clahe-clip", type=float, default=1.0)
    ap.add_argument("--clahe-tiles", type=int, default=2)
    ap.add_argument("--win", type=int, default=21)
    ap.add_argument("--epi-px", type=float, default=2.0)
    ap.add_argument("--fb-px", type=float, default=1.0)
    ap.add_argument("--half", type=int, default=5)
    ap.add_argument("--min-zncc", type=float, default=0.8)
    ap.add_argument("--min-margin", type=float, default=0.05)
    a = ap.parse_args()
    cal: Calibration = load_calibration(next(a.calib_dir.glob("camchain_*.yaml")), a.calib_dir / "imu.yaml")
    T_C0C1 = np.linalg.inv(cal.T_cam1_cam0)
    rig = cuvslam.Rig([okvis_camera(cal.cam0, a.scale, np.eye(4)), okvis_camera(cal.cam1, a.scale, T_C0C1)], [])
    cfg = cuvslam.Tracker.OdometryConfig(odometry_mode=cuvslam.Tracker.OdometryMode.Multicamera,
                                         rectified_stereo_camera=False, async_sba=False, use_gpu=True,
                                         enable_observations_export=True, enable_landmarks_export=True)
    tracker = cuvslam.Tracker(rig, cfg)
    clahe = cv2.createCLAHE(clipLimit=a.clahe_clip, tileGridSize=(a.clahe_tiles, a.clahe_tiles))
    K1, D1 = K_of(cal.cam1, a.scale), cal.cam1.D()
    R10, t10 = cal.T_cam1_cam0[:3, :3], cal.T_cam1_cam0[:3, 3]

    # Rule C setup: rectification of the scaled pair (OpenCV R, T map cam0 points into cam1).
    K0, D0 = K_of(cal.cam0, a.scale), cal.cam0.D()
    size = (round(cal.cam0.width * a.scale), round(cal.cam0.height * a.scale))
    R1, R2, Pr1, Pr2, _, _, _ = cv2.stereoRectify(K0, D0, K1, D1, size, R10, t10, alpha=0)
    m0 = cv2.initUndistortRectifyMap(K0, D0, R1, Pr1, size, cv2.CV_32FC1)
    m1 = cv2.initUndistortRectifyMap(K1, D1, R2, Pr2, size, cv2.CV_32FC1)
    accC_frame, lk_vs_zncc, zncc_vs_pred = [], [], []

    fails = Counter()
    pred_err, fb_err, disp, depth, off, epi = [], [], [], [], [], []
    accB_frame, epiB = [], []
    accepted_per_frame, cand_per_frame = [], []
    for i, (t, l, r) in enumerate(iter_stereo(a.bag)):
        if i >= a.frames:
            break
        l = clahe.apply(cv2.resize(l, None, fx=a.scale, fy=a.scale))
        r = clahe.apply(cv2.resize(r, None, fx=a.scale, fy=a.scale))
        tracker.track(int(t), [l, r])
        obs = {o.id: (o.u, o.v) for o in tracker.get_last_observations(0)}
        lms = [(lm.id, np.asarray(lm.coords, float)) for lm in tracker.get_last_landmarks() if lm.id in obs]
        if not lms:
            accepted_per_frame.append(0); cand_per_frame.append(0)
            continue
        P0 = np.array([p for _, p in lms])
        uv0 = np.array([obs[k] for k, _ in lms], np.float32)
        P1 = (R10 @ P0.T).T + t10
        front = P1[:, 2] > 0.05
        fails["behind_cam1"] += int((~front).sum())
        pr, _ = cv2.projectPoints(P1, np.zeros(3), np.zeros(3), K1, D1)
        pr = pr[:, 0].astype(np.float32)
        inside = front & (pr[:, 0] >= 0) & (pr[:, 1] >= 0) & (pr[:, 0] < r.shape[1]) & (pr[:, 1] < r.shape[0])
        fails["projects_outside"] += int((front & ~inside).sum())
        idx = np.nonzero(inside)[0]
        cand_per_frame.append(len(idx))
        if len(idx) == 0:
            accepted_per_frame.append(0)
            continue
        p0, pp = uv0[idx], pr[idx]
        p1, stf = lk(l, r, p0, pp, a.win, 3)
        back, stb = lk(r, l, p1, p0, a.win, 3)
        fb = np.linalg.norm(back - p0, axis=1)
        pe = np.linalg.norm(p1 - pp, axis=1)
        ok_st = stf & stb
        fails["lk_status"] += int((~ok_st).sum())
        fails["fb>1px"] += int((ok_st & (fb > 1.0)).sum())
        fails["pred>3px"] += int((ok_st & (fb <= 1.0) & (pe > 3.0)).sum())
        acc = ok_st & (fb <= 1.0) & (pe <= 3.0)
        accepted_per_frame.append(int(acc.sum()))
        # Rule B: geometric acceptance (fb, epipolar distance, loose disparity window, depth > 0).
        n0a = cv2.undistortPoints(p0.reshape(-1, 1, 2), K_of(cal.cam0, a.scale), cal.cam0.D()).reshape(-1, 2)
        n1a = cv2.undistortPoints(p1.reshape(-1, 1, 2), K1, D1).reshape(-1, 2)
        E = skew(t10) @ R10
        la = (E @ np.c_[n0a, np.ones(len(n0a))].T).T
        epa = np.abs(np.sum(la * np.c_[n1a, np.ones(len(n1a))], axis=1)) / np.linalg.norm(la[:, :2], axis=1) * K1[0, 0]
        dpred = np.abs(p0[:, 0] - pp[:, 0])
        win_ok = np.abs(p1[:, 0] - pp[:, 0]) <= np.maximum(10.0, 0.3 * dpred)
        dmeas = p0[:, 0] - p1[:, 0]
        accB = ok_st & (fb <= a.fb_px) & (epa <= a.epi_px) & win_ok & (dmeas > 1.0)
        accB_frame.append(int(accB.sum()))
        epiB.extend(epa[accB].tolist())

        # Rule C: rectified ZNCC scanline search around the predicted disparity.
        lr = cv2.remap(l, *m0, cv2.INTER_LINEAR)
        rr = cv2.remap(r, *m1, cv2.INTER_LINEAR)
        q0 = cv2.undistortPoints(p0.reshape(-1, 1, 2), K0, D0, R=R1, P=Pr1).reshape(-1, 2)
        qp = cv2.undistortPoints(pp.reshape(-1, 1, 2), K1, D1, R=R2, P=Pr2).reshape(-1, 2)
        accC = np.zeros(len(p0), bool)
        zr = np.zeros((len(p0), 2), np.float32)
        for k in range(len(p0)):
            ok, x = zncc_search(lr, rr, q0[k], qp[k, 0], a.half, a.min_zncc, a.min_margin)
            if not ok:
                continue
            ray = np.linalg.inv(Pr2[:, :3]) @ np.array([x, q0[k, 1], 1.0])
            d1 = R2.T @ ray
            pix, _ = cv2.projectPoints(d1.reshape(1, 3), np.zeros(3), np.zeros(3), K1, D1)
            zr[k] = pix.ravel()
            accC[k] = True
            zncc_vs_pred.append(abs(x - qp[k, 0]))
        accC_frame.append(int(accC.sum()))
        both = accC & ok_st & (fb < 0.5)
        lk_vs_zncc.extend(np.linalg.norm(zr[both] - p1[both], axis=1).tolist())
        pred_err.extend(pe[ok_st]); fb_err.extend(fb[ok_st])
        good = ok_st & (fb < 0.5)
        if good.any():
            # Signed offset of clean LK matches from the prediction, and their distance to the
            # epipolar line of the cam0 pixel (undistorted, via the essential matrix).
            off.extend((p1[good] - pp[good]).tolist())
            n0 = cv2.undistortPoints(p0[good].reshape(-1, 1, 2), K_of(cal.cam0, a.scale), cal.cam0.D()).reshape(-1, 2)
            n1 = cv2.undistortPoints(p1[good].reshape(-1, 1, 2), K1, D1).reshape(-1, 2)
            E = skew(t10) @ R10
            l1 = (E @ np.c_[n0, np.ones(len(n0))].T).T
            d = np.abs(np.sum(l1 * np.c_[n1, np.ones(len(n1))], axis=1)) / np.linalg.norm(l1[:, :2], axis=1)
            epi.extend((d * K1[0, 0]).tolist())
        disp.extend(p0[:, 0] - pp[:, 0]); depth.extend(P0[idx, 2])

    pe, fb = np.asarray(pred_err), np.asarray(fb_err)
    q = lambda x: " ".join(f"p{p}={np.percentile(x, p):.2f}" for p in (50, 75, 90, 95))  # noqa: E731
    print(f"frames {len(cand_per_frame)}  candidates/frame median {np.median(cand_per_frame):.0f}  "
          f"accepted/frame median {np.median(accepted_per_frame):.0f}  "
          f"accept rate {sum(accepted_per_frame) / max(1, sum(cand_per_frame)):.1%}")
    print(f"rule B (epipolar): accepted/frame median {np.median(accB_frame):.0f}  "
          f"accept rate {sum(accB_frame) / max(1, sum(cand_per_frame)):.1%}  "
          f"epipolar px of accepted p50 {np.median(epiB):.2f} p95 {np.percentile(epiB, 95):.2f}")
    print(f"rule C (rectified ZNCC): accepted/frame median {np.median(accC_frame):.0f}  "
          f"accept rate {sum(accC_frame) / max(1, sum(cand_per_frame)):.1%}  "
          f"|x - predicted| px p50 {np.median(zncc_vs_pred):.2f} p90 {np.percentile(zncc_vs_pred, 90):.2f}")
    if lk_vs_zncc:
        print("  agreement with clean LK matches px:", q(np.asarray(lk_vs_zncc)), f"n={len(lk_vs_zncc)}")
    print("rejections:", dict(fails))
    print("|LK - prediction| px:", q(pe))
    print("forward-backward px :", q(fb))
    print("predicted disparity px:", q(np.asarray(disp)), " depth m:", q(np.asarray(depth)))
    o = np.asarray(off)
    print(f"clean matches (fb<0.5) n={len(o)}: offset from prediction dx median {np.median(o[:, 0]):+.2f} "
          f"(|dx| p90 {np.percentile(np.abs(o[:, 0]), 90):.2f})  dy median {np.median(o[:, 1]):+.2f} "
          f"(|dy| p90 {np.percentile(np.abs(o[:, 1]), 90):.2f})")
    print("  epipolar distance px:", q(np.asarray(epi)))


def zncc_search(lr: np.ndarray, rr: np.ndarray, q0: np.ndarray, x_pred: float, half: int,
                min_zncc: float, min_margin: float) -> tuple[bool, float]:
    """Best ZNCC match of the left patch at q0 along row q0[1] of rr, near x_pred.

    Window: |x - x_pred| <= max(10, 0.3 * predicted disparity). Sub-pixel by parabola fit.
    Rejects weak peaks (< min_zncc) and ambiguous ones (second peak >2 px away within min_margin).
    """
    xl, y = float(q0[0]), float(q0[1])
    w = max(10.0, 0.3 * abs(xl - x_pred))
    x0, x1 = x_pred - w, x_pred + w
    side = 2 * half + 1
    if (xl - half < 0 or xl + half >= lr.shape[1] or y - half < 0 or y + half >= lr.shape[0]
            or x0 - half < 0 or x1 + half >= rr.shape[1]):
        return False, 0.0
    patch = cv2.getRectSubPix(lr, (side, side), (xl, y))
    strip_w = int(np.ceil(x1 - x0)) + side
    strip = cv2.getRectSubPix(rr, (strip_w, side), (x0 + (strip_w - 1) / 2.0 - half, y))
    resp = cv2.matchTemplate(strip.astype(np.float32), patch.astype(np.float32), cv2.TM_CCOEFF_NORMED).ravel()
    if resp.size < 3 or not np.all(np.isfinite(resp)):
        return False, 0.0
    i = int(np.argmax(resp))
    best = float(resp[i])
    far = np.abs(np.arange(resp.size) - i) > 2
    second = float(resp[far].max()) if far.any() else -1.0
    if best < min_zncc or best - second < min_margin or i == 0 or i == resp.size - 1:
        return False, 0.0
    a_, b_, c_ = resp[i - 1], resp[i], resp[i + 1]
    den = a_ - 2 * b_ + c_
    sub = 0.5 * (a_ - c_) / den if abs(den) > 1e-9 else 0.0
    return True, x0 - half + (strip_w - 1) / 2.0 - (strip_w - side) / 2.0 + i + sub


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


if __name__ == "__main__":
    main()
