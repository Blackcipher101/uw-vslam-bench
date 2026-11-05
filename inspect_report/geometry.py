"""Stereo rectification, SGBM depth, box -> 3D / metric size, and cross-frame finding merge.

Frames: 'rect' = rectified left camera (x right, y down, z forward, metres);
'world' = VIO world frame via T_world_rect = T_world_cam0 @ blkdiag(R1^T, 1).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DEPTH_MIN_M, DEPTH_MAX_M = 0.3, 15.0
MIN_VALID_FRACTION = 0.15  # of inner-box pixels that must have a valid depth


@dataclass
class Rectifier:
    map_l: tuple
    map_r: tuple
    R1: np.ndarray
    P1: np.ndarray
    baseline_m: float
    size: tuple  # (w, h)


def build_rectifier(cc: dict, size: tuple) -> Rectifier:
    """OpenCV convention: R,T map cam0 coordinates into cam1 coordinates (= Kalibr T_cn_cnm1)."""
    w, h = size
    c0, c1 = cc["cam0"], cc["cam1"]
    if (w, h) != tuple(c0["size"]):
        raise ValueError(f"image size {(w, h)} != calibrated {c0['size']}")
    R, T = cc["T_c1_c0"][:3, :3], cc["T_c1_c0"][:3, 3].reshape(3, 1)
    if c0["model"] == "equidistant":
        R1, R2, P1, P2, _ = cv2.fisheye.stereoRectify(
            c0["K"], c0["dist"], c1["K"], c1["dist"], (w, h), R, T, cv2.fisheye.CALIB_ZERO_DISPARITY
        )
        m0 = cv2.fisheye.initUndistortRectifyMap(c0["K"], c0["dist"], R1, P1, (w, h), cv2.CV_16SC2)
        m1 = cv2.fisheye.initUndistortRectifyMap(c1["K"], c1["dist"], R2, P2, (w, h), cv2.CV_16SC2)
    elif c0["model"] == "radtan":
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
            c0["K"], c0["dist"], c1["K"], c1["dist"], (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        m0 = cv2.initUndistortRectifyMap(c0["K"], c0["dist"], R1, P1, (w, h), cv2.CV_16SC2)
        m1 = cv2.initUndistortRectifyMap(c1["K"], c1["dist"], R2, P2, (w, h), cv2.CV_16SC2)
    else:
        raise ValueError(f"unsupported distortion model {c0['model']}")
    return Rectifier(m0, m1, R1, P1, float(np.linalg.norm(T)), (w, h))


def rectify(rc: Rectifier, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        cv2.remap(left, *rc.map_l, cv2.INTER_LINEAR),
        cv2.remap(right, *rc.map_r, cv2.INTER_LINEAR),
    )


def T_world_rect(T_world_cam0: np.ndarray, R1: np.ndarray) -> np.ndarray:
    """p_rect = R1 p_cam0  =>  p_world = T_world_cam0 @ [R1^T 0; 0 1] @ p_rect."""
    T = np.eye(4)
    T[:3, :3] = R1.T
    return T_world_cam0 @ T


def sgbm_depth(rect_l: np.ndarray, rect_r: np.ndarray, rc: Rectifier, scale: float = 0.5) -> np.ndarray:
    """Depth (m, float32, NaN invalid) at `scale` x the rectified resolution."""
    gl = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY) if rect_l.ndim == 3 else rect_l
    gr = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY) if rect_r.ndim == 3 else rect_r
    if scale != 1.0:
        gl = cv2.resize(gl, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gr = cv2.resize(gr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    bs = 5
    nd = 16 * int(np.ceil(gl.shape[1] / 8 / 16))  # ~ width/8 disparities, multiple of 16
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=nd,
        blockSize=bs,
        P1=8 * bs * bs,
        P2=32 * bs * bs,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = sgbm.compute(gl, gr).astype(np.float32) / 16.0
    f = float(rc.P1[0, 0]) * scale
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = f * rc.baseline_m / disp
    depth[(disp <= 0.5) | ~np.isfinite(depth) | (depth < DEPTH_MIN_M) | (depth > DEPTH_MAX_M)] = np.nan
    return depth


def locate_box(box: list[float], depth: np.ndarray, P1: np.ndarray, scale: float = 0.5) -> dict | None:
    """box = [x0,y0,x1,y1] in rectified full-res pixels. Returns dict(depth_m, xyz_rect, width_m, height_m,
    valid_fraction) using the median depth of the inner 50% of the box, or None if too few valid pixels."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = x1 - x0, y1 - y0
    ix0, ix1 = int((cx - w / 4) * scale), int(np.ceil((cx + w / 4) * scale))
    iy0, iy1 = int((cy - h / 4) * scale), int(np.ceil((cy + h / 4) * scale))
    H, W = depth.shape
    ix0, ix1, iy0, iy1 = max(ix0, 0), min(ix1, W), max(iy0, 0), min(iy1, H)
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    patch = depth[iy0:iy1, ix0:ix1]
    valid = patch[np.isfinite(patch)]
    frac = valid.size / patch.size
    if valid.size == 0 or frac < MIN_VALID_FRACTION:
        return None
    z = float(np.median(valid))
    fx, fy, px, py = P1[0, 0], P1[1, 1], P1[0, 2], P1[1, 2]
    return {
        "depth_m": z,
        "xyz_rect": [float((cx - px) * z / fx), float((cy - py) * z / fy), z],
        "width_m": float(w * z / fx),
        "height_m": float(h * z / fy),
        "valid_fraction": float(frac),
    }


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def merge_detections(
    dets: list[dict], radius_m: float = 1.0, gap_s: float = 2.5, min_iou: float = 0.3
) -> list[dict]:
    """Greedy time-ordered clustering of per-frame detections into findings.

    A detection joins an existing track of the same class when both have a world position and
    are within radius_m of the track mean; if either lacks a position, when the track's last
    detection is within gap_s and the boxes overlap (IoU >= min_iou, same pixel frame).
    Each det: {cls, score, t, frame, box, xyz_world|None, width_m|None, height_m|None, ...}.
    """
    tracks: list[dict] = []
    for d in sorted(dets, key=lambda d: d["t"]):
        best, best_d = None, None
        for tr in tracks:
            if tr["cls"] != d["cls"]:
                continue
            last = tr["dets"][-1]
            if d.get("xyz_world") is not None and tr["pos"] is not None:
                dist = float(np.linalg.norm(np.asarray(d["xyz_world"]) - tr["pos"]))
                ok = dist <= radius_m
            else:
                dist = 0.0
                ok = d["t"] - last["t"] <= gap_s and iou(d["box"], last["box"]) >= min_iou
            if ok and (best is None or dist < best_d):
                best, best_d = tr, dist
        if best is None:
            best = {"cls": d["cls"], "dets": [], "pos": None}
            tracks.append(best)
        best["dets"].append(d)
        pts = [(x["score"], np.asarray(x["xyz_world"])) for x in best["dets"] if x.get("xyz_world") is not None]
        best["pos"] = (sum(s * p for s, p in pts) / sum(s for s, _ in pts)) if pts else None
    out = []
    for tr in tracks:
        ds = tr["dets"]
        top = max(ds, key=lambda x: x["score"])
        sizes = [(x["width_m"], x["height_m"]) for x in ds if x.get("width_m") is not None]
        out.append(
            {
                "cls": tr["cls"],
                "score_max": float(top["score"]),
                "score_mean": float(np.mean([x["score"] for x in ds])),
                "n_obs": len(ds),
                "t_first": min(x["t"] for x in ds),
                "t_last": max(x["t"] for x in ds),
                "frames": [x["frame"] for x in ds],
                "best_frame": top["frame"],
                "best_box": top["box"],
                "xyz_world": None if tr["pos"] is None else [float(v) for v in tr["pos"]],
                "range_m": top.get("depth_m"),
                "width_m": float(np.median([s[0] for s in sizes])) if sizes else None,
                "height_m": float(np.median([s[1] for s in sizes])) if sizes else None,
            }
        )
    out.sort(key=lambda f: -f["score_max"])
    for i, f in enumerate(out, 1):
        f["id"] = f"F{i:02d}"
    return out
