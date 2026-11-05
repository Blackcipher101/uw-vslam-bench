"""Overlay the sliding window of sonar points on the left camera image (original distorted image, radtan).

  python3 project.py --traj <tum> [--extrinsic svin|<calib.json>] [--start-s 100 --duration-s 20 --step 2]
         [--window-s 4.2] [--future-s 0] --out-dir out/overlay [--video out/overlay.mp4] [--png-every 25]
Points are coloured by sonar range (0.3..3.6 m, turbo). Time is seconds since the first image.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sonar_cam.data import (Bag, GT_COLMAP, TOPIC_LEFT, Trajectory, decode_compressed_image, decode_image_stamp,
                            load_camchain, load_svin_rig, sonar_in_cam0)
from sonar_cam.geometry import ping_range, project, sonar_world_points, world_to_cam


def load_extrinsic(spec: str):
    """-> (T_c0_So, heading_sign)."""
    if spec == "svin":
        return sonar_in_cam0(load_svin_rig()), 1.0
    r = json.load(open(spec))["refined"]
    return np.array(r["T_c0_So"]), float(r["heading_sign"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default=GT_COLMAP)
    ap.add_argument("--extrinsic", default="svin")
    ap.add_argument("--start-s", type=float, default=100.0)
    ap.add_argument("--duration-s", type=float, default=20.0)
    ap.add_argument("--step", type=int, default=2, help="use every N-th left image")
    ap.add_argument("--window-s", type=float, default=4.2, help="past window of pings (one head revolution ~4.2 s)")
    ap.add_argument("--future-s", type=float, default=0.0)
    ap.add_argument("--dt-s", type=float, default=0.0, help="sonar time offset: true ping time = stamp + dt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--video", default=None)
    ap.add_argument("--png-every", type=int, default=25)
    ap.add_argument("--png-best", type=int, default=3, help="also save the K frames with most in-image sonar points")
    ap.add_argument("--out-width", type=int, default=800)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bag, traj = Bag(), Trajectory(a.traj)
    cc = load_camchain()["cam0"]
    T_c0_So, sign = load_extrinsic(a.extrinsic)
    pings = bag.pings()
    tp = np.array([p.t for p in pings]) + a.dt_s
    rng = np.array([ping_range(p.intensity, p.max_range_m)[0] for p in pings])
    head = sign * np.array([p.head_deg for p in pings])
    ok = np.isfinite(rng) & traj.valid(tp)
    P_w = np.full((len(pings), 3), np.nan)
    P_w[ok] = sonar_world_points(traj.poses(tp[ok]), T_c0_So, rng[ok], head[ok])
    rows = bag.rowids(TOPIC_LEFT)
    t_img = np.array([decode_image_stamp(bag.blob(r)) for r in rows])
    sel = np.nonzero((t_img >= t_img[0] + a.start_s) & (t_img <= t_img[0] + a.start_s + a.duration_s))[0][:: a.step]
    scale = a.out_width / cc["size"][0]
    vw = None
    n_in = []
    best = []  # (n_in_image, name, image)
    for k, i in enumerate(sel):
        if not traj.valid(t_img[i:i + 1])[0]:
            continue
        img = cv2.imdecode(np.frombuffer(decode_compressed_image(bag.blob(rows[i])), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        m = ok & (tp > t_img[i] - a.window_s) & (tp <= t_img[i] + a.future_s)
        Pc = world_to_cam(traj.poses(t_img[i:i + 1])[0], P_w[m])
        uv, inn = project(Pc, cc["K"], cc["dist"], cc["size"])
        for (u, v), r in zip(uv[inn] * scale, rng[m][inn]):
            c = cv2.applyColorMap(np.uint8([[np.clip((r - 0.3) / 3.3, 0, 1) * 255]]), cv2.COLORMAP_TURBO)[0, 0]
            cv2.circle(img, (int(u), int(v)), 4, tuple(int(x) for x in c), -1, cv2.LINE_AA)
        n_in.append(int(inn.sum()))
        cv2.putText(img, f"t={t_img[i] - t_img[0]:.1f}s sonar pts in image: {int(inn.sum())}/{int(m.sum())}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if a.video:
            if vw is None:
                vw = cv2.VideoWriter(a.video, cv2.VideoWriter_fourcc(*"mp4v"), 6.0, (img.shape[1], img.shape[0]))
            vw.write(img)
        name = f"overlay_{t_img[i] - t_img[0]:07.2f}.png"
        if k % a.png_every == 0:
            cv2.imwrite(str(out / name), img)
        best = sorted(best + [(int(inn.sum()), name, img.copy())], key=lambda x: -x[0])[: a.png_best]
    for n, name, img in best:
        cv2.imwrite(str(out / f"best{n:03d}_{name}"), img)
    if vw is not None:
        vw.release()
    print(f"frames {len(n_in)}, in-image sonar points per frame: mean {np.mean(n_in):.1f} max {np.max(n_in)}")


if __name__ == "__main__":
    main()
