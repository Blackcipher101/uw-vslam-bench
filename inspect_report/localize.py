"""Stage 3: stereo depth (SGBM), 3D position + metric size per detection, cross-frame merge.

Only detections with score >= --thr in --mode are localised. Without a pose the finding is box-only
(no world position); without valid stereo depth no size/range is reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import geometry as G
from ir_common import StageTimer


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.15, help="fixed detection score threshold")
    ap.add_argument("--mode", default="base", choices=["base", "ens"])
    ap.add_argument("--scale", type=float, default=0.5, help="SGBM working scale")
    ap.add_argument("--merge-radius", type=float, default=1.0, help="metres")
    ap.add_argument("--max-findings", type=int, default=30)
    ap.add_argument("--classes", nargs="*", default=None, help="restrict to these classes")
    a = ap.parse_args(argv)
    out = Path(a.out)
    timer = StageTimer(out)
    kfs = {k["id"]: k for k in json.load(open(out / "keyframes.json"))["frames"]}
    rect = json.load(open(out / "rect.json"))
    P1 = np.array(rect["P1"])
    rc = G.Rectifier(None, None, np.array(rect["R1"]), P1, rect["baseline_m"], tuple(rect["size"]))
    det = json.load(open(out / "detections.json"))["frames"]

    located, n_det, n_depth = [], 0, 0
    for kid, dets in det.items():
        sel = [d for d in dets if d["mode"] == a.mode and d["score"] >= a.thr and (a.classes is None or d["cls"] in a.classes)]
        if not sel:
            continue
        kf = kfs[kid]
        rl = cv2.imread(str(out / kf["left"]))
        rr = cv2.imread(str(out / kf["right"]))
        with timer.time("sgbm_depth", frame=kid):
            depth = G.sgbm_depth(rl, rr, rc, a.scale)
        T = None if kf["T_world_rect"] is None else np.array(kf["T_world_rect"])
        with timer.time("locate_and_pose", frame=kid, n=len(sel)):
            for d in sel:
                n_det += 1
                rec = {"cls": d["cls"], "score": d["score"], "t": kf["t"], "frame": kid, "box": d["box"],
                       "xyz_world": None, "width_m": None, "height_m": None, "depth_m": None}
                loc = G.locate_box(d["box"], depth, P1, a.scale)
                if loc is not None:
                    n_depth += 1
                    rec.update(depth_m=loc["depth_m"], width_m=loc["width_m"], height_m=loc["height_m"],
                               xyz_rect=loc["xyz_rect"])
                    if T is not None:
                        rec["xyz_world"] = (T @ np.array([*loc["xyz_rect"], 1.0]))[:3].tolist()
                located.append(rec)
    with timer.time("merge"):
        findings = G.merge_detections(located, radius_m=a.merge_radius)
    n_total = len(findings)
    findings = findings[: a.max_findings]
    # crops from the best frame (padded 15%), saved for the VLM and report thumbnails
    (out / "crops").mkdir(exist_ok=True)
    for f in findings:
        img = cv2.imread(str(out / kfs[f["best_frame"]]["left"]))
        x0, y0, x1, y1 = f["best_box"]
        pw, ph = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
        H, W = img.shape[:2]
        c = img[max(int(y0 - ph), 0): min(int(y1 + ph), H), max(int(x0 - pw), 0): min(int(x1 + pw), W)]
        s = 448 / max(c.shape[:2])
        if s < 1:
            c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        f["crop"] = f"crops/{f['id']}.jpg"
        cv2.imwrite(str(out / f["crop"]), c, [cv2.IMWRITE_JPEG_QUALITY, 90])
    json.dump({"thr": a.thr, "mode": a.mode, "n_detections_localised": n_det, "n_with_depth": n_depth,
               "n_findings_total": n_total, "findings": findings}, open(out / "findings.json", "w"), indent=1)
    print(f"[localize] thr={a.thr} mode={a.mode} dets={n_det} with_depth={n_depth} findings={n_total} (kept {len(findings)})", flush=True)


if __name__ == "__main__":
    main()
