"""Markdown / HTML rendering of the draft report (pure functions, no model needed)."""
from __future__ import annotations

import base64
import html
from pathlib import Path

import cv2

from prompts import DEFECT_CLASSES

BANNER = ("AI-GENERATED DRAFT - detections come from an open-vocabulary detector and text from a "
          "vision-language model. Severity values are suggestions only and MUST be reviewed by a qualified human inspector.")


def thumb_b64(path: Path, max_side: int = 160, box: list[float] | None = None) -> str:
    img = cv2.imread(str(path))
    if img is None:
        return ""
    if box is not None:
        cv2.rectangle(img, tuple(int(v) for v in box[:2]), tuple(int(v) for v in box[2:]), (0, 255, 255), max(2, img.shape[1] // 200))
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def fmt_loc(f: dict) -> str:
    if f["xyz_world"] is None:
        return "n/a (no pose or depth)"
    x, y, z = f["xyz_world"]
    return f"({x:.1f}, {y:.1f}, {z:.1f})"


def fmt_size(f: dict) -> str:
    if f["width_m"] is None:
        return "n/a"
    return f"{f['width_m'] * 100:.0f} x {f['height_m'] * 100:.0f} cm"


def severity_of(f: dict, vlm: dict) -> str:
    if vlm.get("shows_class") is not True:
        return "n/a (not confirmed)"
    return vlm.get("severity", "n/a") if f["cls"] in DEFECT_CLASSES else "info"


def status_of(vlm: dict) -> str:
    return vlm.get("status", "confirmed")


def plan_svg(track_xz: list[tuple[float, float]], findings: list[dict], size: int = 360) -> str:
    pts = list(track_xz) + [(f["xyz_world"][0], f["xyz_world"][2]) for f in findings if f["xyz_world"]]
    if len(pts) < 2:
        return ""
    xs, zs = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1, z0, z1 = min(xs), max(xs), min(zs), max(zs)
    span = max(x1 - x0, z1 - z0, 1e-6)
    m = 20

    def tf(x, z):
        return m + (x - x0) / span * (size - 2 * m), size - m - (z - z0) / span * (size - 2 * m)

    line = " ".join(f"{tf(*p)[0]:.1f},{tf(*p)[1]:.1f}" for p in track_xz)
    dots = "".join(
        f'<circle cx="{tf(f["xyz_world"][0], f["xyz_world"][2])[0]:.1f}" cy="{tf(f["xyz_world"][0], f["xyz_world"][2])[1]:.1f}" r="4" fill="red"/>'
        f'<text x="{tf(f["xyz_world"][0], f["xyz_world"][2])[0] + 5:.1f}" y="{tf(f["xyz_world"][0], f["xyz_world"][2])[1] - 5:.1f}" font-size="9">{f["id"]}</text>'
        for f in findings if f["xyz_world"]
    )
    return (f'<svg width="{size}" height="{size}" style="border:1px solid #999;background:#fafafa">'
            f'<polyline points="{line}" fill="none" stroke="#06c" stroke-width="1"/>{dots}</svg>')


def _rows(findings, vlm):
    """Confirmed findings first (stable otherwise)."""
    for f in sorted(findings, key=lambda f: vlm.get(f["id"], {}).get("shows_class") is not True):
        v = vlm.get(f["id"], {})
        yield f, v


def render_md(meta: dict, cov: dict, findings: list[dict], vlm: dict, summary: str, faith: dict) -> str:
    L = [f"# Underwater inspection report (draft)", "", f"> **{BANNER}**", "", "## Summary", "", summary.strip(), "",
         "## Coverage", ""]
    L += [f"- Source: `{meta['source']}`", f"- Duration: {cov['duration_s']:.0f} s, keyframes: {cov['n_keyframes']} "
          f"({cov['n_keyframes_with_pose']} with pose)",
          f"- Track length: {'n/a' if cov['track_length_m'] is None else format(cov['track_length_m'], '.1f') + ' m'} "
          f"(VIO trajectory: `{cov['traj_file']}`)",
          f"- Detector: {meta['detector']}; VLM: {meta['vlm']}; score threshold {meta['thr']} ({meta['mode']} prompts)", "",
          "## Findings", "",
          "Positions are in the VIO world frame (metres, start-camera frame, not gravity aligned). Sizes are approximate "
          "box extents from stereo depth. Frame refs are keyframe ids.", "",
          "| ID | Type | Status | Location (x,y,z m) | Size | Conf. | Obs. | Frames | Severity (AI suggestion) | Assessment |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for f, v in _rows(findings, vlm):
        L.append(f"| {f['id']} | {f['cls']} | {status_of(v)} | {fmt_loc(f)} | {fmt_size(f)} | {f['score_max']:.2f} | {f['n_obs']} | "
                 f"{', '.join(f['frames'][:3])}{'...' if len(f['frames']) > 3 else ''} | {severity_of(f, v)} | "
                 f"{v.get('assessment', '').replace('|', '/').replace(chr(10), ' ')} |")
    L += ["", "Crops: `crops/<ID>.jpg` in the output directory (embedded in the HTML version).", "",
          "## Faithfulness check", "",
          f"- Report-text sentences checked: {faith['n_sentences']}; hallucinated statements: **{faith['n_hallucinated']}** "
          f"(unknown ids {len(faith['unknown_ids'])}, absent-class mentions {len(faith['absent_class_mentions'])}, "
          f"id/class mismatches {len(faith['id_class_mismatches'])}, cites of unconfirmed findings {len(faith.get('cites_rejected', []))})",
          f"- Findings not mentioned in the text: {', '.join(faith['findings_not_mentioned']) or 'none'}", ""]
    return "\n".join(L)


def render_html(meta, cov, findings, vlm, summary, faith, out_dir: Path, track_xz) -> str:
    e = html.escape
    rows = []
    for f, v in _rows(findings, vlm):
        c = thumb_b64(out_dir / f["crop"], 140)
        rows.append(
            f"<tr><td>{f['id']}</td><td><img src='data:image/jpeg;base64,{c}'></td><td>{e(f['cls'])}</td><td>{e(status_of(v))}</td>"
            f"<td>{e(fmt_loc(f))}</td><td>{e(fmt_size(f))}</td><td>{f['score_max']:.2f}</td><td>{f['n_obs']}</td>"
            f"<td>{e(', '.join(f['frames'][:3]))}{'...' if len(f['frames']) > 3 else ''}</td>"
            f"<td><b>{e(severity_of(f, v))}</b></td><td>{e(v.get('assessment', ''))}</td></tr>")
    trk = "n/a" if cov["track_length_m"] is None else f"{cov['track_length_m']:.1f} m"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Underwater inspection report (draft)</title>
<style>body{{font-family:sans-serif;max-width:1200px;margin:1em auto}} table{{border-collapse:collapse}}
td,th{{border:1px solid #bbb;padding:4px;font-size:13px;vertical-align:top}} .banner{{background:#fde;border:2px solid #c00;padding:8px;font-weight:bold}}</style></head><body>
<h1>Underwater inspection report (draft)</h1><div class="banner">{e(BANNER)}</div>
<h2>Summary</h2><p>{e(summary.strip())}</p>
<h2>Coverage</h2><ul><li>Source: <code>{e(meta['source'])}</code></li>
<li>Duration {cov['duration_s']:.0f} s; keyframes {cov['n_keyframes']} ({cov['n_keyframes_with_pose']} with pose); track length {trk} (VIO: <code>{e(cov['traj_file'])}</code>)</li>
<li>Detector {e(meta['detector'])}; VLM {e(meta['vlm'])}; threshold {meta['thr']} ({meta['mode']} prompts)</li></ul>
{plan_svg(track_xz, findings)}<p><small>Plan view: VIO world x (horizontal) vs z (vertical); blue = camera track, red = findings.</small></p>
<h2>Findings</h2><p>Positions: VIO world frame (m). Sizes: approximate stereo box extent.</p>
<table><tr><th>ID</th><th>Crop</th><th>Type</th><th>Status</th><th>Location x,y,z (m)</th><th>Size</th><th>Conf.</th><th>Obs.</th><th>Frames</th><th>Severity (AI suggestion)</th><th>Assessment</th></tr>
{''.join(rows)}</table>
<h2>Faithfulness check</h2><p>{faith['n_sentences']} sentences checked; hallucinated statements: <b>{faith['n_hallucinated']}</b>.
Findings not mentioned in text: {e(', '.join(faith['findings_not_mentioned']) or 'none')}.</p></body></html>"""
