"""Stage 4: VLM-written draft report from findings.json (+ crops), rendered to markdown and HTML.

The findings table, locations, sizes and confidences are rendered by code from the detection data.
The VLM only writes (a) per-finding verification + assessment + severity suggestion from the crop and
(b) an executive summary from the structured JSON. Everything it writes is then faithfulness-checked.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

import faithfulness
from ir_common import StageTimer, load_tum
from prompts import CLASS_PROMPTS, DEFECT_CLASSES
from render import render_html, render_md

FINDING_PROMPT = """You are assisting an underwater inspection. An automatic detector flagged this image crop as "{cls}" (detector confidence {score:.2f}).
Look only at the image. Reply with ONE JSON object and nothing else:
{{"shows_class": true or false (does the crop clearly show {cls}?),
 "description": "one short sentence describing what is visible",
 "assessment": "one sentence on the apparent condition, using only what is visible",
 "severity": "none" | "low" | "medium" | "high" (suggested severity of the issue; "none" if it is not a defect or not visible)}}"""

CONTEXT_PROMPT = """You are assisting an underwater inspection. Image 1 is a full camera view with a yellow box. Image 2 is a close-up of the boxed region.
An automatic detector claims the boxed region shows: {cls} (examples: {examples}). Detector confidence {score:.2f}.
Water is murky, so judge carefully; if you cannot tell, answer "unclear". Reply with ONE JSON object and nothing else:
{{"answer": "yes" | "no" | "unclear" (is the boxed region really {cls}?),
 "description": "one short sentence describing what is in the boxed region",
 "assessment": "one sentence on the apparent condition, using only what is visible",
 "severity": "none" | "low" | "medium" | "high" (suggested severity of the issue; "none" if it is not a defect or not visible)}}"""

SUMMARY_PROMPT = """Write a short (3-5 sentences) executive summary of an underwater inspection for a human reviewer.
Use ONLY the structured findings below (these are the findings confirmed by a visual check). {n_unconf} further detector candidates could not be confirmed and are listed separately: do not mention them. Mention a finding only by citing its id in square brackets, e.g. [F01].
Do not mention any defect type or object type that has no finding. Do not invent locations or measurements.
Say that the result is an automatic draft that requires human review.
Inspection coverage: {cov}
Findings JSON:
{findings}"""


def generate(model, proc, content: list[dict], max_new_tokens: int, timer: StageTimer, stage: str, **extra) -> str:
    msgs = [{"role": "user", "content": content}]
    inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
    with timer.time(stage, **extra):
        with torch.no_grad():
            o = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def parse_json_obj(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vlm", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--load-4bit", action="store_true", help="bitsandbytes 4-bit (needs `pip install bitsandbytes`; untested)")
    ap.add_argument("--source-name", default="")
    ap.add_argument("--inputs", default=None, help="dir with findings/coverage/detections/keyframes/crops (default: --out)")
    ap.add_argument("--verify-context", action="store_true",
                    help="verifier sees the boxed keyframe plus the crop and answers yes/no/unclear (default: crop only, true/false)")
    ap.add_argument("--context-width", type=int, default=640)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(a.inputs) if a.inputs else out
    timer = StageTimer(out, cuda=True)
    fj = json.load(open(src / "findings.json"))
    findings = fj["findings"]
    cov = json.load(open(src / "coverage.json"))
    kf_path = {k["id"]: k["left"] for k in json.load(open(src / "keyframes.json"))["frames"]}

    kw = {"torch_dtype": torch.bfloat16, "device_map": "cuda"}
    if a.load_4bit:
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    with timer.time("load_vlm"):
        proc = AutoProcessor.from_pretrained(a.vlm)
        model = AutoModelForImageTextToText.from_pretrained(a.vlm, **kw).eval()

    vlm: dict[str, dict] = {}
    for f in findings:
        img = Image.open(src / f["crop"]).convert("RGB")
        if a.verify_context:
            kf = cv2.imread(str(src / kf_path[f["best_frame"]]))
            sc = a.context_width / kf.shape[1]
            kf = cv2.resize(kf, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            x0, y0, x1, y1 = (int(v * sc) for v in f["best_box"])
            cv2.rectangle(kf, (x0, y0), (x1, y1), (0, 255, 255), 3)
            ctx = Image.fromarray(cv2.cvtColor(kf, cv2.COLOR_BGR2RGB))
            prompt = CONTEXT_PROMPT.format(cls=f["cls"], examples="; ".join(CLASS_PROMPTS[f["cls"]]), score=f["score_max"])
            content = [{"type": "image", "image": ctx}, {"type": "image", "image": img}, {"type": "text", "text": prompt}]
        else:
            content = [{"type": "image", "image": img},
                       {"type": "text", "text": FINDING_PROMPT.format(cls=f["cls"], score=f["score_max"])}]
        txt = generate(model, proc, content, 200, timer, "vlm_finding", id=f["id"])
        obj = parse_json_obj(txt)
        rec = {"raw": txt, "parse_ok": obj is not None}
        if obj is not None:
            if a.verify_context:
                ans = str(obj.get("answer", "")).strip().lower()
                rec["shows_class"] = {"yes": True, "no": False}.get(ans)
            else:
                rec["shows_class"] = obj.get("shows_class") if isinstance(obj.get("shows_class"), bool) else None
            rec["description"] = str(obj.get("description", ""))
            rec["assessment"] = str(obj.get("assessment", ""))
            sev = str(obj.get("severity", "")).lower()
            rec["severity"] = sev if sev in ("none", "low", "medium", "high") else "unrated"
        else:
            rec.update(shows_class=None, description="", assessment=txt[:200], severity="unrated")
        rec["status"] = {True: "confirmed", False: "unconfirmed (rejected by VLM check)", None: "unverified (VLM unclear/unparsed)"}[rec["shows_class"]]
        vlm[f["id"]] = rec
        print(f"[report] {f['id']} {f['cls']} shows_class={rec['shows_class']} sev={rec['severity']}", flush=True)

    confirmed = [f for f in findings if vlm[f["id"]]["shows_class"] is True]
    rejected_ids = {f["id"] for f in findings} - {f["id"] for f in confirmed}
    brief = [{"id": f["id"], "class": f["cls"], "confidence": round(f["score_max"], 2), "observations": f["n_obs"],
              "position_world_m": None if f["xyz_world"] is None else [round(v, 1) for v in f["xyz_world"]],
              "size_cm": None if f["width_m"] is None else [round(f["width_m"] * 100), round(f["height_m"] * 100)],
              "time_s": round(f["t_first"] - cov["t_start"], 1),
              "vlm_assessment": vlm[f["id"]].get("assessment", "")} for f in confirmed]
    cov_s = f"{cov['duration_s']:.0f} s, {cov['n_keyframes']} keyframes, track length " + (
        "unknown" if cov["track_length_m"] is None else f"{cov['track_length_m']:.0f} m")
    if confirmed:
        summary = generate(model, proc, [{"type": "text", "text": SUMMARY_PROMPT.format(
            cov=cov_s, findings=json.dumps(brief), n_unconf=len(rejected_ids))}], 300, timer, "vlm_summary")
    else:
        summary = (f"Automatic draft covering {cov_s}. The detector proposed {len(findings)} candidate findings, but none was "
                   f"confirmed by the visual verification step, so no finding is reported as confirmed. The candidates are "
                   f"listed below as unconfirmed for human review.")
    faith = faithfulness.check_report(summary, findings, vlm)

    tum = None if cov["traj_file"] == "none" else load_tum(cov["traj_file"])
    track_xz = []
    if tum is not None:
        m = (tum[:, 0] >= cov["t_start"]) & (tum[:, 0] <= cov["t_end"])
        track_xz = [(float(r[1]), float(r[3])) for r in tum[m][:: max(1, int(m.sum() // 400))]]
    meta = {"source": a.source_name or str(Path(json.load(open(src / "keyframes.json"))["bag"]).name),
            "detector": json.load(open(src / "detections.json"))["model"], "vlm": a.vlm + (" (context verifier)" if a.verify_context else ""), "thr": fj["thr"], "mode": fj["mode"]}
    json.dump({"summary": summary, "per_finding": vlm, "faithfulness": faith, "meta": meta}, open(out / "vlm_outputs.json", "w"), indent=1)
    (out / "report.md").write_text(render_md(meta, cov, findings, vlm, summary, faith))
    (out / "report.html").write_text(render_html(meta, cov, findings, vlm, summary, faith, src, track_xz))
    print(f"[report] wrote report.md/html; hallucinated statements = {faith['n_hallucinated']}", flush=True)


if __name__ == "__main__":
    main()
