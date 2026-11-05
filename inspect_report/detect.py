"""Stage 2: open-vocabulary detection with OWLv2 (Apache-2.0) on rectified left keyframes.

Per class we keep candidate boxes for two prompt modes computed from the same forward pass:
 base: logit of the first prompt of the class;  ens: max logit over all prompts of the class.
Classes are scored independently (no cross-class competition per box). Boxes are NMSed per
class/mode and kept if score >= --min-score (a later fixed threshold is applied at eval/report).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from ir_common import StageTimer
from prompts import CLASS_PROMPTS, CLASSES


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--min-score", type=float, default=0.05)
    ap.add_argument("--topk", type=int, default=10, help="max boxes per class/mode/frame after NMS")
    ap.add_argument("--nms-iou", type=float, default=0.3)
    a = ap.parse_args(argv)
    out = Path(a.out)
    timer = StageTimer(out, cuda=True)
    kfs = json.load(open(out / "keyframes.json"))["frames"]

    flat, owner = [], []  # prompt list and (class, prompt_idx)
    for c in CLASSES:
        for pi, p in enumerate(CLASS_PROMPTS[c]):
            flat.append(p)
            owner.append((c, pi))
    with timer.time("load_detector"):
        proc = Owlv2Processor.from_pretrained(a.model)
        model = Owlv2ForObjectDetection.from_pretrained(a.model, torch_dtype=torch.float16).cuda().eval()
    text = proc(text=[flat], return_tensors="pt")
    results = {}
    for n, kf in enumerate(kfs):
        img = Image.open(out / kf["left"]).convert("RGB")
        W, H = img.size
        with timer.time("detect", frame=kf["id"]):
            inp = proc(images=img, return_tensors="pt")
            with torch.no_grad():
                o = model(
                    input_ids=text["input_ids"].cuda(), attention_mask=text["attention_mask"].cuda(),
                    pixel_values=inp["pixel_values"].cuda().half(),
                )
            logits = o.logits[0].float()  # (patches, prompts)
            boxes = o.pred_boxes[0].float()  # cxcywh normalised to the padded square
            side = float(max(W, H))
            xyxy = torch.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                                boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], 1) * side
            xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, W)
            xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, H)
            dets = []
            for c in CLASSES:
                idx = [i for i, (cc, _) in enumerate(owner) if cc == c]
                for mode, lg in (("base", logits[:, idx[0]]), ("ens", logits[:, idx].max(1).values)):
                    sc = torch.sigmoid(lg)
                    keep = sc >= a.min_score
                    if not keep.any():
                        continue
                    b, s = xyxy[keep], sc[keep]
                    k = torchvision.ops.nms(b, s, a.nms_iou)[: a.topk]
                    best_prompt = logits[keep][k][:, idx].argmax(1).cpu().numpy()
                    for bb, ss, bp in zip(b[k].cpu().numpy(), s[k].cpu().numpy(), best_prompt):
                        dets.append({"cls": c, "mode": mode, "prompt": CLASS_PROMPTS[c][int(bp)] if mode == "ens" else CLASS_PROMPTS[c][0],
                                     "score": float(ss), "box": [float(v) for v in bb]})
        results[kf["id"]] = dets
        if n % 10 == 0:
            print(f"[detect] {n + 1}/{len(kfs)} {kf['id']} dets={len(dets)}", flush=True)
    json.dump({"model": a.model, "min_score": a.min_score, "frames": results}, open(out / "detections.json", "w"))
    print(f"[detect] done {len(results)} frames", flush=True)


if __name__ == "__main__":
    main()
