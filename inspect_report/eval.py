"""Stage 5: baseline evaluation.

(a) runtime per stage (from timings.jsonl),
(b) image-level per-class precision/recall of the detector vs hand labels (labels.json),
(c) crop-level VLM verification accuracy vs hand crop labels, and faithfulness counts.

labels.json: {"frames": {"kf_0012": ["fish", "corrosion"], ...},   # classes present in each labelled frame
              "unsure": {"kf_0012": ["tire"], ...},            # (frame, class) pairs excluded from metrics
              "crops":  {"F01": true, ...}}                        # optional: does crop F01 really show its class?
SMALL SELF-LABELLED SET: the numbers are indicative only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ir_common import load_timings
from prompts import CLASSES

SWEEP = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]


def image_level(dets: dict, labels: dict, thr: float, mode: str, classes=CLASSES, unsure: dict | None = None) -> dict:
    unsure = unsure or {}
    res = {}
    for c in classes:
        tp = fp = fn = tn = 0
        for fid, present in labels.items():
            if c in unsure.get(fid, ()):
                continue  # (frame, class) the annotator could not judge: excluded
            pred = any(d["cls"] == c and d["mode"] == mode and d["score"] >= thr for d in dets.get(fid, []))
            gt = c in present
            tp += pred and gt
            fp += pred and not gt
            fn += (not pred) and gt
            tn += (not pred) and (not gt)
        p = tp / (tp + fp) if tp + fp else None
        r = tp / (tp + fn) if tp + fn else None
        f1 = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
        res[c] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "support": tp + fn, "precision": p, "recall": r, "f1": f1}
    return res


def micro(res: dict) -> dict:
    tp = sum(v["tp"] for v in res.values())
    fp = sum(v["fp"] for v in res.values())
    fn = sum(v["fn"] for v in res.values())
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r}


def crop_eval(vlm: dict, crop_labels: dict) -> dict:
    ids = [i for i in crop_labels if i in vlm and crop_labels[i] in (True, False)]  # "unsure" labels are skipped
    detector_correct = sum(bool(crop_labels[i]) for i in ids)
    judged = [i for i in ids if vlm[i].get("shows_class") is not None]
    agree = sum(vlm[i]["shows_class"] == bool(crop_labels[i]) for i in judged)
    kept = [i for i in judged if vlm[i]["shows_class"]]
    return {
        "n_labelled_crops": len(ids),
        "vlm_recall_of_true_crops": (sum(vlm[i]["shows_class"] is True for i in ids if crop_labels[i]) / detector_correct) if detector_correct else None,
        "detector_crop_precision": detector_correct / len(ids) if ids else None,
        "vlm_judged": len(judged),
        "vlm_unclear_or_unparsed": len(ids) - len(judged),
        "accuracy_confirm_vs_not": sum((vlm[i].get("shows_class") is True) == bool(crop_labels[i]) for i in ids) / len(ids) if ids else None,
        "vlm_verification_accuracy": agree / len(judged) if judged else None,
        "precision_after_vlm_filter": (sum(bool(crop_labels[i]) for i in kept) / len(kept)) if kept else None,
        "n_kept_after_vlm_filter": len(kept),
    }


def _f(x):
    return "-" if x is None else f"{x:.2f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--thr", type=float, default=0.15)
    ap.add_argument("--detections", default=None, help="detections.json (default: <out>/detections.json)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    labels_all = json.load(open(a.labels))
    labels = labels_all["frames"]
    unsure = labels_all.get("unsure", {})
    dets = json.load(open(a.detections or out / "detections.json"))["frames"]
    missing = [k for k in labels if k not in dets]
    if missing:
        raise SystemExit(f"labelled frames without detections (different run?): {missing[:5]}")
    M: dict = {"n_labelled_frames": len(labels), "thr": a.thr, "runtime": load_timings(out), "image_level": {}, "sweep": {}}
    for mode in ("base", "ens"):
        M["image_level"][mode] = image_level(dets, labels, a.thr, mode, unsure=unsure)
        M["image_level"][mode]["_micro"] = micro({k: v for k, v in M["image_level"][mode].items()})
        M["sweep"][mode] = {str(t): micro(image_level(dets, labels, t, mode, unsure=unsure)) for t in SWEEP}
    vo = out / "vlm_outputs.json"
    if vo.exists():
        v = json.load(open(vo))
        M["faithfulness"] = {k: v["faithfulness"][k] for k in ("n_sentences", "n_ids_cited", "n_hallucinated", "summary_only_hallucinated")}
        M["faithfulness"]["n_unknown_ids"] = len(v["faithfulness"]["unknown_ids"])
        M["faithfulness"]["n_absent_class_mentions"] = len(v["faithfulness"]["absent_class_mentions"])
        M["faithfulness"]["n_cites_unconfirmed"] = len(v["faithfulness"].get("cites_rejected", []))
        M["faithfulness"]["n_id_class_mismatches"] = len(v["faithfulness"]["id_class_mismatches"])
        M["vlm_parse_ok"] = f"{sum(r['parse_ok'] for r in v['per_finding'].values())}/{len(v['per_finding'])}"
        if labels_all.get("crops"):
            M["crop_eval"] = crop_eval(v["per_finding"], labels_all["crops"])
    json.dump(M, open(out / "metrics.json", "w"), indent=1)

    L = [f"# Baseline metrics (self-labelled set of {len(labels)} frames; threshold {a.thr})", "",
         "## Image-level detector quality", ""]
    for mode in ("base", "ens"):
        L += [f"### prompts = {mode}", "", "| class | support | TP | FP | FN | precision | recall | F1 |", "|---|---|---|---|---|---|---|---|"]
        for c in CLASSES:
            r = M["image_level"][mode][c]
            L.append(f"| {c} | {r['support']} | {r['tp']} | {r['fp']} | {r['fn']} | {_f(r['precision'])} | {_f(r['recall'])} | {_f(r['f1'])} |")
        m = M["image_level"][mode]["_micro"]
        L += [f"| **micro** | | {m['tp']} | {m['fp']} | {m['fn']} | {_f(m['precision'])} | {_f(m['recall'])} | |", ""]
        L += ["| threshold | micro precision | micro recall |", "|---|---|---|"]
        L += [f"| {t} | {_f(s['precision'])} | {_f(s['recall'])} |" for t, s in M["sweep"][mode].items()] + [""]
    L += ["## Runtime (RTX 5080 unless noted)", "", "| stage | n | mean s | median s | p95 s | total s |", "|---|---|---|---|---|---|"]
    L += [f"| {k} | {v['n']} | {v['mean_s']:.3f} | {v['median_s']:.3f} | {v['p95_s']:.3f} | {v['total_s']:.1f} |" for k, v in M["runtime"].items()]
    if "faithfulness" in M:
        L += ["", "## Report faithfulness", "", "```", json.dumps(M["faithfulness"], indent=1), "```"]
    if "crop_eval" in M:
        L += ["", "## Crop-level VLM verification", "", "```", json.dumps(M["crop_eval"], indent=1), "```"]
    (out / "metrics.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
