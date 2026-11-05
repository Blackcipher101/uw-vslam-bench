# Baseline auto-generated underwater inspection report

Research prototype. Pipeline: keyframes (+VIO pose) -> OWLv2 open-vocabulary detection -> stereo SGBM depth,
3D position and metric size in the VIO world frame, cross-frame merge -> local VLM verifies each finding and
writes assessments + summary -> markdown/HTML report -> baseline evaluation.
All numbers below are from ONE run on the AFRL `bus_outside` bag (sunken bus, murky lake) on an RTX 5080 (ws1).
Poses: uw_ct_vio `s4_rep.tum` (left-camera pose; 198 keyframes all with pose, 64.5 m track, 585 s).
Example report: `examples/report.html` / `examples/report.md` (context verifier run). It is an AI draft and is imperfect (see below).

## Results (bus_outside, self-labelled, indicative only)

Labels: one AI annotator (me), 40 frames evenly sampled from the 198 keyframes (`labels.json`; unsure (frame, class) pairs
excluded) and 30 finding crops (13 unsure, 17 usable). No human cross-check. Positives are tiny for some classes.

### Detector (OWLv2-base, image level, threshold 0.15; `base` = first prompt, `ens` = max over 3 prompts per class)
| class | support (frames) | base P | base R | ens P | ens R | ens FP |
|---|---|---|---|---|---|---|
| corrosion | 6 | - | 0.00 | 0.40 | 0.33 | 3 |
| marine_growth | 29 | 1.00 | 0.41 | 1.00 | 0.59 | 0 |
| crack | 0 | 0.00 | - | 0.00 | - | 17 |
| hole_damage | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 12 |
| debris | 3 | 0.50 | 0.67 | 0.43 | 1.00 | 4 |
| fish | 0 | 0.00 | - | 0.00 | - | 16 |
| rope_cable | 12 | 1.00 | 0.42 | 1.00 | 0.67 | 0 |
| metal_structure | 33 | 1.00 | 0.06 | 1.00 | 0.12 | 0 |
| window | 7 | 0.75 | 0.86 | 0.75 | 0.86 | 2 |
| tire | 10 | 1.00 | 0.20 | 1.00 | 0.20 | 0 |
| **micro** | | 0.49 | 0.29 | 0.44 | 0.42 | |

Threshold sweep (micro, ens): thr 0.05 P 0.35 / R 0.74; 0.10 P 0.43 / R 0.59; 0.15 P 0.44 / R 0.42; 0.20 P 0.50 / R 0.30; 0.30 P 0.54 / R 0.15; 0.40 P 0.86 / R 0.06.
Full tables: `run_bus/metrics.md`. Reading: precision is high for the common classes (marine_growth, rope_cable,
metal_structure, tire, window) but recall is low (metal_structure R 0.12 although the bus is in 33/40 frames).
Defect prompts are poor: crack (17 FP frames) and fish (16 FP) have no true positives in this set (the scene has no crack/fish that I could see), hole_damage 12 FP,
corrosion P 0.40 / R 0.33 (ens only; base found none). Prompt ensembling raised micro recall 0.29 -> 0.42 at a small precision cost (0.49 -> 0.44), measured on detector output only.
The report pipeline itself ran with `base` prompts (findings come from `base`, thr 0.15); I did not run the full pipeline with `ens`.
Findings are ranked by raw detector score, so the top 30 are dominated by crack/fish false positives plus seabed "marine growth".

### VLM verifier (Qwen3-VL-4B-Instruct) on the 30 finding crops (17 with usable labels; detector crop precision = 8/17 = 0.47)
| verifier | confirmed (of 30) | recall of true crops | precision after filter | accuracy (confirm vs not) | faithfulness flags |
|---|---|---|---|---|---|
| v0: crop only, true/false (default) | 0 | 0/8 = 0.00 | undefined (kept none) | 9/17 = 0.53 | 0 |
| v1: `--verify-context` (boxed keyframe + crop, yes/no/unclear) | 20 (10 unclear) | 8/8 = 1.00 | 8/9 = 0.89 | 16/17 = 0.94 | 5 |

v0 rejected every crop, including obvious tires, windows and marine growth, so the cure was worse than the disease; v1 confirmed all 8 true crops and
left all but one false crop (F27 debris) unconfirmed/unclear. Caveats: 17 labelled crops only; the 12 confirmed crops I could not label
(fish blobs, seabed algae, a porthole ring called hole_damage with severity medium, a debris blob in glare) are unscored, so the true precision of v1 is unknown; the
accuracy for v1 treats "unclear" as "not confirmed". Selection of v1 over v0 is based on this same small set (it was the single pre-declared change, not tuned further).

### Faithfulness (lexical; `faithfulness.check_report`)
Counts: unknown finding ids; citing a finding the verifier did not confirm; a class mentioned (not negated) with no confirmed finding of that class;
class mentioned that differs from the cited finding's class; a stated "N confirmed findings" that is wrong. Assessments of unconfirmed findings are not checked (they are the rejection rationale).
- Original baseline run (before the consistency fix): the summary listed 14 findings the verifier had rejected as "key findings" and mentioned 5 classes with no confirmed finding: 19 flags (the old counter said 0 because it did not know about verification status).
- v0 (fix applied): all unconfirmed, so the summary is a fixed text from code (no VLM call): 0 flags.
- v1: 5 flags: the summary says "10 confirmed findings" but lists/has 20 (1), and 4 assessments of confirmed findings mention a class with no confirmed finding ("corroded" on a tire, window frame and debris crop; "metal structure" on the hole crop).
- Bug found while re-scoring: a first version of the counter treated the table prefix "F01:" of each assessment line as a citation and checked assessments of unconfirmed findings, giving a bogus 45 for v0 (30 cites + 15 class mentions) and 19 for v1. The counter was fixed, and both saved runs were re-scored with `rescore.py` (no model): v0 45 -> 0, v1 19 -> 5. The figures above are the corrected ones.
Not checked: numbers, locations and sizes quoted in free text; omissions; whether severity is justified.

### Runtime (RTX 5080 16 GB, per stage; `run_bus/metrics.md`)
| stage | n | median s | p95 s |
|---|---|---|---|
| decode one stereo pair (CPU) | 198 | 0.007 | 0.008 |
| rectify (CPU) | 198 | 0.002 | 0.005 |
| OWLv2 detection, all 30 prompts, per frame (GPU, fp16) | 198 | 0.180 | 0.191 |
| SGBM depth at 0.5 scale (CPU, ws1) | 177 | 0.021 | 0.028 |
| locate + pose per frame | 177 | <0.001 | 0.001 |
| VLM finding check, v0 crop only | 30 | 0.94 | - |
| VLM finding check, v1 context (2 images) | 30 | 2.79 | 3.16 |
| VLM summary (v1) | 1 | 8.97 | - |
Model load: detector 7-64 s (64 s was first download), VLM 7-9 s. End to end for 198 keyframes: about 36 s detection + 4 s stereo + 85 s VLM (v1).

Unit tests (no models): `python3 -m pytest tests` (19 pass on the laptop).
`rescore.py` re-scores/re-renders saved outputs without a model: `python3 rescore.py --run run_bus_v1 --inputs run_bus --traj <tum>`.

## Run (workstation1, one command)

```
cd ~/uwvio/inspect_report
./run_all.sh ~/uwvio/data/bus_outside/bus_outside.db3 ~/uwvio/data/bus_outside/camchain_bus_outside.yaml \
             ~/uwvio/data/bus_outside/s4_rep.tum ~/uwvio/out/bus_report
```
Creates `.venv` (torch 2.8.0+cu128, transformers 4.57.1, opencv 4.10, rosbags; see `requirements.txt`),
downloads models into the HF cache, writes per-stage logs to `<out>/logs/`, per-event timings to
`<out>/timings.jsonl` (flushed as they occur). Env: `STAGES="keyframes detect localize report eval"`,
`THR=0.15`, `MODE=base|ens`, `VLM=...`, `LABELS=<labels.json>`, `SOURCE_NAME=...`.

Frames for hand labelling only: `STAGES="keyframes" ./run_all.sh <same four args>` then copy
`<out>/label_frames/` (about 40 JPEGs, 800 px wide) back to `inspect_report/label_frames/` on the laptop.
Poses used for the reported run: uw_ct_vio `results/bus_outside/uwct/s4_rep.tum` (~0.34 m Sim3 ATE; standalone cuVSLAM `cuvslam_stereo_slam.tum` had 5.2 m ATE on this bag, SVIn odometry drifts to hundreds of metres). Pass `none` to skip 3D localisation (box-only findings).

Eval: `LABELS=labels.json STAGES=eval ./run_all.sh ... <same out dir>`. `labels.json` format is in `eval.py`.

## Stages and files
| stage | file | notes |
|---|---|---|
| 1 | `keyframes.py` | time (2-4 s) + pose-distance (0.3 m) selection; rectified left/right; label frames |
| 2 | `detect.py` | OWLv2; both prompt modes (`base` first prompt, `ens` max over 3 prompts) from one forward pass |
| 3 | `localize.py`, `geometry.py` | SGBM at 0.5 scale, median depth in inner 50% of box, 1 m / same-class merge |
| 4 | `report.py`, `render.py` | VLM per-crop verification/assessment/severity + summary; tables rendered by code; `--verify-context`, `--inputs` |
| 5 | `eval.py`, `faithfulness.py`, `rescore.py` | runtime, image-level P/R, crop verification, hallucination counts |


## Models and licences
- Detector: `google/owlv2-base-patch16-ensemble` (Apache-2.0).
- VLM: `Qwen/Qwen3-VL-4B-Instruct` (Apache-2.0; bf16 about 9 GB fits 16 GB). 7B/8B bf16 does not fit; `--load-4bit` is wired for
  Qwen2.5-VL-7B-Instruct (Apache-2.0) but untested. Not used: Qwen2.5-VL-3B (research licence), Grounding DINO weights (Apache-2.0, optional alternative, not implemented).

## Known limits (by design, to be stated in any pitch)
- Calibration is the AFRL underwater Kalibr chain; refraction is absorbed by the pinhole/radtan model only approximately.
- Box size = box extent at median depth (fronto-parallel assumption). The VIO world frame is the start-camera frame, not gravity aligned.
- SGBM in murky water has many invalid pixels; findings without valid depth are box-only (no size/position).
- Faithfulness check is lexical (class terms + finding ids); quoted numbers/locations in free text are not verified.
- Severity is a VLM suggestion, not a calibrated assessment.
- One AI annotator, 40 frames and 17 usable crops: all precision/recall values have wide uncertainty; several classes have 0-3 positives.
- Single scene (murky lake, glare from a lamp in about a third of the frames); nothing here shows performance in clear water or on other structures. NTNU data was not run.
- Findings are ranked by detector score (no verifier-aware ranking); the example report includes unconfirmed-by-human items such as F27 (debris, a false positive) and fish/hole_damage crops I could not judge.
- Stereo uses the AFRL calibration; SGBM ran on ws1 CPU. 528 of 641 localised detections got a valid depth; sizes of large boxes (bus panels, windows) are crude.
- Severity labels are VLM suggestions; v1 gave low/medium to 18 of the 20 confirmed findings, including marine-growth patches, with no calibration.
- Qwen3-VL-4B miscounted its own summary ("10" vs 20); the draft must not be used without human review.
- Files: `run_bus*/` hold the raw outputs (detections, findings, crops, VLM outputs, metrics, logs); `examples/` holds the example report with thumbnails (0.5 MB).
