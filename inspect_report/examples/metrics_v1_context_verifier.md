# Baseline metrics (self-labelled set of 40 frames; threshold 0.15)

## Image-level detector quality

### prompts = base

| class | support | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|
| corrosion | 6 | 0 | 0 | 6 | - | 0.00 | - |
| marine_growth | 29 | 12 | 0 | 17 | 1.00 | 0.41 | 0.59 |
| crack | 0 | 0 | 10 | 0 | 0.00 | - | - |
| hole_damage | 1 | 0 | 6 | 1 | 0.00 | 0.00 | 0.00 |
| debris | 3 | 2 | 2 | 1 | 0.50 | 0.67 | 0.57 |
| fish | 0 | 0 | 10 | 0 | 0.00 | - | - |
| rope_cable | 12 | 5 | 0 | 7 | 1.00 | 0.42 | 0.59 |
| metal_structure | 33 | 2 | 0 | 31 | 1.00 | 0.06 | 0.11 |
| window | 7 | 6 | 2 | 1 | 0.75 | 0.86 | 0.80 |
| tire | 10 | 2 | 0 | 8 | 1.00 | 0.20 | 0.33 |
| **micro** | | 29 | 30 | 72 | 0.49 | 0.29 | |

| threshold | micro precision | micro recall |
|---|---|---|
| 0.05 | 0.38 | 0.62 |
| 0.1 | 0.43 | 0.47 |
| 0.15 | 0.49 | 0.29 |
| 0.2 | 0.54 | 0.20 |
| 0.3 | 0.55 | 0.06 |
| 0.4 | 1.00 | 0.02 |

### prompts = ens

| class | support | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|
| corrosion | 6 | 2 | 3 | 4 | 0.40 | 0.33 | 0.36 |
| marine_growth | 29 | 17 | 0 | 12 | 1.00 | 0.59 | 0.74 |
| crack | 0 | 0 | 17 | 0 | 0.00 | - | - |
| hole_damage | 1 | 0 | 12 | 1 | 0.00 | 0.00 | 0.00 |
| debris | 3 | 3 | 4 | 0 | 0.43 | 1.00 | 0.60 |
| fish | 0 | 0 | 16 | 0 | 0.00 | - | - |
| rope_cable | 12 | 8 | 0 | 4 | 1.00 | 0.67 | 0.80 |
| metal_structure | 33 | 4 | 0 | 29 | 1.00 | 0.12 | 0.22 |
| window | 7 | 6 | 2 | 1 | 0.75 | 0.86 | 0.80 |
| tire | 10 | 2 | 0 | 8 | 1.00 | 0.20 | 0.33 |
| **micro** | | 42 | 54 | 59 | 0.44 | 0.42 | |

| threshold | micro precision | micro recall |
|---|---|---|
| 0.05 | 0.35 | 0.74 |
| 0.1 | 0.43 | 0.59 |
| 0.15 | 0.44 | 0.42 |
| 0.2 | 0.50 | 0.30 |
| 0.3 | 0.54 | 0.15 |
| 0.4 | 0.86 | 0.06 |

## Runtime (RTX 5080 unless noted)

| stage | n | mean s | median s | p95 s | total s |
|---|---|---|---|---|---|
| load_vlm | 1 | 6.714 | 6.714 | 6.714 | 6.7 |
| vlm_finding | 30 | 2.829 | 2.794 | 3.163 | 84.9 |
| vlm_summary | 1 | 8.970 | 8.970 | 8.970 | 9.0 |

## Report faithfulness

```
{
 "n_sentences": 25,
 "n_ids_cited": 20,
 "n_hallucinated": 5,
 "summary_only_hallucinated": 1,
 "n_unknown_ids": 0,
 "n_absent_class_mentions": 4,
 "n_cites_unconfirmed": 0,
 "n_id_class_mismatches": 0
}
```

## Crop-level VLM verification

```
{
 "n_labelled_crops": 17,
 "vlm_recall_of_true_crops": 1.0,
 "detector_crop_precision": 0.47058823529411764,
 "vlm_judged": 9,
 "vlm_unclear_or_unparsed": 8,
 "accuracy_confirm_vs_not": 0.9411764705882353,
 "vlm_verification_accuracy": 0.8888888888888888,
 "precision_after_vlm_filter": 0.8888888888888888,
 "n_kept_after_vlm_filter": 9
}
```
