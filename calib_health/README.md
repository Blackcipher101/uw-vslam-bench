# In-water stereo calibration health check (research prototype)

Estimates, from a short stretch (10-60 s) of a normal dive bag, the rotation error of a stereo
extrinsic per axis (deg, left-camera x/y/z), the implied disparity bias and depth error, and a verdict
OK / CHECK / RECALIBRATE. No motion command, no target. Status: **prototype with honest baseline numbers;
absolute accuracy is NOT established** (see "Limits" - read it before trusting any number below).

## Problem
On NTNU 2023 using the in-air stereo extrinsic with in-water intrinsics = 0.28 deg about the camera y axis
(~2.2 px disparity bias at f~457 px) gave 15-50 % depth error at 3-8 m and about doubled VIO error.
Rotation about y biases disparity (depth scale vs range), x and z produce vertical disparity.
Measured here: the real air-vs-water difference is w = (-0.051, -0.275, +0.005) deg (x, y, z; `calib.air_vs_water_error_deg`).

## Conventions
`x_l = R_lr x_r + t_lr`; NTNU left = cam1, right = cam0, f = 457 px, B = 0.1097 m. Candidate `R_cand = Exp(w) R_true`
(w in left-camera axes, small-angle ~ same as right axes). The baseline direction is trusted (only rotation is estimated).
Matches are raw pixels, so one extraction serves every candidate calibration.

## Method (`estimators.py`)
Matching/tracking (`tracks.py`, calibration independent): CLAHE, Shi-Tomasi corners in the left image, pyramidal
KLT left->right with forward-backward check (0.3 px) and loose gates (|dy_raw|<12 px), plus KLT left k -> k+gap (6 frames,
0.3 s) for motion. Cv2 fisheye (equidistant) or radtan model for normalisation.

Rays are rotated into a baseline-aligned frame (x' = X/Z, y' = Y/Z, left ray). To first order in w:

    dy (vertical disparity, px) = f [ w_x (1+y'^2) - w_z x' - w_y x' y' ]
    dd (disparity bias,  px)    = f [ w_z y' + w_x x' y' - w_y (1+x'^2) ]

1. **Static epipolar fit** (`epipolar_fit`): Tukey-IRLS of dy on the three basis functions, iterated 3x with re-rectification.
   w_x and w_z are strongly observable. w_y appears only via the second-order x'y' term.
2. **Motion/scale-consistency** (`motion_intercept`, method (c) of the brief): the left-only essential matrix between frames
   k and k+gap gives inverse depth x_inv up to a per-pair scale. Stereo disparity must obey d = alpha_p x_inv + c(1+x'^2)
   with alpha_p = f B / s_p free per pair; the shared intercept c = -f w_y is a disparity bias that no scale can absorb
   (z_est = f B/(d+b): range-dependent motion scale). Solved by variable projection, effective-variance IRLS (errors in x_inv are
   propagated; plain OLS is attenuated).
3. **Fusion**: w_y = inverse-variance (block-bootstrap) average of 1 and 2; w_x, w_z refit with that w_y removed
   (x/z <-> y coupling otherwise leaks: the free 3-parameter fit is poorly conditioned for z). 90 % CI = block bootstrap
   (4-keyframe blocks) over keyframes.
4. **Verdict** (`verdict`, thresholds fixed a priori, tied to implied depth error `err(Z) = |w_y| Z / B`):
   RECALIBRATE if err(3 m) >= 10 % or RMS vertical disparity >= 1.5 px; CHECK if err(8 m) >= 10 % or RMS vertical disparity >= 0.7 px.
   For NTNU: w_y = 0.079 deg -> CHECK, 0.21 deg -> RECALIBRATE; vertical disparity 0.7 / 1.5 px = 0.09 / 0.19 deg about x.

**Pressure depth (method (a)) was not implemented**: over the 60 s stretches the vehicle's vertical excursion is only 0.1-0.4 m
(mocap) while the motion is mostly horizontal, so a pressure-referenced scale constrains almost nothing. IMU method (b) not tried.

## Observability argument and what the data say
The brief assumes w_y is invisible to a single stereo pair. In theory it is only *weakly* visible: through the x'y' term of the
epipolar field (and its twin, translation-direction t_z, which we do not estimate). With NTNU's wide fisheye (|x'|,|y'| up to 0.6) the
term is large enough to be measured: static epipolar w_y has bootstrap SE 0.017 deg on 60 s, the motion method 0.058 deg (0.04-0.14 depending on
extraction settings). The motion method is the physically independent check but is ~3x noisier here and the fused value is dominated by the static fit.
Caveat: the static w_y term is also what intrinsic/distortion-model error would look like, so it can absorb such error (not separable here).

## Results (all numbers produced by the scripts in this dir; logs in `results/`)

### Experiment A: calibration-side injection (`baseline.py`, `summarize.py`; 3 eval stretches 105-165, 170-230, 235-295 s of traj_06, 60 s each; dev stretch 40-100 s used for tuning)
Estimated shift (est - estimate at zero injection on the same stretch) equals the injected value to <= 0.001 deg on every axis and level
(x,z,y rows in `results/results_summary.txt`). **This is equivariance by construction** (identical matches, only R_cand changes) and
validates sign conventions/linearity only; it is NOT a measurement-accuracy figure. Off-axis leakage <= 0.001 deg.

What injection of the real in-air extrinsic gives: fused estimate -0.171 deg (mean of 3 stretches) = +0.103 (residual of the GOOD calibration) - 0.275 (true air-water difference).

Estimator on the *unperturbed* in-water calibration (injected = 0), fused, deg:

| stretch | w_x | w_y | w_z |
|---|---|---|---|
| 40-100 (dev) | +0.002 | +0.170 | -0.014 |
| 105-165 | +0.001 | +0.112 | -0.006 |
| 170-230 | +0.002 | +0.102 | -0.002 |
| 235-295 | +0.001 | +0.098 | +0.004 |

=> a **constant offset of about +0.1 deg in w_y** that the in-water "GOOD" calibration reads. Either the in-water calibration retains a real
y residual (an oracle with mocap scale, `validate_gt.py`, finds +0.26 / +0.29 deg on two stretches, sign agrees, magnitude does not;
that oracle itself is noisy: unknown body-camera lever arm, stereo/GT scale ratio IQR 0.99-1.40) or the estimator has an absolute bias
(e.g. intrinsics). **Unresolved.** Consequence: in absolute mode the GOOD rig is flagged CHECK on 3/3 stretches (false-alarm rate 1.00).

Verdict table, absolute mode, per level (3 stretches, flagged = verdict != OK; "truth" = verdict of the injected value alone):

| injected | axis | flag rate | verdict == truth |
|---|---|---|---|
| 0 (false alarm) | all | 1.00 (CHECK) | 0.00 (truth OK) |
| 0.1 | x / y / z | 1.00 / 0.50 / 1.00 | 1.00 / 0.33 / 0.00 |
| 0.2 | x / y / z | 1.00 / 1.00 / 1.00 | 1.00 / 0.50 / 0.00 |
| 0.3 | x / y / z | 1.00 / 1.00 / 1.00 | 1.00 / 0.50 / 1.00 |
| 0.5 | x / y / z | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |

(the +-0.1/+-0.2 y cases are decided by the +0.1 offset, not by the injection; z is verdicted CHECK from 0.3 deg because 0.2 deg about z is only ~0.5 px RMS vertical disparity.)

Differential mode (reference = estimate on a different stretch of the same rig, dev stretch 40-100): false alarm 0.00; flag rate for y = +-0.1 deg 0.50, for any |w| >= 0.2 on x, y: 1.00.

### Experiment B: image-level injection (`warp_study.py`, stretches 105 and 170; the RIGHT image is resampled as if the camera were rotated, tracks re-extracted, GOOD calibration unchanged) - the honest accuracy test of the whole chain
Shift of the fused estimate relative to the unwarped stretch vs injected w (deg); n = 12 (5 injections + air case, 2 stretches):

| axis | bias of (shift - inj) | RMSE |
|---|---|---|
| x | -0.002 | 0.002 |
| y | -0.058 | 0.059 |
| z | +0.014 | 0.015 |

y: +0.3 -> +0.233 / +0.227; -0.3 -> -0.345 / -0.341; +0.1 -> +0.062 / +0.050; air case (-0.275) -> -0.332 / -0.341. y shift is also -0.05..-0.08 for
x and z injections, i.e. a nearly constant offset introduced by the warp (suspected resampling blur/interpolation mismatch between left and right; not
investigated), so the y *slope* is ~0.95-1.0 and the offset ~-0.06 deg (0.5 px). Detection of the real air-extrinsic change (differential verdict) = RECALIBRATE on both stretches.
Per-method y shift in the warp study: static epi and motion agree within 0.03-0.2 deg (motion is noisier, e.g. x+0.3 on 170: +0.144).

### Experiment C: window length (`window_study.py`, `summarize_window.py`), injection of 0, y+-0.1, y+-0.3, x+0.2, z+0.2; windows of the 3 eval stretches
Repeatability of the unperturbed estimate across disjoint windows:

| window | n | y mean | y sd | x sd | z sd | abs-mode false alarm | diff-mode false alarm | y+-0.1 flagged (diff) | y+-0.3 flagged (diff) |
|---|---|---|---|---|---|---|---|---|---|
| 10 s | 18 | +0.134 | 0.094 | 0.002 | 0.030 | 0.67 | 0.28 | 0.64 | 0.97 |
| 20 s | 9 | +0.109 | 0.044 | 0.001 | 0.010 | 0.56 | 0.00 | 0.50 | 1.00 |
| 30 s | 6 | +0.118 | 0.030 | 0.001 | 0.011 | 1.00 | 0.00 | 0.58 | 1.00 |
| 60 s | 3 | +0.108 | 0.008 | 0.001 | 0.005 | 1.00 | 0.00 | 1.00 | 1.00 |

(diff-mode reference here = mean zero-injection estimate of all windows of that length, i.e. NOT independent; y sd at 60 s is from only 3 stretches.)
Practical resolution on this rig: x < 0.01 deg, z ~0.01 deg, y ~0.03 deg (30 s) to 0.09 deg (10 s) repeatability; w_y changes of >= 0.1 deg are detectable only from >= 60 s,
>= 0.3 deg from 10 s.

### Experiment D: bus_outside, given calibration, radtan 1600x1200, B = 0.2318 m, f = 1167 px (`run_bus.py`; no GT for the true error)
| stretch | matches / mono feats / pairs | w_x | w_y fused (epi) | 90 % CI y | motion y (SE) | RMS dy | verdict |
|---|---|---|---|---|---|---|---|
| 100-160 | 3417 / 608 / 14 | +0.095 | -0.305 (-0.301) | [-0.531, -0.201] | -4.0 (3.4) | 2.2 px | RECALIBRATE |
| 250-310 | 2974 / 890 / 21 | +0.087 | -0.336 (-0.334) | [-0.392, -0.259] | -2.7 (1.5) | 2.1 px | RECALIBRATE |
| 400-460 | 7104 / 2347 / 27 | +0.096 | -0.500 (-0.498) | [-0.583, -0.454] | -2.0 (1.5) | 2.4 px | RECALIBRATE |

The bus stretches agree in sign and are consistent in w_x (+0.09) but w_y varies by 0.2 deg between stretches, more than the bootstrap CIs; the motion method cannot confirm
anything here (few tracked pairs, SE >> 1 deg). Treat as "something to look at", NOT a verified error: the bus rig was not necessarily calibrated in water, the radtan
model of a 1600 px lens may leave systematic x'y' residuals, and the CIs are likely too narrow.

### Runtime (laptop CPU, single process)
Extraction (KLT, 60 s of 20 Hz pairs, 800 corners, 10 Hz keyframes): 6-8 s warm (22-33 s when other jobs ran); prepare 1.1 s; estimate without CI 0.3 s; with 30 bootstrap draws ~9.5 s per candidate.
Bag decode dominates 60 s of data. No GPU.

## Limits (be explicit)
- Absolute accuracy of w_y is not established: the estimator reads +0.10 deg on the GOOD in-water calibration; the oracle suggests the true residual may be larger, the estimator may be biased by intrinsics. Only *changes* (differential mode) are validated, and only in simulation of error (A, B), plus one real change (air extrinsic) which is recovered (-0.275 true, -0.332 / -0.341 estimated from warped images).
- Single rig (NTNU, flat-port camera in a tank, ranges ~1.2-4 m: disparity 12-42 px) plus one unlabeled second rig. Depth-error thresholds are computed for 3 and 8 m but the data contain few points beyond 4 m.
- Baseline direction/translation errors, intrinsic errors, time offsets and rolling effects are not estimated; they alias into w.
- w_y from the static term needs a wide field of view and well-distributed matches; the motion method needs parallax and range diversity and is noisy (SE 0.04-0.14 deg NTNU, >1 deg bus).
- Verdict thresholds are fixed by the depth-error argument, not tuned to maximise detection; extraction settings (gap 6, 800 corners, temporal FB 0.5 px) were picked on the dev stretch (40-100 s), the same bag as the eval stretches (different time spans, same vehicle/scene).
- Bootstrap CIs are optimistic: matches are not independent, and stretch-to-stretch spread (bus) exceeds them.
- Warp experiment shows a ~-0.06 deg y offset of unknown origin (likely resampling blur).

## How to run
Environment: `/home/nevle.nehal/.venvs/rosbags/bin/python` (numpy 2.5, cv2 5.0, rosbags 0.11.7, ruamel; no scipy). The estimator itself needs only numpy + OpenCV (+ PyYAML or ruamel); `rosbags` only for bag reading.
```
cd calib_health
PY=/home/nevle.nehal/.venvs/rosbags/bin/python
$PY extract_cache.py ntnu 105 60          # tracks -> cache/ (2-3 MB per stretch)
$PY baseline.py --boot 30 | tee results/results_baseline.log   # experiment A (~18 min), then: $PY summarize.py
$PY warp_study.py 105,170                  # experiment B (~3 min)
$PY window_study.py && $PY summarize_window.py   # experiment C
$PY run_bus.py 100,250,400 60              # experiment D
$PY validate_gt.py 40 60                   # mocap-scale oracle (not part of the tool)
python3 -m pytest -q tests                 # synthetic tests, 13 tests, ~1.5 s, no bag needed
```
Programmatic use: `prep = estimators.prepare(rig, stretch); r = estimators.estimate(prep, rig); estimators.verdict(rig, r.fused_deg)`.
`estimate` needs n_boot >= 1 (the fusion weights come from the bootstrap).

## Tests (`tests/test_synthetic.py`)
Synthetic fisheye stereo scene (ranges 1.2-6 m, random camera motion, 0.2 px noise) with injected rotations: static fit recovers x and z (+-0.03 deg),
static fit recovers y via the x'y' term (+-0.08), motion intercept recovers y (+-0.12), full estimate + verdict (0 -> OK, 0.5 deg about y -> RECALIBRATE),
inject round trip, threshold arithmetic. A mutation (basis-column to axis mapping bug that occurred during development) makes 6 of 13 tests fail.

## Validation in the VIO: is the +0.10 deg w_y reading real? (ws1, 2026-11-04) - NO

Pre-declared test: rotate the in-water stereo extrinsic (T_cn_cnm1) by the correction each hypothesis implies
and run uw_ct_vio (best config h05) on all 6 NTNU 2023 trajectories. If the tool's absolute reading were right,
the -0.10 deg correction should improve ATE.

| stereo extrinsic | mean ATE (m) | mean RPE (%) |
|---|---|---|
| in-water calibration as given | **0.401** | **6.75** |
| -0.10 deg about y (tool's correction) | 0.431 | 7.28 |
| -0.27 deg about y (mocap-oracle correction) | 0.664 | 8.58 |
| +0.10 deg about y (wrong-sign control) | 0.510 | 7.24 |

The given in-water calibration is best; both corrections and the control are worse. So the +0.10 deg absolute
reading is an estimator bias (intrinsics/distortion aliasing into the x'y' term is the leading suspect), not a
calibration residual, and the mocap oracle is wrong too. Conclusion for the product: use the tool in
DIFFERENTIAL mode (compare against a reference recorded with a known-good calibration on the same rig), where
it detected every >= 0.2 deg change with 0 false alarms at >= 20 s windows; do not trust its absolute w_y.
The VIO sensitivity itself is a useful number: 0.1 deg of stereo rotation error costs ~8-27 % ATE here.
