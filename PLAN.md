# Underwater stereo-inertial odometry: plan

Goal: a stereo + IMU (+ depth/sonar when available) odometry for underwater robots that is
measurably more accurate and robust than SVIn on the AFRL data, real-time on a laptop/Orin GPU.

## Where we are (2026-10-29)

Full bus_outside bag (585 s, 54.7 m), SVIn back end, odometry, rate 0.5, clean harness:

| Front end | ATE SE3 | ATE Sim3 | fitted scale | RPE 5 m | keyframes |
|---|---|---|---|---|---|
| upstream SVIn (BRISK) | 0.94 m | 0.28 m | 0.80 | 5.6 % | 24 % |
| BRISK (our branch) | 0.96 m | 0.33 m | 0.80 | 5.7 % | 24 % |
| cuVSLAM tracks | 1.01 m | 0.31 m | 0.79 | 5.3 % | 13 % |
| SuperPoint+LightGlue tracks | 1.33 m | 0.73 m | 0.76 | 6.5 % | 60 % |
| cuVSLAM standalone | 5.19 m | 2.86 m | — | 45 % | — |

Per-60 s windows, each aligned on its own, all three SVIn front ends have the same error
(0.15–0.54 m, same peaks at 300 s and 480 s). So on this bag the front end is not the limit:
the shared error is in something common (ground truth, calibration, time offset) or in slow
global drift (SP+LG's 60 % keyframe rate shortens the window and costs scale/yaw).

## Phase 0: fix the measuring stick (2–3 days, laptop)

Findings so far (2026-10-29):
- SVIn's config matches the Kalibr calibration exactly (extrinsics, intrinsics, distortion); the
  cam–IMU time shift is -0.08 ms, so imageDelay 0 is right. Calibration transcription is not the issue.
- The COLMAP ground truth is ~20 % too small: frame-to-frame motion from stereo geometry alone
  (scripts/check_gt_scale.py, 510 pairs at 2 s) gives |t_gt|/|t_stereo| = 0.795 (IQR 0.75–0.83),
  the same 0.8 every VIO fitted. On bus_outside the Sim3 ATE and RPE are the meaningful metrics;
  metric (SE3) numbers need the NTNU motion-capture bags.
- NTNU traj_06 (mocap GT, scripts: make_svin_config.py, bag_odom_to_tum.py, evaluate.py --hand-eye
  --hand-eye-first-s 60): ROVIO (in the bag) ATE 0.65 m / RPE 11.7 %; SVIn BRISK ATE 2.86 m /
  RPE 33 %. Per-30 s windows SVIn is as good or better than ROVIO except 150–210 s (1.02 m), where
  it breaks down and never recovers. Mocap clock matches the IMU (gyro cross-correlation flat at 0).
  Hand-eye: relative-motion AX=XB is degenerate for a yawing ROV; the evaluator now fits lever arm
  + world alignment from positions on the first 60 s. RPE now skips segments across GT gaps > 1 s
  (bus_outside re-scored: upstream 0.285 / 4.44 %, BRISK 0.325 / 4.50 %, cuVSLAM 0.314 / 4.44 %,
  SP+LG 0.725 / 6.16 %, SP+LG keyframe fix 0.640 / 5.47 %; Sim3 ATE / RPE).
- SP+LG's 60 % keyframe rate comes from OKVIS's coverage test: most SP tracks never become landmarks
  (new-landmark cap), so the matched ratio sits at the 0.2 threshold. Fix: option
  keypoints_include_unmappable=false (only landmark / stereo tracks are keypoints).

Nothing later can be judged until the error floor is understood.

1. **Repeatability.** 3 runs each of BRISK and SP+LG; report mean ± spread. Differences inside the
   spread are not results.
2. **Ground-truth audit.** Every method fits best at scale ~0.8 to the COLMAP GT. Check the GT's
   metric scale against the stereo baseline (triangulate known points), its smoothness (jitter,
   jumps), and whether the per-window error peaks line up with GT reconstruction gaps.
3. **Calibration.** Confirm the Kalibr cam–IMU time shift is actually applied in SVIn; sweep
   ±10 ms and the extrinsic within its uncertainty; check IMU noise values. The per-window error
   being identical across front ends is the signature of a shared calibration error.
4. **Ground-truth dataset.** Add the NTNU underwater dataset, subset 3 (huggingface
   ntnu-arl/underwater-datasets, BSD-3): Alphasense stereo 20 Hz + IMU 200 Hz + pressure 50 Hz on an
   ROV in an indoor pool, motion-capture ground truth (TUM) for 6 trajectories. Extrinsics are
   in-air only (expect calibration work); bags up to ~18 GB, so store and run on workstation1.
   Keep bus_outside as the real-sea case. (HAUD: stereo+IMU, pool, mm GT — availability unverified.)
5. **SP+LG fixes.** 512 keypoints (as DIVO uses), keyframe rule that ignores never-matched
   keypoints, and ALIKED+LightGlue (commercially usable; SuperPoint weights are non-commercial).

Gate: a noise band per method, a trusted GT (or a corrected one), and calibration confirmed.

## Enhancement experiment (2026-10-29)

FUnIE-GAN (MIT; generator re-implemented in scripts/enhance.py, weights loaded weights_only, BatchNorm
folded into convs: exact to 1e-5, 28 ms/frame at 800x600 on the GPU). Same pipeline for every bag;
scripts/enhance_bag.py writes enhanced and control bags (control = same decode/grey/re-encode, no enhancement).
- Front end (scripts/probe_enhance.py): NTNU native +20–45 % SP+LG temporal inliers, worst frames 2–3x;
  bus_outside mixed (+54 % ORB at 160 s, −50 % SP stereo at 460 s, slightly worse in clear water).
  Enhanced JPEGs are 70 % larger at the same quality: the network adds high-frequency content (noise).
- End to end, SVIn BRISK thr10 on NTNU: raw 2.44 / 2.71 m; enhanced DIVERGED in 2/2 runs (289 m, 4.8 km).
  Enhancement is not usable with SVIn's BRISK front end. bus_outside control vs enhanced and NTNU
  enhanced at thr20: pending.

## Phase 2 status (2026-10-29)

uw_ct_vio built (GTSAM 4.2.0 + unstable in deps/): DepthGpFactor (4 tests), Estimator (IMU preint,
stereo landmarks + weak 5 m prior, gating, fixed-lag 3 s, reset on solver failure), KLT stereo tracker,
offline rosbag2 runner. Synthetic: 9.1 mm clean, 9.7 mm with 10 % outliers, 13.6 mm with depth.
NTNU traj_06 v1: ATE 1.05 m / RPE 9.3 % / scale 1.015 (ROVIO 0.65 / 11.7 %, SVIn 2.44 / 27 %),
0 solver failures, 38 ms/frame (real time at 20 Hz).

## Next: uw_ct_vio v2 (2026-10-29)

Plan in DESIGN_V2.md: raw images only; robustness in the tracker (affine-brightness KLT + NCC,
SNR-weighted measurement noise from the lamp field) and the estimator (gravity prior,
degeneracy-aware vision, depth factor, dense stereo factor), ALIKED+LightGlue keyframe association.
Baselines on raw bags: SVIn, ORB-SLAM3, OpenVINS, Basalt (no ROVIO). Each step must improve the mean
over all 6 NTNU mocap trajectories + bus_outside.

## Phase 1: does dense registration work underwater? (2–3 days, workstation1 GPU)

Run GLIM unchanged (depth-camera mode) on stereo point clouds + IMU.
- Stereo depth on the rectified pair (OpenCV SGBM first; learned stereo if SGBM is too sparse),
  depth-range and confidence filtering, per-point covariance from disparity noise.
- Build GTSAM 4.3 + gtsam_points + GLIM (official borglab/koide3 sources) on workstation1.

Gate: GLIM odometry at least matches SVIn, or clearly beats it in the feature-poor windows
(160 s, 460 s). If dense registration fails underwater, Phase 3 is dropped.

## Phase 2: continuous-time GTSAM back end (1–2 weeks)

New package `uw_ct_vio` (C++, ROS 2 node + offline bag runner, same TUM output and harness).
- **States** at camera times: pose, velocity, IMU bias.
- **Continuous-time prior:** white-noise-on-acceleration GP between states (as DIVO), so any
  measurement can be attached at its own timestamp by interpolation.
- **Factors:** IMU preintegration; SP+LG stereo landmarks (reuses the track files now, ONNX/TensorRT
  later); pressure-depth and sonar-range factors at their own stamps (SVIn has these sensors but
  they are disabled on ROS 2); robust losses.
- **Solver:** iSAM2 fixed-lag smoother, ~3 s window (no hand-written marginalisation).
- **Guard:** GenZ-style deviation gate: a visual update that pulls far from the IMU prediction is
  down-weighted (the cuVSLAM 2–3 m jumps).
- **Tests:** numerical-Jacobian tests for every new factor; synthetic trajectory with known
  ground truth for the GP interpolation; then the bags.

Gate: matches or beats SVIn on both sequences (outside the Phase 0 noise band); real-time factor
≥ 1 on the laptop.

## Phase 3: dense stereo-cloud factor (1–2 weeks)

Ideas from FAST-LIO2, GLIM and GenZ-LIO, reimplemented (GenZ-LIO is GPL v2):
- **Local map:** voxel map with per-voxel plane fit and covariance; each keyframe's stereo cloud
  registered directly to the map (FAST-LIO2 style), not just to the last keyframe.
- **Hybrid residual:** point-to-plane where a voxel holds a reliable plane, point-to-point
  otherwise, weighted by per-point stereo covariance (depth error grows with z²) — GenZ-LIO.
- **Adaptive budget:** GenZ's PD controller on voxel size / keypoint count / new-landmark cap so
  the point count stays steady as visibility changes.
- **GPU alternative:** gtsam_points VGICP factors (GLIM) for comparison.

Gate: better in the feature-poor windows with no regression elsewhere.

## Phase 4: global consistency (1 week)

GLIM-style submaps every few metres, submap-to-submap dense matching factors in a global iSAM2
graph; loop candidates from submap overlap plus SP/ALIKED descriptor retrieval.
Gate: lower full-bag ATE and higher loop-path coverage than SVIn's pose graph.

## GPU in GTSAM

- gtsam_points (koide3, MIT, GLIM's engine): GPU VGICP factors and GPU-aware optimisers
  (IncrementalFixedLagSmootherExt, ISAM2Ext); tested with GTSAM 4.2 and on Jetson Orin. Used for the
  Phase 3 dense factor.
- GTSAM develop CUDA solvers: SparseLevenbergMarquardtOptimizer (cuDSS or PCG) and a CUDA SfM/BA
  optimiser. Used for the Phase 4 global graph and batch re-optimisation. cuDSS >= 0.8.0 is a separate
  install.
- The live fixed-lag window (~3 s) stays on CPU iSAM2 unless profiling shows the solve is the
  bottleneck. Check early that gtsam_points (4.2) and the develop CUDA solvers (4.3) build together.

## Phase 5: real-time and deployment

ALIKED+LightGlue via TensorRT in C++, profiled on the laptop and the Orin (mean/P95 latency,
GPU memory, CPU). DIVO reports ~8 fps for SP+LG on an 8 GB laptop GPU; we need ≥12.6 Hz or a
frame-skip policy.

## Learned baselines (parallel, when approved)

MASt3R-SLAM (mono, built, run pending), MASt3R-Fusion and DBA-Fusion (learned + IMU) — both need
the GTSAM fork yuxuanzhou97/gtsam, whose build was blocked pending the user's approval.

## Licences

GTSAM BSD; GLIM and gtsam_points MIT; LightGlue Apache-2.0 (SuperPoint weights non-commercial,
use ALIKED/DISK); GenZ-LIO GPL v2 (ideas only); cuVSLAM NVIDIA community licence (not vendored);
MASt3R checkpoints CC BY-NC-SA.

## Compute and housekeeping

Laptop: RTX 5060 8 GB, 15 GB RAM, disk tight. Heavy builds/runs go to workstation1 (idle RTX 5080,
200 GB free). Delete spent builds and recordings as we go. Publishing the repo is a user action.
