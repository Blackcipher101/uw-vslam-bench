# uw-vslam-bench: underwater stereo-inertial state estimation and inspection, with measured baselines

Research code for underwater robots: a GTSAM stereo-visual-inertial-pressure estimator (`uw_ct_vio`) and five
projects built on it. Every project has an honest baseline measured on public underwater datasets. All runs are
offline (recorded bags); nothing here has been run on a vehicle.

## Projects and baseline results (2026-11-04)

| # | Project | Dir | Baseline result | Status |
|---|---|---|---|---|
| 1 | **Tightly coupled DVL that survives dropouts** | `uw_ct_vio/`, `scripts/sim_dvl.py`, `DESIGN_DVL.md` | 20 s camera blackout: mean ATE **3.88 m -> 0.66 m** with DVL (worst 10.7 -> 0.94 m); DVL lost inside the blackout: 0.89 m; no dropout/outlier scenario worse than no DVL | strong; **simulated DVL** only |
| 2 | **GPU (CUDA) KLT stereo front end** | `gpu_frontend/` | front end 19.3 -> 9.5 ms/frame on ws1 (2.0x), 37.7 -> 7.7 ms on laptop (4.9x); end-to-end ATE no loss (0.401 vs 0.341 m, 6 traj) | strong; Orin not run |
| 3 | **Auto-generated inspection report (detector + VLM)** | `inspect_report/` | VLM verifier with frame context: true findings confirmed 8/8, precision 0.89 (crop-only verifier: 0/8); report faithfulness checks; ~2 min for a 10-min dive on an RTX 5080 | promising; small self-labelled set |
| 4 | **In-water stereo calibration health check** | `calib_health/` | differential mode detects every >= 0.2 deg rotation change, 0 false alarms (>= 20 s windows); absolute +0.1 deg reading shown to be bias by a VIO test | useful in differential mode |
| 5 | Learned localization-failure warning | `health/` | LOTO AUROC 0.63, held-out AUROC 0.57 (chance 0.5): **negative** | negative baseline |
| 6 | Sonar projected into the camera image | `sonar_cam/` | AFRL rig: scan plane perpendicular to the optical axis -> no camera/sonar overlap; no association found | negative (rig geometry) |

Estimator base (`DESIGN_V2.md`): NTNU 2023 subset-3, 6 trajectories with mocap, mean ATE **0.401 m** / RPE
6.75 % (baselines on the same bags: Basalt 1.10 m with the in-air stereo extrinsic, SVIn 1.80 m, ORB-SLAM3
initialises on 3/6, OpenVINS 3/6 with dynamic init). On the NTNU 2025 dev set with the same in-water stereo
for both: ours (huber 0.5) 0.552 m / 2.97 % vs Basalt 0.583 m / 2.24 % (better ATE, worse RPE).

## Datasets

- NTNU underwater (2023 subset-3 with mocap; 2025 mclab/fjord with ReAqROVIO reference; fjord_6 kept blind).
- AFRL stereo-VI underwater "bus_outside" (COLMAP GT; Imagenex 831L profiling sonar).
- Not yet: a real DVL dataset (Heriot-Watt Tank dataset needs a request form; SOLAQUA unverified).

## Layout

- `uw_ct_vio/` C++/GTSAM estimator (fixed-lag iSAM2, IMU preintegration, stereo, pressure depth, DVL, dense
  stereo factor (off)), `app/run_bag.cpp` offline runner, gtests.
- `gpu_frontend/` CUDA kernels + drop-in `GpuKltStereoTracker`, bench, tests.
- `inspect_report/` keyframes -> OWLv2 detection -> stereo localisation -> Qwen3-VL verification/report, eval.
- `calib_health/` stereo extrinsic error estimation, injection experiments.
- `health/` frame-log features, drift labels, LOTO/held-out evaluation.
- `sonar_cam/` sonar decoding, projection, sonar-vs-stereo baseline.
- `scripts/` evaluation (`evaluate.py`: ATE/RPE with hand-eye), dataset tools, ablation and benchmark runners.
- `configs/`, `patches/` baseline configs and build patches; `DESIGN_*.md`, `PLAN.md` design notes and results.

Not in the repo: datasets, `deps/` (GTSAM 4.2 local prefix), `third_party/` (separate clones; SuperPoint
weights, MASt3R and others are non-commercial), build trees, large outputs.

## Build (summary)

GTSAM 4.2.0 (+unstable) built to a local prefix (see `DESIGN_PHASE2.md` / ws1 build script), ROS 2 Jazzy
(rosbag2), OpenCV 4. `colcon build --base-paths uw_ct_vio --cmake-args -DGTSAM_DIR=<prefix>/lib/cmake/GTSAM
-DGTSAM_UNSTABLE_DIR=<prefix>/lib/cmake/GTSAM_UNSTABLE -DBUILD_TESTING=ON`. Source ROS from bash, not zsh.
Each project README has its exact commands.

## Licences of models used

OWLv2 and Qwen3-VL-4B-Instruct: Apache-2.0. GTSAM: BSD. cuVSLAM, GenZ-LIO, AirSLAM: ideas only, no code.
This repository has no licence file yet: choose one before publishing.
