# uw-vslam-bench

Benchmark harness for underwater stereo-inertial estimation and inspection research, with measured baselines;
it holds the shared evaluation and baseline runners plus the two projects whose results came out negative. All runs
are offline (recorded bags); nothing ran on a vehicle.

| Learned failure warning (`health/`): negative | Sonar in the camera image (`sonar_cam/`): negative |
|---|---|
| [<img src="docs/figures/health_auc_bars.png" width="480">](health/README.md) | [<img src="docs/figures/sonar_geometry.png" width="480">](sonar_cam/README.md) |
| AUROC / AUPRC barely above chance, LOTO on NTNU underwater 2023 and held-out NTNU 2025 | the scan plane sits 0.177 m ahead of the lens, perpendicular to the optical axis, so same-time sonar points are out of view |

## Projects

The four projects with positive baselines live in their own repositories:

| Project | Repository | Baseline result |
|---|---|---|
| Tightly coupled DVL that survives dropouts (+ the `uw_ct_vio` estimator) | [`uw-vio-dvl`](https://github.com/Blackcipher101/uw-vio-dvl) | 20 s camera blackout: mean ATE **3.88 m -> 0.66 m** with (simulated) DVL; no dropout scenario worse than no DVL |
| CUDA KLT stereo front end | [`uw-gpu-klt`](https://github.com/Blackcipher101/uw-gpu-klt) | front end 2.0x (ws1) / 4.9x (laptop) faster, no end-to-end ATE loss |
| Auto-generated inspection report (detector + VLM) | [`uw-inspect-report`](https://github.com/Blackcipher101/uw-inspect-report) | context verifier confirms 8/8 true findings at precision 0.89 (crop-only: 0/8) |
| In-water stereo calibration health check | [`uw-calib-health`](https://github.com/Blackcipher101/uw-calib-health) | differential mode: every >= 0.2 deg change detected, 0 false alarms |

This repository keeps the shared benchmark harness and the two negative results:

| # | Project | Dir | Result |
|---|---|---|---|
| 5 | Learned localization-failure warning | [`health/`](health/README.md) | LOTO AUROC 0.63, held-out AUROC 0.57 (chance 0.5): **negative** |
| 6 | Sonar projected into the camera image | [`sonar_cam/`](sonar_cam/README.md) | AFRL rig: scan plane never in camera view, no association: **negative** |

Estimator base (NTNU 2023 subset-3, 6 trajectories with mocap): mean ATE **0.401 m** / RPE 6.75 %; Basalt 1.10 m
with the in-air stereo extrinsic, SVIn 1.80 m,
ORB-SLAM3 initialises on 3/6, OpenVINS 3/6 with dynamic init.

## Quick start

```bash
python3 -m pytest -q health/tests                 # health: label timing, feature causality, metrics
python3 scripts/evaluate.py <gt.tum> <est.tum> --hand-eye    # ATE/RPE with hand-eye
python3 health/make_figures.py                    # regenerate figures (sonar_cam/make_figures.py needs sonar_cam/out/)
```

## More

- [docs/DETAILS.md](docs/DETAILS.md): the harness, the estimator base and NTNU 2025 comparison, datasets, layout,
  build summary and model licences.
- [health/README.md](health/README.md): labels, features, LOTO and held-out results, risk-vs-time plot.
- [sonar_cam/README.md](sonar_cam/README.md): sonar decoding, projection, sweeps and overlays.
- [PLAN.md](PLAN.md): the overall research plan.

No real DVL dataset yet. This repository has no licence file yet: choose one before publishing.
