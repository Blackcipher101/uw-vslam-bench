# sonar_cam: Imagenex 831L -> stereo camera projection, bus_outside (offline, no hardware)

Use `/usr/bin/python3` (numpy, scipy, cv2 4.6, yaml, pytest). Outputs go to `out/` (git-ignored, ~65 MB).

## Commands
```
python3 depth_cache.py --out out/depth_cache.npz --step 2        # SGBM depth cache, 3667 frames, ~3 min
python3 project.py --traj <tum> --extrinsic svin --start-s 100 --duration-s 40 --window-s 20 \
        --out-dir out/overlay_svin --video out/overlay_svin.mp4 --png-every 1000
python3 baseline.py --cache out/depth_cache.npz --traj <tum> --window-s 20 --min-range 0.5 --out out/baseline_w20_x.json
python3 window_sweep.py --cache out/depth_cache.npz --traj <tum>
python3 calibrate.py --cache out/depth_cache.npz --out out/calib.json   # pose-free rotation search
python3 calibrate.py --cache out/depth_cache.npz --picker contrast --min-range 0.5 --step-deg 8 --max-pairs 8000 --out out/calib_contrast.json
python3 -m pytest -q tests                                              # 13 tests
```
`<tum>` = `/home/nevle.nehal/svin_data/bus_outside/bus_outside.txt` (COLMAP GT, metric, T_world_cam0, 3388 poses ~6 Hz)
or `/home/nevle.nehal/uw-vslam-bench/results/bus_outside/uwct/r2_off.tum` (uw_ct_vio "off" base run, 7331 poses; chosen
from the `results/bus_outside/uwct/` ablation table, r2_off == s2_off, ATE-SE3 1.125 m; the choice of "base" is my reading, not confirmed).

## Data decoding (custom msgs not installed; hand-decoded CDR, `sonar_cam/data.py`)
`/imagenex831l/range` (imagenex831l/msg/ProcessedRange, 316 B, 55923 msgs, ~95 Hz): Header (frame_id "sonar");
`uint8[] intensity` (len 250, offset 28); 4 unidentified bytes (0x00000100); `float32 head_position` [deg] at byte 284;
`uint32` (=1, unidentified); `float32 max_range` [m] at 292 (=6.0 in all msgs); 5 more words unidentified and unused.
Evidence: head steps +0.9 deg and wraps -180..179.1 (400 steps/rev, 4.2 s); intensity bin = max_range/250 = 0.024 m.
Field names `head_position`, `max_range`, `intensity` and the deg->rad conversion come from SVIn's (commented-out)
`Subscriber::sonarCallback` (okvis_ros/src/Subscriber.cpp:359-379). `/range_raw` holds the same 250 bytes after a 32 B raw header.
Header stamp == bag time (<1 us). Range pick = SVIn rule (argmax of first 150 bins, intensity>10, range<4.5 m,
range=(idx+1)*0.024). Point = (r cos h, r sin h, 0) in the sonar frame (SonarError.cpp:144-150).
Names of the unidentified words were not derived; they are unused.

## Rig / extrinsic
`config_stereorig_v1.yaml` matches: its camera intrinsics/distortion equal camchain exactly (v2 has fx=1156.5, not 1167.87)
and `inv(camchain T_cam_imu) == T_SC` to 5e-13 for both cams (test `test_svin_rig_v1_matches_camchain`).
`T_c0_So = inv(T_SC0) T_SSo`: sonar x -> cam +y, sonar y -> cam -x, sonar z -> cam +z; t = (0.127, 0.227, 0.177) m.
cam0 = `/slave1` = left (camchain rostopic).
**Consequence: the scan plane (sonar x-y) is perpendicular to the camera optical axis, 0.177 m ahead of the camera, so at
the same instant essentially no sonar point is in the image.** Points reach the FOV only through yaw/reverse motion, i.e.
from stale pings placed with drifting poses.

## Results (SVIn extrinsic; residual = z_stereo - z_sonar, z in rectified-left frame; SGBM 800x600, 1/8-res block-median depth)
Same-time projection (window 4.2 s): GT poses 0 comparable points (0/133k in-image); VIO poses 116 pairs (8 frames).

| window | traj | n pairs | unique pings | median (m) | MAD (m) | inlier frac |
|---|---|---|---|---|---|---|
| 4.2 s | VIO | 116 | 67 | +1.36 | 0.17 | 0.000 |
| 10 s | GT | 1169 | 174 | -1.49 | 0.63 | 0.029 |
| 20 s | GT | 16414 | 697 | -1.70 | 0.96 | 0.056 |
| 40 s | GT | 101461 | 2957 | -1.89 | 0.86 | 0.031 |
| 20 s | VIO | 13889 | 850 | -1.21 | 1.64 | 0.058 |

Inlier = |res| < 0.15 m + 5 % z. Full tables: `out/window_sweep_*.log`, `out/baseline_w20_*.json` (min range 0.5 m; includes Spearman).
Spearman(z_sonar, z_stereo), 20 s window: GT -0.24, VIO -0.55 (negative = selection effect: long windows keep far, old points).
Relative error by sonar-z bin (20 s, GT): 1.5-2.0 m -18 %; 2.0-2.5 -21 %; 2.5-3.0 -49 %; 3.0-3.6 -61 %; 3.6-4.5 -65 %
(MAD 3-21 %). Residual is dominated by sonar z itself: stereo z is ~1-2 m wherever valid.
Time offset sweep -0.5..+0.5 s (0.1 step, 20 s window): median -1.69..-1.73 m (GT), -1.35..-1.37 (VIO); Spearman -0.21..-0.24 (GT),
-0.53..-0.56 (VIO); inlier 4.6-7.5 %: flat, no optimum.
Perturbations (rot +-1/2/5 deg, trans +-2/5/10 cm per camera axis): median shifts < 0.05 m, Spearman within -0.14..-0.31 (GT),
-0.44..-0.62 (VIO): no sensitivity.
Contrast picker (`--picker contrast`: prominence over a running median >= 30; avoids the 20 % of SVIn picks at the window edge)
gives the same picture.
Pose-free extrinsic search (same-time pairs, 6 deg SO(3) grid, both heading signs, 12000 pairs): best matches real 201/210 vs
time-shifted null 201/208 (contrast picker, 8 deg grid: real 147/152 vs null 181/165). The real pairing never beats the null,
so no correction can be estimated and none is reported. (`out/calib_svinpicker.json` holds a "refined" extrinsic: it fits noise, ignore.)

## Interpretation and limits
- This data neither validates nor refutes the SVIn extrinsic: the overlapping pairs are stale and show no sonar/stereo association.
- Not done: pose-dependent (windowed) extrinsic search; alternative mounts; independent check of stereo accuracy (murky
  water, 25-45 % valid pixels, 1/8-res cache); GT is COLMAP, not independent truth.
- Heading zero is assumed as in SVIn; a heading offset equals a rotation about sonar z, which the rotation search spans.
- Overlay `out/overlay_svin/best047_*.png`: sonar points (turbo by range) sit on the right image edge near the bus hull;
  no alignment judgement was made.
- The remote workstation was not used (blocked for this agent); everything ran on the laptop.
