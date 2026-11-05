#!/bin/bash
# Run uw_ct_vio on the benchmark bags with extra flags and score it, one variant per call.
#   [UWCT_BIN=path] [TRACK_MODE=augment|replace] scripts/uwct_ablate2.sh <variant_tag> [extra run_bag flags...]
# TRACK_MODE adds each dataset's learned-track file (~/svin_data/tracks/{ntnu06,bus}_aliked${TRACK_SUFFIX}.uwtrk).
# Writes results/<dataset>/uwct/<tag>.{tum,log} and appends one line per dataset to
# results/uwct_ablation.tsv. Base config: raw images + pressure depth (step 1).
# ROS setup scripts read unset variables, so source them before enabling -u.
source /opt/ros/jazzy/setup.bash
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
tag=$1; shift
export LD_LIBRARY_PATH=$PWD/deps/gtsam-4.2.0/lib:${LD_LIBRARY_PATH:-}
RUN=${UWCT_BIN:-./build_ws/build/uw_ct_vio/run_bag}
B=$HOME/svin_data/bus_outside; C=$HOME/svin_data/ntnu/calibration
ntnu() {
  $RUN --bag $HOME/svin_data/ntnu/traj_06/traj_06_ros2 --intrinsics $C/stereo_intrinsics_under_water/camchain-ariel_stereo_underwater.yaml \
    --imucam $C/stereo_extrinsics_above_water/camchain-imucam-ariel_stereo_extrinsics_above_water.yaml \
    --imu-yaml $C/stereo_extrinsics_above_water/imu0_alphasense_noise.yaml --left cam1 --right cam0 \
    --left-topic /alphasense_driver_ros/cam1 --right-topic /alphasense_driver_ros/cam0 --imu-topic /alphasense_driver_ros/imu \
    --pressure-topic /mavros/imu/static_pressure ${TRACK_MODE:+--track-file $HOME/svin_data/tracks/ntnu06_aliked${TRACK_SUFFIX:-}.uwtrk --track-mode $TRACK_MODE} "$@" --out results/ntnu_traj06/uwct/$tag.tum > results/ntnu_traj06/uwct/$tag.log 2>&1
  python3 scripts/evaluate.py $HOME/svin_data/ntnu/traj_06/traj_06_mocap.tum results/ntnu_traj06/uwct/$tag.tum --hand-eye --hand-eye-first-s 60
}
bus() {
  $RUN --bag $B/bus_outside.db3 --intrinsics $B/camchain_bus_outside.yaml --imucam $B/camchain_bus_outside.yaml --imu-yaml $B/imu.yaml \
    --left cam0 --right cam1 --left-topic /slave1/image_raw/compressed --right-topic /slave2/image_raw/compressed --imu-topic /imu/imu \
    --resize 0.5 --pressure-topic /bar30/pressure --pressure-scale 100 ${TRACK_MODE:+--track-file $HOME/svin_data/tracks/bus_aliked${TRACK_SUFFIX:-}.uwtrk --track-mode $TRACK_MODE} "$@" --out results/bus_outside/uwct/$tag.tum > results/bus_outside/uwct/$tag.log 2>&1
  python3 scripts/evaluate.py $B/bus_outside.txt results/bus_outside/uwct/$tag.tum
}
summ() {  # dataset json_line log -> tsv row
  python3 -c "
import json,sys,re
d=json.loads(sys.argv[2].split(' ',1)[1]); log=open(sys.argv[3]).read()
m=re.search(r'done: (\d+) frames, (\d+) without vision.*solver failures (\d+), gate recoveries (\d+); ms/frame track ([\d.]+) est ([\d.]+)',log)
print('\t'.join([sys.argv[4],sys.argv[1],f\"{d['ate_se3_m']:.3f}\",f\"{d['ate_sim3_m']:.3f}\",f\"{d['rpe_trans_pct_5m']:.2f}\",f\"{d['rpe_rot_deg_per_m_5m']:.3f}\"]+(list(m.groups()) if m else ['?']*6)))
" "$1" "$2" "$3" "$tag"
}
T=results/ablate_tmp; mkdir -p $T
ntnu "$@" > $T/ntnu_$tag.txt 2>&1 &
bus "$@" > $T/bus_$tag.txt 2>&1
wait
[ -f results/uwct_ablation.tsv ] || printf 'variant\tdataset\tate_se3\tate_sim3\trpe_pct\trpe_deg_m\tframes\tno_vision\tsolver_fail\trecoveries\tms_track\tms_est\n' > results/uwct_ablation.tsv
summ ntnu "$(grep '[.]tum {' $T/ntnu_$tag.txt | tail -1)" results/ntnu_traj06/uwct/$tag.log >> results/uwct_ablation.tsv
summ bus "$(grep '[.]tum {' $T/bus_$tag.txt | tail -1)" results/bus_outside/uwct/$tag.log >> results/uwct_ablation.tsv
tail -2 results/uwct_ablation.tsv
