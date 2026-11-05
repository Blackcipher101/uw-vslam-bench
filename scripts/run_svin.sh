#!/usr/bin/env bash
# Replay a bag through SVIn (okvis_node + pose_graph_node, no RViz) and record trajectories.
#
#   run_svin.sh <svin_ws> <base_config.yaml> <bag.db3> <out_dir> <tag> <brisk|cuvslam|track_file> [rate]
#
# The base config is copied and patched for headless benchmarking: no OpenCV windows,
# odometry published as T_WS (trackedBodyFrame: S), absolute pose_graph output dir, and
# the chosen frontend. The upstream config is not modified.
set -euo pipefail
ws=$1; base=$2; bag=$3; out=$4; tag=$5; frontend=$6; rate=${7:-0.5}
here="$(dirname "$(readlink -f "$0")")"
mkdir -p "$out"
cfg="$out/${tag}_config.yaml"

sed -e 's/^displayImages:.*/displayImages: false/' \
    -e 's/^\(\s*trackedBodyFrame:\).*/\1 S/' \
    -e "s|^\(\s*output_dir:\).*|\1 $out/${tag}_pose_graph|" "$base" >| "$cfg"
cat >> "$cfg" <<EOF

frontend:
    type: $frontend
EOF
# frontend track_file: TRACK_FILE=<path from lightglue_tracks.py>; MAX_NEW_LANDMARKS caps the
# new landmarks per frame (dense learned trackers give hundreds of stereo points per frame).
if [ "$frontend" = track_file ]; then
  [ -f "${TRACK_FILE:-}" ] || { echo "[$tag] ERROR: frontend track_file needs TRACK_FILE=<existing file>" >&2; exit 1; }
  echo "    track_file: \"$(readlink -f "$TRACK_FILE")\"" >> "$cfg"
fi
# KEYPOINTS_UNMAPPABLE=false hands OKVIS only tracks that are or can become landmarks.
if [ -n "${MAX_NEW_LANDMARKS:-}${KEYPOINTS_UNMAPPABLE:-}" ]; then
  echo "    cuvslam:" >> "$cfg"
  [ -z "${MAX_NEW_LANDMARKS:-}" ] || printf '        max_new_landmarks_per_frame: %d\n' "$MAX_NEW_LANDMARKS" >> "$cfg"
  [ -z "${KEYPOINTS_UNMAPPABLE:-}" ] || echo "        keypoints_include_unmappable: $KEYPOINTS_UNMAPPABLE" >> "$cfg"
fi
mkdir -p "$out/${tag}_pose_graph"

# Isolated DDS domain so the replay never meets live robot traffic.
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID_BENCH:-87}
set +u
source /opt/ros/jazzy/setup.bash
source "$ws/install/setup.bash"
# Optional overlay (e.g. a dev build of okvis_ros) sourced on top of the main install.
if [ -n "${SVIN_OVERLAY:-}" ]; then source "$SVIN_OVERLAY/local_setup.bash"; fi
set -u

# Exact process names (pgrep -x matches the command name, so it never matches this script).
NODE_NAMES=(okvis_node stereo_sync pose_graph_node)
leftover_nodes() {
  local n
  for n in "${NODE_NAMES[@]}"; do pgrep -x "$n" || true; done
}
# A node left from an earlier run republishes images/odometry into this one and corrupts it.
if [ -n "$(leftover_nodes)" ]; then
  echo "[$tag] ERROR: SVIn nodes already running (pids: $(leftover_nodes | tr '\n' ' ')); refusing to start" >&2
  exit 1
fi

# Each node runs in its own process group (setsid), so cleanup reaches the real node and not
# just the `ros2 run` wrapper, which does not forward signals to its child.
pgids=()
cleanup() {
  for g in "${pgids[@]}"; do kill -INT -- "-$g" 2>/dev/null || true; done
  sleep 3
  for g in "${pgids[@]}"; do kill -KILL -- "-$g" 2>/dev/null || true; done
  sleep 1
  if [ -n "$(leftover_nodes)" ]; then
    echo "[$tag] ERROR: nodes survived cleanup (pids: $(leftover_nodes | tr '\n' ' ')); kill them before the next run" >&2
    exit 1
  fi
}
trap cleanup EXIT

# Bag topics (defaults: AFRL stereo rig). COMPRESSED=true plays <topic>/compressed.
LEFT_TOPIC=${LEFT_TOPIC:-/slave1/image_raw}; RIGHT_TOPIC=${RIGHT_TOPIC:-/slave2/image_raw}
IMU_TOPIC=${IMU_TOPIC:-/imu/imu}; COMPRESSED=${COMPRESSED:-true}
case "$COMPRESSED" in
  true) play_topics=("$LEFT_TOPIC/compressed" "$RIGHT_TOPIC/compressed" "$IMU_TOPIC") ;;
  false) play_topics=("$LEFT_TOPIC" "$RIGHT_TOPIC" "$IMU_TOPIC") ;;
  *) echo "[$tag] ERROR: COMPRESSED must be true or false" >&2; exit 1 ;;
esac
setsid ros2 run okvis_ros stereo_sync --ros-args -p use_sim_time:=true -p config_filename:="$cfg" \
  -p left_img_topic:="$LEFT_TOPIC" -p right_img_topic:="$RIGHT_TOPIC" -p compressed:="$COMPRESSED" \
  >"$out/${tag}_stereo_sync.log" 2>&1 & pgids+=($!)
# pose_graph freezes for good after freeze_timeout_sec (steady clock, default 60 s) without a
# keyframe; reduced-rate replay stretches keyframe gaps in wall time, so raise it for benchmarks.
setsid ros2 run pose_graph pose_graph_node --ros-args -p use_sim_time:=true -p config_file:="$cfg" \
  -p freeze_timeout_sec:=3600.0 \
  >"$out/${tag}_pose_graph.log" 2>&1 & pgids+=($!)
# OKVIS_PREFIX: optional wrapper for okvis_node, e.g. "gdb -batch -ex run -ex bt --args".
setsid ros2 run ${OKVIS_PREFIX:+--prefix "$OKVIS_PREFIX"} okvis_ros okvis_node --ros-args \
  -p use_sim_time:=true -p config_filename:="$cfg" \
  -p mesh_file:=firefly.dae -r /camera0:=/cam0/image_raw -r /camera1:=/cam1/image_raw -r /imu:="$IMU_TOPIC" \
  >"$out/${tag}_okvis.log" 2>&1 & pgids+=($!)
PYTHONPATH="$here:${PYTHONPATH:-}" python3 -u "$here/record_svin.py" "$cfg" "$out" "$tag" \
  >"$out/${tag}_recorder.log" 2>&1 &
rec=$!
sleep 8  # let nodes come up before the first message

echo "[$tag] playing $bag at rate $rate"
ros2 bag play "$bag" --clock 100 --rate "$rate" ${PLAY_DURATION_S:+--playback-duration "$PLAY_DURATION_S"} \
  --topics "${play_topics[@]}" >"$out/${tag}_play.log" 2>&1
echo "[$tag] bag finished; flushing"
sleep 10  # let the last frames finish optimisation and pose-graph publishing
kill -INT "$rec"; wait "$rec" || true
cat "$out/${tag}_recorder.log"
