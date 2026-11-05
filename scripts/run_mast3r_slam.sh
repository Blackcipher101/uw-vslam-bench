#!/usr/bin/env bash
# Run MASt3R-SLAM (calibrated, no viewer) on a folder from mast3r_prepare.py and write a
# bag-time TUM trajectory.
#
#   run_mast3r_slam.sh <prepared_dir> <out_dir> <tag> [config.yaml]
set -euo pipefail
prep=$(readlink -f "$1"); out=$(readlink -f "$2"); tag=$3
here="$(dirname "$(readlink -f "$0")")"
cfg=$(readlink -f "${4:-$here/../configs/mast3r_bus_outside.yaml}")
m3=$here/../third_party/MASt3R-SLAM
py=${MAST3R_PYTHON:-$HOME/.venvs/mast3r/bin/python}
mkdir -p "$out"
# MASt3R-SLAM resolves config/ and checkpoints/ relative to its repo and writes logs/<save-as>/.
cd "$m3"
rm -rf "logs/$tag"
"$py" -u main.py --dataset "$prep/images" --config "$cfg" --calib "$prep/intrinsics.yaml" \
  --no-viz --save-as "$tag" 2>&1 | tee "$out/${tag}_mast3r.log"
"$py" "$here/mast3r_to_tum.py" "logs/$tag/images.txt" "$prep/stamps.txt" "$out/${tag}_kf.tum"
# Keep only the trajectory; the per-keyframe images and .ply are large.
mv "logs/$tag/images.ply" "$out/${tag}.ply" 2>/dev/null || true
rm -rf "logs/$tag"
