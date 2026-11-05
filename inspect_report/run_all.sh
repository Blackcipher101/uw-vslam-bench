#!/usr/bin/env bash
# Usage: run_all.sh <bag_dir_or_db3> <camchain.yaml> <traj.tum|none> <out_dir>
# Env:   STAGES="keyframes detect localize report eval"  (default: all; eval needs $LABELS)
#        THR=0.15 MODE=base|ens VLM=Qwen/Qwen3-VL-4B-Instruct LABELS=<labels.json> SOURCE_NAME=<text>
set -euo pipefail
[ $# -eq 4 ] || { sed -n 2,5p "$0"; exit 2; }
BAG=$(readlink -f "$1"); CAM=$(readlink -f "$2"); TRAJ=$3; OUT=$(mkdir -p "$4" && readlink -f "$4")
[ "$TRAJ" = none ] || TRAJ=$(readlink -f "$TRAJ")
HERE=$(cd "$(dirname "$0")" && pwd)
STAGES=${STAGES:-"keyframes detect localize report eval"}
THR=${THR:-0.15}; MODE=${MODE:-base}; VLM=${VLM:-Qwen/Qwen3-VL-4B-Instruct}
VENV=$HERE/.venv
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
  "$VENV/bin/pip" install -r "$HERE/requirements.txt"
fi
PY="$VENV/bin/python"; export PYTHONPATH=$HERE; mkdir -p "$OUT/logs"
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,torch.cuda.get_device_name(0))" | tee "$OUT/logs/env.log"
run() { local s=$1; shift; echo "=== stage $s ===" | tee -a "$OUT/logs/all.log"; "$@" 2>&1 | tee "$OUT/logs/$s.log"; }
for s in $STAGES; do
  case $s in
    keyframes) run keyframes $PY "$HERE/keyframes.py" --bag "$BAG" --camchain "$CAM" --traj "$TRAJ" --out "$OUT" ;;
    detect)    run detect $PY "$HERE/detect.py" --out "$OUT" ;;
    localize)  run localize $PY "$HERE/localize.py" --out "$OUT" --thr "$THR" --mode "$MODE" ;;
    report)    run report $PY "$HERE/report.py" --out "$OUT" --vlm "$VLM" --source-name "${SOURCE_NAME:-}" ;;
    eval)      if [ -n "${LABELS:-}" ] && [ -f "$LABELS" ]; then run eval $PY "$HERE/eval.py" --out "$OUT" --labels "$LABELS" --thr "$THR"; else echo "eval skipped: set LABELS=<labels.json>"; fi ;;
    *) echo "unknown stage $s"; exit 2 ;;
  esac
done
