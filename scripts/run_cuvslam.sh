#!/usr/bin/env bash
# ROS setup.bash must be sourced from bash (zsh leaves PYTHONPATH unset).
# cuvslam 17.0.1+cu12 needs cuBLAS/cuSOLVER/cuSPARSE 12 at runtime. Point
# CUVSLAM_CUDA_LIBS at any site-packages/nvidia dir that has them (e.g. a
# torch cu12x venv); defaults to the one on the author's machine.
set -eo pipefail
source /opt/ros/jazzy/setup.bash
NV=${CUVSLAM_CUDA_LIBS:-$HOME/.venvs/metricanything/lib/python3.12/site-packages/nvidia}
export LD_LIBRARY_PATH=$NV/cublas/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:${LD_LIBRARY_PATH}
PY=${CUVSLAM_PYTHON:-$HOME/.venvs/cuvslam/bin/python}
here="$(dirname "$(readlink -f "$0")")"
exec env PYTHONPATH="$here:$PYTHONPATH" "$PY" -u "$here/${CUVSLAM_ENTRY:-run_cuvslam.py}" "$@"
