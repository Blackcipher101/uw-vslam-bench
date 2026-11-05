#!/usr/bin/env bash
# Build MASt3R-SLAM into ~/.venvs/mast3r, reusing the torch 2.11+cu128 install from a donor venv
# (saves ~5 GB of disk). Compiles the CUDA extensions for the local GPU only.
#
#   setup_mast3r.sh [donor_site_packages] [cuda_arch]
set -euo pipefail
donor=${1:-$HOME/.venvs/metricanything/lib/python3.12/site-packages}
arch=${2:-120}
here="$(dirname "$(readlink -f "$0")")"; tp="$here/../third_party"
venv=$HOME/.venvs/mast3r
cuda=$HOME/miniforge3/envs/cuda128   # nvcc 12.8, matches torch's CUDA

[ -d "$venv" ] || /usr/bin/python3.12 -m venv "$venv"
echo "$donor" >| "$venv/lib/python3.12/site-packages/donor_torch.pth"
py="$venv/bin/python"
"$py" -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__)"

# conda's CUDA keeps headers/libs under targets/, which torch's extension builder does not search.
export CPATH=$cuda/targets/x86_64-linux/include${CPATH:+:$CPATH} \
       LIBRARY_PATH=$cuda/targets/x86_64-linux/lib:$cuda/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
export CUDA_HOME=$cuda PATH=$cuda/bin:$PATH TORCH_CUDA_ARCH_LIST="${arch:0:-1}.${arch: -1}" \
       MAST3R_SLAM_CUDA_ARCHS=$arch MAX_JOBS=4
"$py" -m pip install -q roma trimesh natsort plyfile ninja evo moderngl moderngl-window "pyglet<2" imgui
# lietorch: reuse MASt3R-SLAM's eigen checkout instead of cloning the submodule.
[ -e "$tp/lietorch/eigen/Eigen" ] || { rmdir "$tp/lietorch/eigen" 2>/dev/null || true; ln -s "$tp/MASt3R-SLAM/thirdparty/eigen" "$tp/lietorch/eigen"; }
"$py" -m pip install --no-build-isolation --no-deps -e "$tp/lietorch"
"$py" -m pip install --no-deps -e "$tp/MASt3R-SLAM/thirdparty/mast3r"
"$py" -m pip install --no-deps -e "$tp/MASt3R-SLAM/thirdparty/in3d"
"$py" -m pip install --no-build-isolation --no-deps -e "$tp/MASt3R-SLAM"
"$py" -c "import lietorch, mast3r_slam_backends, mast3r, in3d; print('mast3r-slam import OK')"
