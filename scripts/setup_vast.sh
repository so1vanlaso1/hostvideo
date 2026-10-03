#!/usr/bin/env bash
set -euo pipefail
umask 077

SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
H3_ROOT="${H3_ROOT:-/data/minimax-h3}"
export H3_ROOT
PROFILE="${H3_PROFILE:-int8-encoder}"
COMFY_REVISION=e9027f2b30f37bb3052714eb08fcf479542f4fc0
DOWNLOAD=false
if [[ "${1:-}" == "--download-models" ]]; then
  DOWNLOAD=true
elif [[ -n "${1:-}" ]]; then
  echo 'Usage: H3_ROOT=/data/minimax-h3 H3_PROFILE=int8-encoder bash scripts/setup_vast.sh [--download-models]' >&2
  exit 2
fi
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Run this setup on the Linux x86-64 Vast instance.' >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo 'NVIDIA GPU access is missing; choose a GPU template/host.' >&2; exit 1; }
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
[[ "${DRIVER%%.*}" -ge 580 ]] || { echo 'Choose a host with a compatible R580-or-newer driver for CUDA 13.0.' >&2; exit 1; }

if [[ "$H3_ROOT" == /data/* ]]; then
  mountpoint -q /data || { echo '/data is not a mounted volume. Attach/verify persistent storage or explicitly set H3_ROOT to the intended disk.' >&2; exit 1; }
fi
for tool in git ffmpeg ffprobe python3; do
  command -v "$tool" >/dev/null || { echo "Missing $tool. Install git ffmpeg python3 python3-venv first." >&2; exit 1; }
done
mkdir -p "$H3_ROOT"/{models,input,output,temp,user,logs,locks,jobs,private}
chmod 700 "$H3_ROOT/private"
command -v flock >/dev/null || { echo 'Install util-linux for flock.' >&2; exit 1; }
exec 9>"$H3_ROOT/locks/setup.lock"
flock -n 9 || { echo 'Another setup process is running.' >&2; exit 1; }

if [[ ! -d "$H3_ROOT/ComfyUI/.git" ]]; then
  [[ ! -e "$H3_ROOT/ComfyUI" ]] || { echo 'An unmanaged ComfyUI directory already exists; inspect it before continuing.' >&2; exit 1; }
  git clone --depth 1 https://github.com/Comfy-Org/ComfyUI.git "$H3_ROOT/ComfyUI"
fi
REMOTE="$(git -C "$H3_ROOT/ComfyUI" remote get-url origin)"
[[ "$REMOTE" == https://github.com/Comfy-Org/ComfyUI.git ]] || { echo 'Existing checkout has an unexpected origin.' >&2; exit 1; }
[[ -z "$(git -C "$H3_ROOT/ComfyUI" status --porcelain --untracked-files=no)" ]] || { echo 'ComfyUI has local changes; preserve them before setup.' >&2; exit 1; }
git -C "$H3_ROOT/ComfyUI" fetch --depth 1 origin "$COMFY_REVISION"
git -C "$H3_ROOT/ComfyUI" checkout --detach "$COMFY_REVISION"

if [[ ! -x "$H3_ROOT/venv/bin/python" ]]; then
  if command -v python3.11 >/dev/null; then
    python3.11 -m venv "$H3_ROOT/venv"
  else
    # uv supplies Python 3.11 without changing the container's system interpreter.
    if [[ ! -x "$H3_ROOT/bootstrap-env/bin/python" ]]; then
      python3 -m venv "$H3_ROOT/bootstrap-env"
    fi
    "$H3_ROOT/bootstrap-env/bin/python" -m pip install uv==0.8.17
    "$H3_ROOT/bootstrap-env/bin/uv" python install 3.11
    "$H3_ROOT/bootstrap-env/bin/uv" venv --seed --python 3.11 "$H3_ROOT/venv"
  fi
fi
PYTHON="$H3_ROOT/venv/bin/python"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,11), "Use a fresh Python 3.11 venv"'
printf 'torch==2.13.0\ntorchvision==0.28.0\n' > "$H3_ROOT/locks/torch-constraints.txt"
"$PYTHON" -m pip install --upgrade pip wheel setuptools
"$PYTHON" -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
if [[ -s "$H3_ROOT/locks/requirements-freeze.txt" ]]; then
  "$PYTHON" -m pip install -r "$H3_ROOT/locks/requirements-freeze.txt" --extra-index-url https://download.pytorch.org/whl/cu130
fi
"$PYTHON" -m pip install -c "$H3_ROOT/locks/torch-constraints.txt" -r "$H3_ROOT/ComfyUI/requirements.txt" -e "$SOURCE_DIR"
"$PYTHON" -m pip check

for name in h3_lowvram h3_native_adapters; do
  target="$H3_ROOT/ComfyUI/custom_nodes/$name"
  if [[ -e "$target" || -L "$target" ]]; then
    [[ -L "$target" && "$(readlink "$target")" == "$SOURCE_DIR/custom_nodes/$name" ]] || { echo "Refusing to replace $target" >&2; exit 1; }
  else
    ln -s "$SOURCE_DIR/custom_nodes/$name" "$target"
  fi
done
"$PYTHON" -m h3_pipeline --root "$H3_ROOT" preflight --profile "$PROFILE" --kernels --output "$H3_ROOT/locks/preflight.json"
# Store a portable package lock; the editable source is installed separately above.
"$PYTHON" -m pip freeze --exclude-editable > "$H3_ROOT/locks/requirements-freeze.txt"
"$PYTHON" --version > "$H3_ROOT/locks/python-version.txt"
"$PYTHON" -m h3_pipeline export-workflows --directory "$H3_ROOT/user/default/workflows"
if [[ "$DOWNLOAD" == true ]]; then
  "$PYTHON" -m h3_pipeline --root "$H3_ROOT" download-models --profile "$PROFILE"
fi
echo "Setup verified. Start ComfyUI with: H3_ROOT=$H3_ROOT bash $SOURCE_DIR/scripts/start_comfy.sh"
