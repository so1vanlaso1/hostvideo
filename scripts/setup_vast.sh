#!/usr/bin/env bash
set -euo pipefail
umask 077

SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
COMFY_REVISION=e9027f2b30f37bb3052714eb08fcf479542f4fc0
KITCHEN_VERSION=0.2.37
KITCHEN_REVISION=be003b7c23c5b01328657955b8bc5d3f073d868e
SAGE_REVISION=d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
DOWNLOAD=true
START=true
usage() {
  cat <<'HELP'
Usage: bash setup.sh [options]

Install dependencies, verify GPU kernels, download models and start ComfyUI.
Run on Linux x86-64 (CUDA 12.8+/13.x) or ARM64 GB10 (CUDA 13.x), with nvcc.

  --root PATH         Runtime/model storage (auto: mounted /data, /workspace, .runtime)
  --profile NAME      int8-encoder (default), primary, fp8, fp8-int8-encoder
  --port NUMBER       ComfyUI port (default: 8188)
  --no-start          Install and download only
  --skip-download     Install only; also skips startup
  --download-models   Download models (already enabled by default; legacy option)
  -h, --help          Show this help

H3_ROOT, H3_PROFILE and H3_PORT also work. Successful setup saves these settings.
H3_MEMORY_MODE=auto|static|dynamic and H3_RESERVE_VRAM=GiB tune memory policy.
H3_INSTALL_POSE=1 installs optional CPU pose tools (default: off on ARM64).
Later: bash scripts/vast.sh start|stop|restart|status|logs
HELP
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root|--profile|--port)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --root) export H3_ROOT="$2" ;;
        --profile) export H3_PROFILE="$2" ;;
        --port) export H3_PORT="$2" ;;
      esac
      shift 2 ;;
    --no-start) START=false; shift ;;
    --skip-download) DOWNLOAD=false; START=false; shift ;;
    --download-models) DOWNLOAD=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
MACHINE="$(uname -m)"
[[ "$(uname -s)" == Linux ]] || { echo 'Run this setup on the Linux Vast instance.' >&2; exit 1; }
case "$MACHINE" in
  x86_64) ARM64=false ;;
  aarch64|arm64) ARM64=true ;;
  *) echo "Unsupported CPU architecture: $MACHINE. Use x86_64 or aarch64/ARM64." >&2; exit 1 ;;
esac
H3_INSTALL_POSE="${H3_INSTALL_POSE:-$([[ "$ARM64" == true ]] && echo 0 || echo 1)}"
[[ "$H3_INSTALL_POSE" == 0 || "$H3_INSTALL_POSE" == 1 ]] || { echo 'H3_INSTALL_POSE must be 0 or 1.' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'NVIDIA GPU access is missing; choose a GPU template/host.' >&2; exit 1; }
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
# Some development templates do not add the toolkit to PATH.
if [[ -n "${CUDA_HOME:-}" ]]; then
  [[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo 'CUDA_HOME must point to a development toolkit containing bin/nvcc.' >&2; exit 1; }
  export PATH="$CUDA_HOME/bin:$PATH"
fi
if ! command -v nvcc >/dev/null && [[ -x /usr/local/cuda/bin/nvcc ]]; then
  export PATH="/usr/local/cuda/bin:$PATH"
fi
command -v nvcc >/dev/null || { echo 'Choose a CUDA 12.8+ development template: nvcc is missing.' >&2; exit 1; }
NVCC_INFO="$(nvcc --version)"
[[ "$NVCC_INFO" =~ release[[:space:]]+([0-9]+)\.([0-9]+) ]] || { echo 'Cannot determine the CUDA toolkit version from nvcc.' >&2; exit 1; }
CUDA_MAJOR="${BASH_REMATCH[1]}"
CUDA_MINOR="${BASH_REMATCH[2]}"
CUDA_VERSION="$CUDA_MAJOR.$CUDA_MINOR"
if [[ "$ARM64" == true && "$CUDA_MAJOR" -ne 13 ]]; then
  echo 'ARM64/GB10 requires a CUDA 13.x development template for native sm_121 kernels.' >&2
  exit 1
fi
if [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -ge 8 ]]; then
  TORCH_VERSION=2.11.0; VISION_VERSION=0.26.0; CUDA_WHEEL=cu128; MIN_DRIVER=570
elif [[ "$CUDA_MAJOR" -eq 13 ]]; then
  TORCH_VERSION=2.13.0; VISION_VERSION=0.28.0; CUDA_WHEEL=cu130; MIN_DRIVER=580
else
  echo "CUDA $CUDA_VERSION is unsupported: use CUDA 12.8+ in the 12.x or 13.x family; other major versions need matching PyTorch builds." >&2
  exit 1
fi
[[ "${DRIVER%%.*}" -ge "$MIN_DRIVER" ]] || { echo "CUDA $CUDA_VERSION requires a compatible R$MIN_DRIVER-or-newer host driver." >&2; exit 1; }
# Both extension builders must use the nvcc selected above, including symlinks.
export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
TORCH_INDEX="https://download.pytorch.org/whl/$CUDA_WHEEL"

# Install basic tools automatically on Ubuntu/Debian Vast containers.
MISSING=()
for tool in git curl ffmpeg ffprobe python3 flock mountpoint g++ make tmux; do
  command -v "$tool" >/dev/null || MISSING+=("$tool")
done
if ! command -v python3 >/dev/null || ! python3 -c 'import venv, ensurepip' >/dev/null 2>&1; then
  MISSING+=(python3-venv)
fi
[[ -s /etc/ssl/certs/ca-certificates.crt ]] || MISSING+=(ca-certificates)
if [[ ${#MISSING[@]} -gt 0 ]]; then
  command -v apt-get >/dev/null || { echo "Missing tools: ${MISSING[*]}. Use an Ubuntu/Debian template or install them first." >&2; exit 1; }
  APT=(env DEBIAN_FRONTEND=noninteractive apt-get)
  if [[ "$EUID" -ne 0 ]]; then
    command -v sudo >/dev/null && sudo -n true || { echo "Missing tools: ${MISSING[*]}. Run setup as root or with passwordless sudo." >&2; exit 1; }
    APT=(sudo -n env DEBIAN_FRONTEND=noninteractive apt-get)
  fi
  echo "[1/6] Installing system tools: ${MISSING[*]}"
  "${APT[@]}" update
  "${APT[@]}" install -y git curl ca-certificates ffmpeg tmux python3 python3-venv util-linux build-essential
else
  echo '[1/6] System tools ready.'
fi
source "$SOURCE_DIR/scripts/runtime_env.sh"
case "$H3_PROFILE" in
  primary|fp8|int8-encoder|fp8-int8-encoder) ;;
  *) echo "Unknown profile: $H3_PROFILE" >&2; exit 2 ;;
esac
# Resolve relative roots before saving them for subsequent service commands.
H3_ROOT="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$H3_ROOT")"
export H3_ROOT H3_PORT
PROFILE="$H3_PROFILE"
mkdir -p "$H3_ROOT"/{models,input,output,temp,user,logs,locks,jobs,private}
chmod 700 "$H3_ROOT/private"
exec 9>"$H3_ROOT/locks/setup.lock"
flock -n 9 || { echo 'Another setup process is running.' >&2; exit 1; }
exec 8>"$H3_ROOT/locks/server.lock"
flock -n 8 || { echo 'Stop ComfyUI before updating: bash scripts/vast.sh stop (or stop your supervisor service).' >&2; exit 1; }
exec > >(tee -a "$H3_ROOT/logs/setup.log" 8>&- 9>&-) 2>&1
trap 'echo "Setup failed at line $LINENO. See $H3_ROOT/logs/setup.log; fix the reported error and rerun bash setup.sh." >&2' ERR
echo "Runtime: $H3_ROOT | profile: $PROFILE | port: $H3_PORT"
# Fail before large dependency downloads when storage cannot hold the selected models.
if [[ "$DOWNLOAD" == true ]]; then
  python3 - "$SOURCE_DIR" "$H3_ROOT" "$PROFILE" <<'PY'
import shutil
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from h3_pipeline.config import model_files
root = Path(sys.argv[2])
remaining = sum(item["size"] for item in model_files(sys.argv[3]).values()
                if not (root / "models" / item["path"]).exists())
required = remaining + 25 * 1024**3
if shutil.disk_usage(root).free < required:
    raise SystemExit(f"Need at least {required / 1024**3:.1f} GiB free for remaining models, dependencies and outputs; choose a larger disk or --root.")
PY
fi

echo '[2/6] Preparing pinned ComfyUI and Python 3.11.'
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
# Native builders invoke cmake/ninja by name, including pip-installed binaries.
export PATH="$H3_ROOT/venv/bin:$PATH"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,11), "Use a fresh Python 3.11 venv"'
printf 'torch==%s\ntorchvision==%s\ncomfy-kitchen==%s\n' "$TORCH_VERSION" "$VISION_VERSION" "$KITCHEN_VERSION" > "$H3_ROOT/locks/torch-constraints.txt"
echo '[3/6] Installing Python dependencies and SageAttention (first build can take several minutes).'
"$PYTHON" -m pip install --upgrade pip wheel setuptools
echo "CUDA toolkit $CUDA_VERSION | PyTorch $TORCH_VERSION ($CUDA_WHEEL) | driver $DRIVER"
"$PYTHON" -m pip install --only-binary=:all: "torch==$TORCH_VERSION" "torchvision==$VISION_VERSION" --index-url "$TORCH_INDEX"
# Fail before extension builds if pip selected a CPU wheel or a wheel that cannot
# execute on this GPU. This also rejects CUDA 12 on GB10 and emulated userspace.
"$PYTHON" - "$SOURCE_DIR" "$MACHINE" "$DRIVER" <<'PY'
import sys, torch
sys.path.insert(0, sys.argv[1])
from h3_pipeline.hardware import device_info, is_gb10
from h3_pipeline.preflight import cuda_environment_errors
if not torch.cuda.is_available():
    raise SystemExit("Installed PyTorch cannot access CUDA; select a native CUDA-enabled template/wheel")
hardware = device_info(torch)
errors = cuda_environment_errors(torch.version.cuda, sys.argv[3], machine=sys.argv[2], capability=hardware["capability"])
if is_gb10(hardware["gpu_model"], hardware["capability"], sys.argv[2]) and sys.argv[2] not in {"aarch64", "arm64"}:
    errors.append("GB10 requires native ARM64/aarch64 userspace")
if errors:
    raise SystemExit("; ".join(errors))
(torch.ones((16, 16), device="cuda") @ torch.ones((16, 16), device="cuda")).sum().item()
print("Verified CUDA wheel:", torch.__version__, hardware)
PY
# A source revision alone cannot identify a reusable CUDA extension. Include its
# ABI, toolkit, architectures and our cuBLAS compatibility patch in the build key.
"$PYTHON" - "$CUDA_VERSION" "$CUDA_HOME" "$KITCHEN_REVISION" "$SAGE_REVISION" "$SOURCE_DIR" "$MACHINE" > "$H3_ROOT/locks/cuda-build.pending.json" <<'PY'
import hashlib, json, os, platform, sys, sysconfig, torch
from pathlib import Path
print(json.dumps({"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                  "toolkit": sys.argv[1], "cuda_home": sys.argv[2],
                  "kitchen_revision": sys.argv[3], "sage_revision": sys.argv[4],
                  "machine": sys.argv[6], "python": platform.python_version(), "soabi": sysconfig.get_config_var("SOABI"),
                  "compatibility_patch": {str(p.relative_to(sys.argv[5])): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [Path(sys.argv[5]) / "scripts/build_kitchen.sh", *sorted((Path(sys.argv[5]) / "scripts/patches").glob("*.patch"))]},
                  "gpu_capabilities": [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())],
                  "arch_override": os.environ.get("TORCH_CUDA_ARCH_LIST", ""),
                  "kitchen_arch_override": os.environ.get("COMFY_CUDA_ARCHS", "")}, sort_keys=True))
PY
REUSE_CUDA=false
if cmp -s "$H3_ROOT/locks/cuda-build.pending.json" "$H3_ROOT/locks/cuda-build.json"; then
  REUSE_CUDA=true
fi
if [[ "$REUSE_CUDA" == true && -s "$H3_ROOT/locks/requirements-freeze.txt" ]]; then
  "$PYTHON" -m pip install -c "$H3_ROOT/locks/torch-constraints.txt" -r "$H3_ROOT/locks/requirements-freeze.txt" --extra-index-url "$TORCH_INDEX"
elif [[ -s "$H3_ROOT/locks/requirements-freeze.txt" ]]; then
  echo 'CUDA build environment changed; resolving dependencies again instead of replaying the previous lock.'
fi
# An interrupted migration must never leave the previous environment marked as
# reusable: it may already contain extensions from the unsuccessful new build.
rm -f "$H3_ROOT/locks/cuda-build.json"
PACKAGE="$SOURCE_DIR"
if [[ "$H3_INSTALL_POSE" == 1 ]]; then
  PACKAGE="$SOURCE_DIR[dance-quality]"
else
  echo 'Optional MediaPipe CPU pose screening skipped (H3_INSTALL_POSE=0); generation and visual reviews remain available.'
fi
"$PYTHON" -m pip install -c "$H3_ROOT/locks/torch-constraints.txt" -r "$H3_ROOT/ComfyUI/requirements.txt" -e "$PACKAGE"
# SageAttention 2.2.0 builds against the pinned Torch/CUDA environment.
# A runtime-only CUDA image lacks nvcc: select a CUDA 12.8+ devel image instead.
"$PYTHON" -m pip install ninja==1.11.1.4 packaging
if [[ "$CUDA_MAJOR" -eq 12 || "$ARM64" == true ]]; then
  if [[ "$REUSE_CUDA" != true ]]; then
    bash "$SOURCE_DIR/scripts/build_kitchen.sh" "$PYTHON" "$H3_ROOT" "$KITCHEN_REVISION" "$CUDA_MAJOR"
  fi
elif [[ "$REUSE_CUDA" != true ]]; then
  "$PYTHON" -m pip install --no-deps --force-reinstall "comfy-kitchen==$KITCHEN_VERSION"
fi
# Reuse a matching source build on reruns. Kernel preflight still runs every time.
if [[ "$REUSE_CUDA" != true ]] || ! "$PYTHON" - "$SAGE_REVISION" <<'PY'
import sys
from h3_pipeline.attention import metadata
actual = metadata()
sys.exit(0 if actual["sageattention_version"] == "2.2.0" and actual["sageattention_revision"] == sys.argv[1] else 1)
PY
then
  MAX_JOBS="${MAX_JOBS:-4}" EXT_PARALLEL="${EXT_PARALLEL:-2}" \
  "$PYTHON" -m pip install -c "$H3_ROOT/locks/torch-constraints.txt" \
  "sageattention @ git+https://github.com/thu-ml/SageAttention.git@$SAGE_REVISION" --no-build-isolation --no-deps --no-cache-dir --force-reinstall
fi
"$PYTHON" -m pip check
"$PYTHON" -c 'import pathlib, sys, h3_pipeline; actual = pathlib.Path(h3_pipeline.__file__).resolve(); expected = pathlib.Path(sys.argv[1]).resolve() / "h3_pipeline/__init__.py"; assert actual == expected, f"Wrong source checkout: {actual}"; print(f"Verified source: {actual}")' "$SOURCE_DIR"

for name in h3_lowvram h3_native_adapters; do
  target="$H3_ROOT/ComfyUI/custom_nodes/$name"
  if [[ -e "$target" || -L "$target" ]]; then
    [[ -L "$target" && "$(readlink "$target")" == "$SOURCE_DIR/custom_nodes/$name" ]] || { echo "Refusing to replace $target" >&2; exit 1; }
  else
    ln -s "$SOURCE_DIR/custom_nodes/$name" "$target"
  fi
done
echo '[4/6] Verifying CUDA kernels and exporting browser workflows.'
"$PYTHON" -m h3_pipeline --root "$H3_ROOT" preflight --profile "$PROFILE" --kernels --output "$H3_ROOT/locks/preflight.json"
# Source and the CUDA extension are installed separately above. In particular,
# Sage must never be rebuilt by pip's default isolated freeze replay.
"$PYTHON" -m pip freeze --exclude-editable --exclude sageattention --exclude comfy-kitchen > "$H3_ROOT/locks/requirements-freeze.txt"
mv "$H3_ROOT/locks/cuda-build.pending.json" "$H3_ROOT/locks/cuda-build.json"
"$PYTHON" --version > "$H3_ROOT/locks/python-version.txt"
"$PYTHON" -m h3_pipeline export-workflows --directory "$H3_ROOT/user/default/workflows"
if [[ "$DOWNLOAD" == true ]]; then
  echo '[5/6] Downloading/verifying models (completed downloads are reused).'
  "$PYTHON" -m h3_pipeline --root "$H3_ROOT" download-models --profile "$PROFILE"
else
  echo '[5/6] Model download skipped.'
fi
# Save only non-secret settings, atomically, once installation has succeeded.
{
  printf '# Generated by setup.sh. Environment overrides take precedence in helper scripts.\n'
  printf 'export H3_ROOT=%q\nexport H3_PROFILE=%q\nexport H3_PORT=%q\n' "$H3_ROOT" "$H3_PROFILE" "$H3_PORT"
  printf 'export H3_SERVER_URL=%q\n' "http://127.0.0.1:$H3_PORT"
  printf 'export H3_MEMORY_MODE=%q\n' "$H3_MEMORY_MODE"
  if [[ -n "${H3_RESERVE_VRAM:-}" ]]; then
    printf 'export H3_RESERVE_VRAM=%q\n' "$H3_RESERVE_VRAM"
  fi
} > "$SOURCE_DIR/.h3-vast.env.tmp"
mv "$SOURCE_DIR/.h3-vast.env.tmp" "$SOURCE_DIR/.h3-vast.env"
flock -u 8
exec 8>&-
if [[ "$START" == true ]]; then
  echo '[6/6] Starting ComfyUI in the background.'
  # Do not pass the setup lock or tee's output pipe to the detached server.
  bash "$SOURCE_DIR/scripts/vast.sh" start 9>&-
else
  echo '[6/6] Startup skipped. Later: bash scripts/vast.sh start'
fi
echo 'Setup complete. Settings saved in .h3-vast.env.'
echo 'Manage ComfyUI: bash scripts/vast.sh start|stop|restart|status|logs'
echo "Setup log: $H3_ROOT/logs/setup.log"
