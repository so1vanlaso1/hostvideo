#!/usr/bin/env bash
# Build native CUDA kernels for CUDA 12 or ARM64/GB10 with CUDA 13.
set -euo pipefail
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$1"
ROOT="$2"
REVISION="$3"
CUDA_MAJOR="${4:-12}"
case "$CUDA_MAJOR" in
  12) PATCH=comfy-kitchen-cu12.patch ;;
  13) PATCH=comfy-kitchen-build-jobs.patch ;;
  *) echo "Unsupported Comfy Kitchen toolkit major: $CUDA_MAJOR" >&2; exit 1 ;;
esac
mkdir -p "$ROOT/temp"
BUILD_DIR="$(mktemp -d "$ROOT/temp/kitchen-build.XXXXXX")"
trap 'rm -rf -- "$BUILD_DIR"' EXIT
"$PYTHON" -m pip install 'cmake>=3.26' 'nanobind>=2.0' wheel
git clone --depth 1 --no-checkout https://github.com/Comfy-Org/comfy-kitchen.git "$BUILD_DIR/source"
git -C "$BUILD_DIR/source" fetch --depth 1 origin "$REVISION"
git -C "$BUILD_DIR/source" checkout --detach "$REVISION"
git -C "$BUILD_DIR/source" submodule update --init --depth 1
git -C "$BUILD_DIR/source" apply --check "$SOURCE_DIR/scripts/patches/$PATCH"
git -C "$BUILD_DIR/source" apply "$SOURCE_DIR/scripts/patches/$PATCH"
# CUDA 12.8 does not understand the default 100f/120f family targets. Build
# native cubins for the visible devices, just as SageAttention does.
ARCHS="${COMFY_CUDA_ARCHS:-$("$PYTHON" -c 'import torch; print(";".join(sorted({f"{a}{b}" + ("a" if a >= 9 else "") for a,b in (torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count()))})))')}"
[[ -n "$ARCHS" ]] || { echo 'No CUDA GPU visible for the Comfy Kitchen source build.' >&2; exit 1; }
COMFY_CUDA_ARCHS="$ARCHS" MAX_JOBS="${MAX_JOBS:-4}" COMFY_KITCHEN_BUILD_NO_HIP=1 \
  "$PYTHON" -m pip install --no-build-isolation --no-deps --no-cache-dir --force-reinstall "$BUILD_DIR/source"
