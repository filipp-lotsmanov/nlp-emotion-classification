#!/usr/bin/env bash
# Give faster-whisper (stage 4) the CUDA 12 runtime it links against.
#
#   ./scripts/setup_ctranslate2_cuda.sh
#   ./scripts/setup_ctranslate2_cuda.sh --into /some/dir --no-env
#
# THE PROBLEM
#
# Stage 4 dies with:
#
#   RuntimeError: Library libcublas.so.12 is not found or cannot be loaded
#
# faster-whisper runs on CTranslate2, and CTranslate2 4.x wheels are built
# against **CUDA 12**: they need libcublas.so.12 and libcudnn.so.9. torch, on
# the other hand, now resolves to a CUDA 13 build on PyPI and brings
# `nvidia-cublas` 13.x with it, which provides libcublas.so.**13**. Same
# library, different soname, so the dynamic linker finds nothing it can use.
#
# Nothing is wrong with either package. They are simply built for different
# CUDA majors, and one process needs both - torch for stages 5B-7B, CTranslate2
# for stage 4.
#
# THE FIX
#
# Install the CUDA 12 cuBLAS and cuDNN wheels and put them on LD_LIBRARY_PATH.
# They coexist with the CUDA 13 ones because the sonames differ, so torch keeps
# using libcublas.so.13 while CTranslate2 finds libcublas.so.12.
#
# They are installed OUTSIDE the project venv on purpose: `uv sync` prunes
# anything not in uv.lock, so a venv install would vanish on the next sync and
# stage 4 would break again with no obvious cause.
#
# Cost: about 1.4 GB (cuBLAS 581 MB, cuDNN 770 MB).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

INTO=""
WRITE_ENV=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --into) INTO="${2:?--into needs a path}"; shift 2 ;;
        --no-env) WRITE_ENV=0; shift ;;
        -h|--help) sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

for candidate in /workspace/vea-env.sh "${REPO_ROOT}/../vea-env.sh"; do
    if [[ -f "$candidate" ]]; then
        # shellcheck disable=SC1090
        source "$candidate"
        ENV_FILE="$candidate"
        break
    fi
done

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv not found. Run ./scripts/setup_server.sh first." >&2
    exit 1
fi

# Default beside the models rather than inside the repo: it is a 1.4 GB
# environment artefact, not source, and it must survive a git clean.
TARGET="${INTO:-$(dirname "${VEA_MODELS_DIR:-${REPO_ROOT}/models}")/ct2-cuda12}"
mkdir -p "$TARGET"

echo "installing CUDA 12 runtime libraries into $TARGET"
uv pip install --quiet --target "$TARGET" nvidia-cublas-cu12 nvidia-cudnn-cu12

CUBLAS_DIR="${TARGET}/nvidia/cublas/lib"
CUDNN_DIR="${TARGET}/nvidia/cudnn/lib"

echo
echo "verifying"
missing=0
for lib in "${CUBLAS_DIR}/libcublas.so.12" "${CUDNN_DIR}/libcudnn.so.9"; do
    if [[ -f "$lib" ]]; then
        echo "  ok  $lib"
    else
        echo "  MISSING  $lib" >&2
        missing=1
    fi
done
if (( missing )); then
    echo "error: the wheels did not lay out as expected; inspect $TARGET" >&2
    exit 1
fi

LD_LINE="export LD_LIBRARY_PATH=\"${CUBLAS_DIR}:${CUDNN_DIR}:\${LD_LIBRARY_PATH:-}\""

if (( WRITE_ENV )) && [[ -n "${ENV_FILE:-}" ]]; then
    if grep -qF "$CUBLAS_DIR" "$ENV_FILE"; then
        echo
        echo "$ENV_FILE already sets LD_LIBRARY_PATH for these"
    else
        {
            echo
            echo "# CUDA 12 runtime for CTranslate2 (stage 4). See"
            echo "# scripts/setup_ctranslate2_cuda.sh for why this is separate from torch's CUDA 13."
            echo "$LD_LINE"
        } >> "$ENV_FILE"
        echo
        echo "appended LD_LIBRARY_PATH to $ENV_FILE"
    fi
fi

echo
echo "For this shell, run:"
echo "  $LD_LINE"
echo
echo "Then confirm CTranslate2 can reach a GPU:"
echo "  uv run python -c \"import ctranslate2; print(ctranslate2.get_cuda_device_count())\""
echo "A count of 1 or more means stage 4 will load. Zero means the libraries are"
echo "still not on the path for that process."
