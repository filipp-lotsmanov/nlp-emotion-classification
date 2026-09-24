#!/usr/bin/env bash
# Detect hardware, pick a free GPU, launch VA training, verify the checkpoint.
#
#   ./scripts/train_va.sh --smoke                    # ~2 min plumbing check
#   ./scripts/train_va.sh --dataset data/emobank.csv --valence-col V --arousal-col A \
#                         --va-min 1 --va-max 5
#   ./scripts/train_va.sh --dataset data/va.csv --tmux    # detach, survives SSH drop
#
# Everything after a recognised flag is forwarded to
# training/va_regressor/train.py, so any of its options work here too.
#
# What this wraps, and why each step matters on the shared BUAS box:
#   - sources the persistent env (HF cache, VEA_* paths) if setup_server.sh made one
#   - picks the emptiest GPU at launch time and fails fast if none has room,
#     instead of OOMing after the model is half loaded
#   - logs to a timestamped file so a detached run is inspectable afterwards
#   - runs verify_checkpoint.py on success, because a checkpoint that trains
#     cleanly can still violate the pipeline's inference contract

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SMOKE=0
USE_TMUX=0
MIN_FREE_MIB=20000
OUTPUT=""
DATASET=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke) SMOKE=1; shift ;;
        --tmux) USE_TMUX=1; shift ;;
        --min-free) MIN_FREE_MIB="${2:?--min-free needs MiB}"; shift 2 ;;
        --dataset) DATASET="${2:?--dataset needs a value}"; shift 2 ;;
        --output) OUTPUT="${2:?--output needs a path}"; shift 2 ;;
        -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
for candidate in /workspace/vea-env.sh "${REPO_ROOT}/../vea-env.sh"; do
    if [[ -f "$candidate" ]]; then
        # shellcheck disable=SC1090
        source "$candidate"
        echo "sourced $candidate"
        break
    fi
done

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv not found. Run ./scripts/setup_server.sh first." >&2
    exit 1
fi

MODELS_DIR="${VEA_MODELS_DIR:-${REPO_ROOT}/models}"
LOG_DIR="${VEA_DATA_DIR:-${REPO_ROOT}/data}/logs"
mkdir -p "$MODELS_DIR" "$LOG_DIR"

# ---------------------------------------------------------------------------
# Run shape: smoke check versus the real thing
# ---------------------------------------------------------------------------
if (( SMOKE )); then
    # Synthetic data and a tiny encoder: proves gradients flow, the sigmoid
    # objective behaves, and the saved checkpoint passes the contract check.
    # It says nothing about VA quality.
    DATASET="${DATASET:-smoke}"
    OUTPUT="${OUTPUT:-${MODELS_DIR}/va-smoke}"
    EXTRA_ARGS+=(
        --base-model "${VA_SMOKE_MODEL:-xlm-roberta-base}"
        --epochs 2
        --batch-size 16
        --lang-col lang
        --num-workers 2
    )
    MIN_FREE_MIB=4000
else
    if [[ -z "$DATASET" ]]; then
        echo "error: --dataset is required (a corpus path, 'hf:<id>', or use --smoke)." >&2
        echo "       See training/README.md for the corpus decision and the scale mapping." >&2
        exit 2
    fi
    OUTPUT="${OUTPUT:-${MODELS_DIR}/xlmroberta-large-va}"
fi

# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------
if gpu="$("${REPO_ROOT}/scripts/pick_free_gpu.sh" --min-free "$MIN_FREE_MIB")"; then
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "using physical GPU $gpu (exposed to torch as cuda:0)"
else
    echo "error: no GPU with ${MIN_FREE_MIB} MiB free. Training XLM-R-large on CPU is not practical." >&2
    echo "       Check 'scripts/pick_free_gpu.sh --list', or pass --min-free to lower the floor." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
echo "syncing dependencies (train extra)"
uv sync --extra train --quiet

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
STAMP="$(date +%Y%m%d_%H%M%S)"
# VEA_TRAIN_LOG is set by the --tmux branch below when it re-execs this
# script inside the session. Without it the inner run computed a fresh
# timestamp and wrote to a different file than the one the outer run had
# just told the user to tail, so `follow:` pointed at a path that never
# existed.
LOG_FILE="${VEA_TRAIN_LOG:-${LOG_DIR}/va_train_${STAMP}.log}"

CMD=(uv run python training/va_regressor/train.py
     --dataset "$DATASET" --output "$OUTPUT" "${EXTRA_ARGS[@]}")

echo
echo "command: ${CMD[*]}"
echo "log:     $LOG_FILE"
echo

run_and_verify() {
    # pipefail is set, so a training failure propagates through tee.
    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
    echo
    echo "=== contract verification ==="
    # A smoke checkpoint trains for seconds on synthetic templates, so the
    # behavioural checks (prediction spread, probe ordering) will fail on it
    # legitimately. Gate on the structural contract there and report the rest as
    # advisory; a real run gates on everything.
    verify_args=(va "$OUTPUT")
    (( SMOKE )) && verify_args+=(--structural-only)
    uv run python training/verify_checkpoint.py "${verify_args[@]}" 2>&1 | tee -a "$LOG_FILE"
}

if (( USE_TMUX )); then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "error: --tmux requested but tmux is not installed (apt-get install -y tmux)." >&2
        exit 1
    fi
    SESSION="va_train_${STAMP}"
    # Re-exec this script inside tmux without --tmux, so the detached session
    # runs the identical path including verification.
    ARGS=(--dataset "$DATASET" --output "$OUTPUT" --min-free "$MIN_FREE_MIB" "${EXTRA_ARGS[@]}")
    (( SMOKE )) && ARGS+=(--smoke)
    tmux new-session -d -s "$SESSION" \
        "export VEA_TRAIN_LOG=$(printf '%q' "$LOG_FILE") && cd $(printf '%q' "$REPO_ROOT") && $(printf '%q' "$0") $(printf '%q ' "${ARGS[@]}")"
    echo "started detached tmux session '$SESSION'"
    echo "  attach:  tmux attach -t $SESSION"
    echo "  follow:  tail -f $LOG_FILE"
    exit 0
fi

run_and_verify

echo
echo "checkpoint: $OUTPUT"
echo "metrics:    ${OUTPUT}/train_config.json  (see .metrics)"
echo "log:        $LOG_FILE"
if [[ "$OUTPUT" == *va-smoke* ]]; then
    echo
    echo "That was the smoke run - synthetic data, so the correlations mean only that"
    echo "the training loop and the saved contract are sound. Now run it for real with"
    echo "--dataset pointing at a VA corpus."
else
    echo
    echo "If verification passed, register it: VEA_MODELS_DIR is ${MODELS_DIR}, and"
    echo "MODEL_REGISTRY expects the directory name 'xlmroberta-large-va'."
    echo "Then: uv run vea models"
fi
