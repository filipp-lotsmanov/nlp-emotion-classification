#!/usr/bin/env bash
# Detect hardware, pick a free GPU, launch DeBERTa emotion training, verify it.
#
#   ./scripts/train_emotion_en.sh --smoke           # ~2 min plumbing check
#   ./scripts/train_emotion_en.sh                   # the real run, ~3-4 h
#   ./scripts/train_emotion_en.sh --tmux            # detach, survives SSH drop
#   ./scripts/train_emotion_en.sh --class-weights balanced --tmux \
#       --output "$VEA_MODELS_DIR/emotion-en-deberta-balanced"
#
# Note the --output on that last one. Without it every run writes to the
# checkpoint the pipeline loads; the script now refuses rather than
# overwriting one, and --force is the way to say you meant it.
#
# Everything after a recognised flag is forwarded to
# training/emotion_en_deberta/train.py, so any of its options work here too.
#
# No --corpus is needed: corpora/super_emotion_clean.csv.gz is committed, and
# the trainer validates it against docs/dataset_build.md before touching a GPU.
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
FORCE=0
# deberta-v3-base at batch 16 with dynamic padding is modest; 12 GiB is ample
# and leaves the big cards free for the VA regressor.
MIN_FREE_MIB=12000
OUTPUT=""
CORPUS=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke) SMOKE=1; shift ;;
        --tmux) USE_TMUX=1; shift ;;
        --min-free) MIN_FREE_MIB="${2:?--min-free needs MiB}"; shift 2 ;;
        --corpus) CORPUS="${2:?--corpus needs a value}"; shift 2 ;;
        --output) OUTPUT="${2:?--output needs a path}"; shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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
    # Synthetic templates and a small encoder: proves gradients flow, the label
    # order survives the round trip through id2label, and the saved checkpoint
    # passes the contract check. It says nothing about emotion quality.
    CORPUS="${CORPUS:-smoke}"
    OUTPUT="${OUTPUT:-${MODELS_DIR}/emotion-en-smoke}"
    EXTRA_ARGS+=(
        --base-model "${EMOTION_SMOKE_MODEL:-distilroberta-base}"
        --epochs 3
        --batch-size 16
        --max-length 128
        --num-workers 2
    )
    MIN_FREE_MIB=4000
else
    OUTPUT="${OUTPUT:-${MODELS_DIR}/emotion-en-deberta}"
fi

# ---------------------------------------------------------------------------
# Never silently replace a checkpoint the pipeline is loading
# ---------------------------------------------------------------------------
# The default output is the directory MODEL_REGISTRY resolves
# `emotion-en-deberta` to. Any experiment run without --output therefore
# overwrites the model in production use - and an experiment is, by
# definition, not yet known to be better than what it replaces. Training it
# takes hours; noticing afterwards costs those hours twice.
if [[ -f "${OUTPUT}/config.json" && $FORCE -eq 0 ]]; then
    cat >&2 <<EOF
error: ${OUTPUT} already holds a checkpoint.

  Training would replace it, and the pipeline loads that directory.

  For an experiment, write it somewhere else and compare:
    --output "${MODELS_DIR}/emotion-en-deberta-<variant>"

  To deliberately retrain in place, pass --force.
EOF
    exit 3
fi

# ---------------------------------------------------------------------------
# Corpus, before the GPU is claimed
# ---------------------------------------------------------------------------
if [[ "${CORPUS:-}" != "smoke" ]]; then
    DEFAULT_CORPUS="${REPO_ROOT}/corpora/super_emotion_clean.csv.gz"
    if [[ -z "$CORPUS" && ! -f "$DEFAULT_CORPUS" ]]; then
        echo "error: $DEFAULT_CORPUS is missing." >&2
        echo "       It is committed to the repository; fetch it, or pass --corpus." >&2
        exit 2
    fi
    if [[ -n "$CORPUS" && ! -f "$CORPUS" ]]; then
        echo "error: --corpus $CORPUS does not exist." >&2
        exit 2
    fi
fi

# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------
if gpu="$("${REPO_ROOT}/scripts/pick_free_gpu.sh" --min-free "$MIN_FREE_MIB")"; then
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "using physical GPU $gpu (exposed to torch as cuda:0)"
else
    echo "error: no GPU with ${MIN_FREE_MIB} MiB free." >&2
    echo "       419,180 rows x 3 epochs on CPU is days, not hours." >&2
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
LOG_FILE="${VEA_TRAIN_LOG:-${LOG_DIR}/emotion_en_train_${STAMP}.log}"

CMD=(uv run python training/emotion_en_deberta/train.py --output "$OUTPUT")
[[ -n "$CORPUS" ]] && CMD+=(--corpus "$CORPUS")
CMD+=("${EXTRA_ARGS[@]}")

echo
echo "command: ${CMD[*]}"
echo "log:     $LOG_FILE"
echo

run_and_verify() {
    # pipefail is set, so a training failure propagates through tee.
    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
    echo
    echo "=== contract verification ==="
    # A smoke checkpoint trains for seconds on a few hundred synthetic templates,
    # so its softmax sits just above uniform (1/7 = 0.143) and the 0.25 stage-8
    # confidence floor is unreachable no matter how correct the predictions are.
    # Gate on the structural contract there and report the rest as advisory; a
    # real run gates on everything.
    verify_args=(emotion-en "$OUTPUT")
    (( SMOKE )) && verify_args+=(--structural-only)
    uv run python training/verify_checkpoint.py "${verify_args[@]}" 2>&1 | tee -a "$LOG_FILE"
}

if (( USE_TMUX )); then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "error: --tmux requested but tmux is not installed (apt-get install -y tmux)." >&2
        exit 1
    fi
    SESSION="emotion_en_train_${STAMP}"
    # Re-exec this script inside tmux without --tmux, so the detached session
    # runs the identical path including verification.
    ARGS=(--output "$OUTPUT" --min-free "$MIN_FREE_MIB")
    [[ -n "$CORPUS" ]] && ARGS+=(--corpus "$CORPUS")
    ARGS+=("${EXTRA_ARGS[@]}")
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
if (( SMOKE )); then
    echo
    echo "That was the smoke run - synthetic templates, so the F1 means only that the"
    echo "training loop and the saved contract are sound. Now run it without --smoke."
else
    echo
    echo "Two things to do before reporting anything from this checkpoint:"
    echo "  1. MODEL_REGISTRY expects the directory name 'emotion-en-deberta' under"
    echo "     ${MODELS_DIR}. Confirm with: uv run vea models"
    echo "  2. src/vea/pipeline.py stage_7b_english_emotion still names"
    echo "     'emotion-en-emoberta' in its second slot. Switch it to"
    echo "     'emotion-en-deberta' and re-run any video you intend to report."
fi
