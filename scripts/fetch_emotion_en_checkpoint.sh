#!/usr/bin/env bash
# Fetch the English emotion classifier for stage 7B.
#
#   ./scripts/fetch_emotion_en_checkpoint.sh
#   ./scripts/fetch_emotion_en_checkpoint.sh --balanced
#   ./scripts/fetch_emotion_en_checkpoint.sh --into /some/other/dir
#
# DeBERTa-v3-base fine-tuned on the cleaned super-emotion corpus by
# training/emotion_en_deberta/train.py, published as a release asset so that a
# user without a GPU does not have to reproduce a multi-hour run before the
# pipeline can complete.
#
# CC BY-SA 4.0, inherited from cirimus/super-emotion. ShareAlike propagates to
# anything derived from these weights - read docs/LICENSING.md before
# redistributing them or a model trained on their output.
#
# --balanced fetches the class-weighted run instead. No stage loads it; it is
# published so training/emotion_en_deberta/compare_runs.py can reproduce the
# weighting comparison on a machine with no GPU and no corpus.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RELEASE="https://github.com/filipp-lotsmanov/nlp-emotion-classification/releases/download/weights-emotion-en-v1"

INTO=""
FORCE=0
VARIANT="default"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --balanced) VARIANT="balanced"; shift ;;
        --into) INTO="${2:?--into needs a path}"; shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# The archive's own top-level directory name is what MODEL_REGISTRY's `ref`
# resolves to. Renaming either without the other makes `vea models` report a
# checkpoint missing that is sitting on disk.
case "$VARIANT" in
    default)
        ASSET="emotion-en-deberta-v1.tar.gz"
        DIRNAME="emotion-en-deberta"
        TARBALL_SHA256="91a8888d113fd513ab6a862fab5a9d819ca3dcb79cc51574e7d88e4bf96694a2"
        ;;
    balanced)
        ASSET="emotion-en-deberta-balanced-v1.tar.gz"
        DIRNAME="emotion-en-deberta-balanced"
        TARBALL_SHA256="89906154d6d85ed36aba3c2aa2228e515ade6a4ce24d06f212f8ebbc1e2e136d"
        ;;
esac

# Fail before spending 590 MB of bandwidth rather than after. A placeholder
# left in by mistake would otherwise download the whole asset and then delete
# it on a digest mismatch, which reads as a corrupt release rather than an
# unfinished script.
if [[ "$TARBALL_SHA256" == REPLACE_WITH_* ]]; then
    echo "error: $(basename "$0") has no sha256 pinned for the $VARIANT checkpoint." >&2
    echo "  Take it from the ${ASSET}.sha256 asset on the release." >&2
    exit 2
fi

for candidate in /workspace/vea-env.sh "${REPO_ROOT}/../vea-env.sh"; do
    if [[ -f "$candidate" ]]; then
        # shellcheck disable=SC1090
        source "$candidate"
        break
    fi
done

MODELS_DIR="${VEA_MODELS_DIR:-${REPO_ROOT}/models}"
TARGET="${INTO:-${MODELS_DIR}/${DIRNAME}}"
mkdir -p "$MODELS_DIR"

# A locally trained checkpoint is hours of GPU time and is not recoverable from
# this release - it is a different run with different weights. Refuse rather
# than overwrite, the same way setup_docker.sh does, and exit 3 so a caller can
# tell "already there" apart from a real failure.
if [[ -e "$TARGET" && $FORCE -eq 0 ]]; then
    echo "error: $TARGET already exists." >&2
    echo "  re-run with --force to replace it." >&2
    exit 3
fi

echo "target: $TARGET"

TARBALL="${MODELS_DIR}/${ASSET}"
if [[ -s "$TARBALL" ]]; then
    echo "  have $ASSET"
else
    echo "  get  $ASSET  (~590 MB)"
    curl -fL --retry 3 --retry-delay 2 --progress-bar -o "${TARBALL}.part" "${RELEASE}/${ASSET}"
    mv "${TARBALL}.part" "$TARBALL"
fi

echo
echo "verifying digest"
actual="$(sha256sum "$TARBALL" | cut -d' ' -f1)"
if [[ "$actual" != "$TARBALL_SHA256" ]]; then
    echo "error: sha256 mismatch." >&2
    echo "  expected $TARBALL_SHA256" >&2
    echo "  got      $actual" >&2
    # Deleting matters: leaving it makes every later run a cache hit on a bad file.
    rm -f "$TARBALL"
    exit 1
fi
echo "  ok  $actual"

echo
echo "extracting"
# Into a staging directory first. --into may name a path that differs from the
# archive's own top-level directory, and a tar that fails halfway must not leave
# a half-written checkpoint where the loader will find it and try to use it.
STAGE="$(mktemp -d "${MODELS_DIR}/.fetch-XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
tar -xzf "$TARBALL" -C "$STAGE"
if [[ ! -d "${STAGE}/${DIRNAME}" ]]; then
    echo "error: ${ASSET} does not contain a ${DIRNAME}/ directory." >&2
    exit 1
fi
rm -rf "$TARGET"
mv "${STAGE}/${DIRNAME}" "$TARGET"
# The 590 MB archive has served its purpose; keeping it doubles the disk cost of
# a checkpoint for no benefit, since a re-run re-downloads and re-verifies.
rm -f "$TARBALL"

echo
echo "=== contract verification ==="
# Loading it is the only thing that proves the tokenizer, the config and the
# seven-way head agree. None of that raises if it is wrong, so it is checked.
uv run python training/verify_checkpoint.py emotion-en "$TARGET"

echo
echo "checkpoint: $TARGET"
if [[ "$VARIANT" == "default" ]]; then
    echo "Next: uv run vea models   (expect emotion-en-deberta to read [present])"
else
    echo "Next: uv run python training/emotion_en_deberta/compare_runs.py \\"
    echo "          \"${MODELS_DIR}/emotion-en-deberta\" \"$TARGET\""
fi
