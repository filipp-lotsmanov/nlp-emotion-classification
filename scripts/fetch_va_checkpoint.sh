#!/usr/bin/env bash
# Fetch the published valence-arousal checkpoint for stages 6A and 6B.
#
#   ./scripts/fetch_va_checkpoint.sh
#   ./scripts/fetch_va_checkpoint.sh --into /some/other/dir
#
# This is NOT a rebuild. It is the checkpoint the original coursework cited and
# never shipped: gmendes9/multilingual_va_prediction (Mendes & Martins, ECIR
# 2023, MIT), XLM-RoBERTa trained on 34 psycho-linguistic datasets across 100
# languages. It reads Russian without translation, which is why one checkpoint
# can serve both stage 6A (Russian) and 6B (English).
#
# The mirror is a public GitHub release on a group member's repository, which
# repackaged it as safetensors so it can be pinned by digest instead of
# unpickling a .bin out of a Google Drive folder. No credential is involved.
# The digest below is checked, and a mismatched download is deleted rather than
# left to poison the next run as a cache hit.
#
# Read docs/PROVENANCE.md section 11 before reporting anything derived from the
# arousal output. It separates this project's seven classes at AUC 0.5734,
# against valence's 0.8223, and stage 6A thresholds on arousal.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RELEASE="https://github.com/alex-krasnoshtanov/Emotion-Timeline/releases/download/weights-va-v1"
WEIGHTS_SHA256="f75773cb738a8f279832b5dd8b24209c5b1c3c71d4eb09d97b2981ecc9041332"

INTO=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --into) INTO="${2:?--into needs a path}"; shift 2 ;;
        -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

for candidate in /workspace/vea-env.sh "${REPO_ROOT}/../vea-env.sh"; do
    if [[ -f "$candidate" ]]; then
        # shellcheck disable=SC1090
        source "$candidate"
        break
    fi
done

MODELS_DIR="${VEA_MODELS_DIR:-${REPO_ROOT}/models}"
# The directory name is what MODEL_REGISTRY's `ref` resolves to. Changing it
# here without changing config.py makes `vea models` report it missing.
TARGET="${INTO:-${MODELS_DIR}/xlmroberta-base-va}"
mkdir -p "$TARGET"

echo "target: $TARGET"

# transformers needs the config and tokenizer beside the weights before
# from_pretrained will read the directory. They are separate assets on the same
# release, published as va-v1-<name>.
fetch() {
    local url="$1" dest="$2"
    if [[ -s "$dest" ]]; then
        echo "  have $(basename "$dest")"
        return
    fi
    echo "  get  $(basename "$dest")"
    curl -fsSL --retry 3 --retry-delay 2 -o "$dest.part" "$url"
    mv "$dest.part" "$dest"
}

fetch "${RELEASE}/emotion-timeline-va-v1.safetensors" "${TARGET}/model.safetensors"
for name in config.json sentencepiece.bpe.model special_tokens_map.json tokenizer_config.json; do
    fetch "${RELEASE}/va-v1-${name}" "${TARGET}/${name}"
done

echo
echo "verifying digest"
actual="$(sha256sum "${TARGET}/model.safetensors" | cut -d' ' -f1)"
if [[ "$actual" != "$WEIGHTS_SHA256" ]]; then
    echo "error: sha256 mismatch." >&2
    echo "  expected $WEIGHTS_SHA256" >&2
    echo "  got      $actual" >&2
    # Deleting matters: leaving it makes every later run a cache hit on a bad file.
    rm -f "${TARGET}/model.safetensors"
    exit 1
fi
echo "  ok  $actual"

echo
echo "naming the output heads"
# The published checkpoint ships id2label = {0: LABEL_0, 1: LABEL_1}: it never
# named its two outputs. The pipeline does not read id2label - intensity_ru.py
# indexes logits[0, 0] and logits[0, 1] directly - so the order is load-bearing
# and undocumented at the same time, which is the worst combination. Writing the
# names in makes it self-describing and lets verify_checkpoint.py assert it.
#
# Order confirmed against the checkpoint's own behaviour, not assumed: the
# "sitting calmly reading" probe scores 0.78-0.79 on column 0 and 0.08-0.10 on
# column 1. A calm positive sentence is high valence and low arousal, so column
# 0 is valence. Were the columns swapped, that probe would read low on column 0.
#
# This edits config.json only. The sha256 above covers model.safetensors, so the
# digest check is unaffected.
uv run python - "$TARGET" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1]) / "config.json"
config = json.loads(config_path.read_text(encoding="utf-8"))
labels = ["valence", "arousal"]

if config.get("id2label") == {"0": labels[0], "1": labels[1]}:
    print("  already named")
else:
    before = config.get("id2label")
    if len(before or {}) not in (0, 2):
        raise SystemExit(f"  refusing to relabel a {len(before)}-output head: {before}")
    config["id2label"] = dict(enumerate(labels))
    config["label2id"] = {name: i for i, name in enumerate(labels)}
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"  id2label {before} -> {{0: 'valence', 1: 'arousal'}}")
PY

echo
echo "=== contract verification ==="
# The checkpoint declares XLMRobertaForSequenceClassificationSig, a stock
# XLM-R with sigmoid folded into forward(). Loaded through the Auto class it
# comes back as the stock model returning raw logits, which is exactly what
# src/vea/stages/intensity_ru.py expects, since it applies torch.sigmoid itself.
# Index 0 is valence and index 1 is arousal. None of that raises if it is wrong,
# so it is checked.
uv run python training/verify_checkpoint.py va "$TARGET"

echo
echo "checkpoint: $TARGET"
echo "Next: uv run vea models   (expect va-xlmroberta-large to read [present])"
