"""Compare finished training runs from the `train_config.json` beside each one.

    uv run python training/emotion_en_deberta/compare_runs.py \
        "$VEA_MODELS_DIR/emotion-en-deberta" \
        "$VEA_MODELS_DIR/emotion-en-deberta-balanced"

Reads only what the trainer already wrote, so it needs no GPU, no corpus and
no torch - the point is that a comparison should never require re-running
either side of it.

Macro F1 leads, and per-class precision and recall are printed together,
because the class-weighting question is precisely a precision/recall trade:
`balanced` buys recall on the rare classes by spending precision, and a single
macro number hides which way each class moved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CONFIG_NAME = "train_config.json"
#: Fixed order, so two runs always line up row for row.
CLASSES = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")
HEADLINE = (("macro_f1", "macro F1"), ("accuracy", "accuracy"), ("weighted_f1", "weighted F1"))


def load(directory: Path) -> dict:
    """The run record beside a checkpoint.

    Raises:
        SystemExit: with a usable message. A missing or half-written record is
            the normal case for a run that crashed, and a traceback about a
            KeyError would say less than the path does.
    """
    path = directory / CONFIG_NAME
    if not path.is_file():
        raise SystemExit(f"no {CONFIG_NAME} in {directory} - did that run finish?")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}") from exc
    if not config.get("metrics", {}).get("best_epoch_report"):
        raise SystemExit(f"{path} has no best_epoch_report; the run did not reach an evaluation")
    return config


def cell(value: float | None) -> str:
    """Right-aligned by the caller's column width; a missing class reads as -."""
    return f"{value:.4f}" if value is not None else "-"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoints", nargs="+", type=Path, help="checkpoint directories")
    args = parser.parse_args(argv)

    runs = [(path, load(path)) for path in args.checkpoints]
    names = [path.name for path, _ in runs]
    reports = [c["metrics"]["best_epoch_report"] for _, c in runs]

    # Widths from the data, not guessed. `emotion-en-deberta-balanced` is 27
    # characters and a fixed 16-wide column ran the headings together.
    label = max(22, max(len(name) for name in CLASSES) + 4)
    col = max(12, max(len(name) for name in names) + 2)
    rule = "-" * (label + col * len(runs))

    def row(title: str, cells: list[str], indent: int = 0) -> None:
        pad = " " * indent
        print(f"{pad}{title:<{label - indent}}" + "".join(f"{c:>{col}}" for c in cells))

    row("", names)
    row("class weights", [str(c.get("class_weights", "?")) for _, c in runs])
    row("epochs", [str(c.get("epochs", "?")) for _, c in runs])
    row("validation rows", [str(r.get("n", "?")) for r in reports])
    print(rule)

    for key, title in HEADLINE:
        row(title, [cell(r.get(key)) for r in reports])

    for metric in ("precision", "recall", "f1"):
        print(rule)
        print(metric)
        for name in CLASSES:
            values = [r.get("per_class", {}).get(name, {}).get(metric) for r in reports]
            # A class the model never predicts scores 0 and is the failure mode
            # that still looks like a successful run; it is printed, not hidden.
            row(name, [cell(v) for v in values], indent=2)

    if len(runs) == 2:
        first, second = (r["macro_f1"] for r in reports)
        print(rule)
        print(
            f"{'macro F1 delta':<{label}}{second - first:>+{col}.4f}   {names[1]} minus {names[0]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
