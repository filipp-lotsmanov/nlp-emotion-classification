"""Where a trained emotion classifier is wrong, and what the errors have in common.

    uv run python training/emotion_en_deberta/error_analysis.py \
        "$VEA_MODELS_DIR/emotion-en-deberta" \
        "$VEA_MODELS_DIR/emotion-en-deberta-balanced" \
        --out-json docs/evaluation/error_analysis_report.json \
        --out-md   docs/evaluation/error_analysis.md

Why this exists: `docs/evaluation/error_analysis.md` reported 64,250 samples at
0.8995 accuracy with no script in the repository that produced them, and those
figures describe the model from before the rebuild. A report nobody can
regenerate is a claim, not a measurement.

The validation split is **reproduced, not re-drawn**. Every parameter comes
from the `train_config.json` the trainer wrote beside the weights, and the
split itself is `train.grouped_stratified_split` with the recorded seed. Any
other split would evaluate on rows the model was trained on and report a
number that means nothing.

Only the prediction step needs torch. Everything that turns predictions into a
report is a pure function of arrays, which is what lets the tests cover it in
the dev group.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import (  # noqa: E402
    LABELS,
    TrainConfig,
    classification_metrics,
    encode_labels,
    grouped_stratified_split,
    load_frame,
)

CONFIG_NAME = "train_config.json"
#: Stage 8 discards predictions below this, so the share under it is a
#: property of the timeline, not only of the table.
STAGE8_THRESHOLD = 0.25
#: Batches between progress lines. Often enough to show life, rarely
#: enough not to bury the report in a log.
PROGRESS_EVERY = 20
#: Thresholds the sweep reports. 0.25 is stage 8's, and is first so the
#: table opens with what the pipeline does today.
SWEEP = (0.25, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)


# --------------------------------------------------------------------------
# Pure analysis: arrays in, report out
# --------------------------------------------------------------------------


def confusion(true_ids: np.ndarray, pred_ids: np.ndarray) -> np.ndarray:
    """Rows are truth, columns are prediction."""
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    np.add.at(matrix, (np.asarray(true_ids), np.asarray(pred_ids)), 1)
    return matrix


def top_confusions(matrix: np.ndarray, per_class: int = 3) -> dict[str, list[dict]]:
    """For each true class, where its errors actually went.

    A confusion matrix is the answer to "how often is it wrong"; this is the
    answer to "wrong as what", which is the one that suggests a fix.
    """
    out: dict[str, list[dict]] = {}
    for i, name in enumerate(LABELS):
        errors = matrix[i].copy()
        errors[i] = 0
        total = int(matrix[i].sum())
        order = np.argsort(errors)[::-1][:per_class]
        out[name] = [
            {
                "predicted": LABELS[j],
                "count": int(errors[j]),
                "share_of_class": float(errors[j] / total) if total else 0.0,
            }
            for j in order
            if errors[j] > 0
        ]
    return out


def length_effects(texts: list[str], hit: np.ndarray) -> dict:
    """Are the mistakes shorter than the successes?

    The previous report found they were, and used it to argue for context
    windows. Recomputing it is the only way to know whether that still holds
    of a different model.
    """
    chars = np.array([len(t) for t in texts], dtype=np.float64)
    words = np.array([len(t.split()) for t in texts], dtype=np.float64)
    hit = np.asarray(hit, dtype=bool)
    if not hit.any() or not (~hit).any():
        return {"note": "one of the two groups is empty; no comparison possible"}
    return {
        "correct_mean_chars": float(chars[hit].mean()),
        "incorrect_mean_chars": float(chars[~hit].mean()),
        "correct_mean_words": float(words[hit].mean()),
        "incorrect_mean_words": float(words[~hit].mean()),
        "correct_median_words": float(np.median(words[hit])),
        "incorrect_median_words": float(np.median(words[~hit])),
    }


def confidence_effects(confidence: np.ndarray, hit: np.ndarray) -> dict:
    """The calibration evidence, including what stage 8 would throw away."""
    confidence = np.asarray(confidence, dtype=np.float64)
    hit = np.asarray(hit, dtype=bool)
    below = confidence < STAGE8_THRESHOLD
    return {
        "correct_mean": float(confidence[hit].mean()) if hit.any() else float("nan"),
        "incorrect_mean": float(confidence[~hit].mean()) if (~hit).any() else float("nan"),
        "below_stage8_threshold": float(below.mean()),
        "accuracy_above_threshold": float(hit[~below].mean()) if (~below).any() else float("nan"),
        # Of all the errors, the fraction confident enough that stage 8 keeps
        # them. A confident error is one the timeline draws.
        "share_of_errors_above_threshold": (
            float((~hit[~below]).sum() / (~hit).sum()) if (~hit).any() else 0.0
        ),
    }


def threshold_sweep(confidence: np.ndarray, hit: np.ndarray) -> list[dict]:
    """What each confidence gate would actually buy and cost.

    Stage 8 discards predictions below 0.25. Measured, that is 0.03% of them
    and 0.4% of the errors - a filter that looks like quality control and is
    not one. Choosing a different number is a trade between how much of the
    timeline survives and how wrong the surviving part is, and that trade is
    only visible as a curve.
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    hit = np.asarray(hit, dtype=bool)
    errors = int((~hit).sum())
    out = []
    for threshold in SWEEP:
        kept = confidence >= threshold
        out.append(
            {
                "threshold": float(threshold),
                "coverage": float(kept.mean()),
                "accuracy_kept": float(hit[kept].mean()) if kept.any() else float("nan"),
                # The two costs, side by side: a gate that removes errors also
                # removes correct answers, and only the ratio is interesting.
                "errors_removed": float((~hit & ~kept).sum() / errors) if errors else 0.0,
                "correct_removed": float((hit & ~kept).sum() / hit.sum()) if hit.any() else 0.0,
            }
        )
    return out


def build_report(
    texts: list[str],
    true_ids: np.ndarray,
    pred_ids: np.ndarray,
    confidence: np.ndarray,
    meta: dict,
) -> dict:
    """Everything the markdown renders, as plain JSON-able data."""
    hit = np.asarray(pred_ids) == np.asarray(true_ids)
    matrix = confusion(true_ids, pred_ids)
    report = classification_metrics(pred_ids, true_ids)
    return {
        **meta,
        "n": int(len(texts)),
        "errors": int((~hit).sum()),
        "error_rate": float((~hit).mean()) if len(hit) else 0.0,
        "metrics": report,
        "confusion": {
            "labels": list(LABELS),
            "matrix": matrix.tolist(),
        },
        "top_confusions": top_confusions(matrix),
        "length": length_effects(texts, hit),
        "confidence": confidence_effects(confidence, hit),
        "threshold_sweep": threshold_sweep(confidence, hit),
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def render_markdown(reports: list[dict]) -> str:
    """The report, written from the numbers rather than around them."""
    names = [r["checkpoint"] for r in reports]
    generated = datetime.now(UTC).strftime("%Y-%m-%d")

    out = [
        "# Error analysis: English emotion classification",
        "",
        f"Generated {generated} by `training/emotion_en_deberta/error_analysis.py`.",
        "Every figure below is recomputed from the checkpoints named here, on the",
        "validation split reproduced from each run's own `train_config.json`.",
        "",
        "Regenerate with:",
        "",
        "```bash",
        "uv run python training/emotion_en_deberta/error_analysis.py \\",
        *[f'    "$VEA_MODELS_DIR/{name}" \\' for name in names],
        "    --out-json docs/evaluation/error_analysis_report.json \\",
        "    --out-md   docs/evaluation/error_analysis.md",
        "```",
        "",
        "## Headline",
        "",
    ]
    if any(r.get("subsampled") for r in reports):
        out += [
            "> **Subsampled run — not a result.** `--limit` was used, so these",
            "> figures come from a random subset of the validation split and carry",
            "> its sampling error. Rare classes are the worst affected: Disgust is",
            "> under 1% of the corpus. Re-run without `--limit` before quoting",
            "> anything here.",
            "",
        ]
    out += [
        _table(
            ["", *names],
            [
                ["validation rows", *[f"{r['n']:,}" for r in reports]],
                ["errors", *[f"{r['errors']:,}" for r in reports]],
                ["accuracy", *[f"{r['metrics']['accuracy']:.4f}" for r in reports]],
                ["macro F1", *[f"{r['metrics']['macro_f1']:.4f}" for r in reports]],
                ["weighted F1", *[f"{r['metrics']['weighted_f1']:.4f}" for r in reports]],
            ],
        ),
        "",
        "Macro F1 leads. Accuracy on this corpus is largely decided by Joy and",
        "Sadness, so a model answering only those two already scores well.",
        "",
        "## Per class",
        "",
    ]

    for metric in ("precision", "recall", "f1"):
        out += [
            f"### {metric.capitalize()}",
            "",
            _table(
                ["class", *names],
                [
                    [
                        label,
                        *[f"{r['metrics']['per_class'][label][metric]:.4f}" for r in reports],
                    ]
                    for label in LABELS
                ],
            ),
            "",
        ]

    for report in reports:
        out += [
            f"## {report['checkpoint']}",
            "",
            f"Class weighting: `{report.get('class_weights', 'unknown')}`. "
            f"Error rate {report['error_rate']:.4f} over {report['n']:,} rows.",
            "",
            "### Where the errors go",
            "",
            "For each true class, the classes its mistakes were predicted as.",
            "",
        ]
        rows = []
        for label, confusions in report["top_confusions"].items():
            if not confusions:
                rows.append([label, "—", "—", "—"])
                continue
            for i, entry in enumerate(confusions):
                rows.append(
                    [
                        label if i == 0 else "",
                        entry["predicted"],
                        f"{entry['count']:,}",
                        f"{entry['share_of_class']:.3f}",
                    ]
                )
        out += [
            _table(["true class", "predicted as", "count", "share of class"], rows),
            "",
            "### Confidence",
            "",
            _table(
                ["", "value"],
                [
                    ["mean on correct", f"{report['confidence']['correct_mean']:.4f}"],
                    ["mean on incorrect", f"{report['confidence']['incorrect_mean']:.4f}"],
                    [
                        f"below stage 8's {STAGE8_THRESHOLD} threshold",
                        f"{report['confidence']['below_stage8_threshold']:.4f}",
                    ],
                    [
                        "accuracy among kept predictions",
                        f"{report['confidence']['accuracy_above_threshold']:.4f}",
                    ],
                    [
                        "errors that survive the threshold",
                        f"{report['confidence']['share_of_errors_above_threshold']:.4f}",
                    ],
                ],
            ),
            "",
            "Stage 8 drops predictions below the threshold, so the last two rows are",
            "properties of the timeline rather than of the table: a confident error is",
            "one the plot will draw.",
            "",
            "#### What a different gate would buy",
            "",
            _table(
                ["threshold", "coverage", "accuracy kept", "errors removed", "correct removed"],
                [
                    [
                        f"{row['threshold']:.2f}"
                        + ("  (stage 8)" if row["threshold"] == 0.25 else ""),
                        f"{row['coverage']:.4f}",
                        f"{row['accuracy_kept']:.4f}",
                        f"{row['errors_removed']:.4f}",
                        f"{row['correct_removed']:.4f}",
                    ]
                    for row in report.get("threshold_sweep", [])
                ],
            ),
            "",
            "Read the last two columns together. A gate is only worth raising while it",
            "removes errors faster than it removes correct answers; where those columns",
            "converge, the filter is just discarding timeline.",
            "",
            "### Length",
            "",
        ]
        length = report["length"]
        if "note" in length:
            out += [length["note"], ""]
        else:
            out += [
                _table(
                    ["", "correct", "incorrect"],
                    [
                        [
                            "mean characters",
                            f"{length['correct_mean_chars']:.2f}",
                            f"{length['incorrect_mean_chars']:.2f}",
                        ],
                        [
                            "mean words",
                            f"{length['correct_mean_words']:.2f}",
                            f"{length['incorrect_mean_words']:.2f}",
                        ],
                        [
                            "median words",
                            f"{length['correct_median_words']:.1f}",
                            f"{length['incorrect_median_words']:.1f}",
                        ],
                    ],
                ),
                "",
            ]

    out += [
        "## Reading this against the pipeline",
        "",
        "These figures are measured on the training corpus's own validation split,",
        "which is 97% Twitter text. The pipeline applies the model to translated",
        "Russian documentary speech. In-distribution accuracy here is an upper",
        "bound on what the timeline gets, not an estimate of it - see",
        "`docs/PROVENANCE.md` sections 13 and 14.",
        "",
    ]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# The part that needs a GPU
# --------------------------------------------------------------------------


def subsample(val_idx: np.ndarray, limit: int, seed: int) -> np.ndarray:
    """A random `limit` of the validation split, not its first `limit`.

    The corpus is ordered by source - the dair-ai block, then MELD, then
    Reddit - and the split preserves that order, so a head slice is one
    register rather than a sample of all of them. Measured on the real
    corpus, `val_idx[:2000]` is 25.9% Neutral against 3.2% in the full split:
    an eight-fold over-representation of the weakest class. It reported
    accuracy 0.7610 for a model measured at 0.9202, which reads as a broken
    evaluation and is really a broken subset.

    Seeded from the run's own seed, so a smoke check is reproducible.
    """
    if limit >= len(val_idx):
        return val_idx
    chosen = np.random.default_rng(seed).choice(val_idx, size=limit, replace=False)
    # Sorted so the evaluation order still follows the corpus, which keeps
    # tokenised batches roughly length-homogeneous.
    return np.sort(chosen)


#: Texts and labels per distinct corpus configuration. Two checkpoints trained
#: on the same corpus need the same frame, and loading and validating 419,180
#: gzipped rows is the slowest step in a run that is otherwise a minute of GPU
#: per model. Keyed rather than global so a comparison across two different
#: corpora still loads both.
_CORPUS_CACHE: dict[tuple, tuple[list[str], np.ndarray]] = {}


def corpus_for(cfg: TrainConfig) -> tuple[list[str], np.ndarray]:
    """The texts and encoded labels a run was trained on, loaded once."""
    key = (str(cfg.corpus), cfg.text_col, cfg.label_col, cfg.limit, cfg.seed)
    if key not in _CORPUS_CACHE:
        frame = load_frame(cfg)
        _CORPUS_CACHE[key] = (
            frame[cfg.text_col].astype(str).tolist(),
            encode_labels(frame[cfg.label_col]),
        )
    return _CORPUS_CACHE[key]


def config_from(checkpoint: Path) -> TrainConfig:
    """The run's own parameters, so the split can be reproduced exactly."""
    path = checkpoint / CONFIG_NAME
    if not path.is_file():
        raise SystemExit(f"no {CONFIG_NAME} in {checkpoint} - did that run finish?")
    stored = json.loads(path.read_text(encoding="utf-8"))
    fields = set(TrainConfig.__dataclass_fields__)
    # `metrics` and anything else the trainer added is dropped rather than
    # passed through, so a new recorded field cannot break this script.
    return TrainConfig(**{k: v for k, v in stored.items() if k in fields})


def predict(checkpoint: Path, texts: list[str], cfg: TrainConfig, batch_size: int):
    """Top-1 prediction and its softmax confidence, per row.

    Reports progress to stderr. Minutes of silence is indistinguishable from
    a hung process, and the first long run of this script was killed by a
    dropped SSH session with no way to tell from the log whether it had
    stopped or was still working.
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from train import EmotionDataset, make_collator, resolve_device

    device = resolve_device(None)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint)).to(device)
    model.eval()

    dataset = EmotionDataset(texts, np.zeros(len(texts), dtype=np.int64), tokenizer, cfg.max_length)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=make_collator(tokenizer)
    )

    total = len(dataset)
    print(f"{checkpoint.name}: {total:,} rows on {device}", file=sys.stderr, flush=True)

    preds, confidences = [], []
    done = 0
    started = time.time()
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            probabilities = torch.nn.functional.softmax(model(**batch).logits.float(), dim=-1)
            top = torch.max(probabilities, dim=-1)
            preds.append(top.indices.cpu().numpy())
            confidences.append(top.values.cpu().numpy())

            done += len(top.indices)
            if done % (batch_size * PROGRESS_EVERY) == 0 or done == total:
                rate = done / max(time.time() - started, 1e-9)
                remaining = (total - done) / rate if rate else 0
                print(
                    f"  {done:,}/{total:,} ({done / total:.0%})  "
                    f"{rate:,.0f} rows/s  ~{remaining:.0f}s left",
                    file=sys.stderr,
                    flush=True,
                )
    return np.concatenate(preds), np.concatenate(confidences)


def analyse(checkpoint: Path, batch_size: int, limit: int | None = None) -> dict:
    cfg = config_from(checkpoint)
    texts, label_ids = corpus_for(cfg)
    _, val_idx = grouped_stratified_split(texts, label_ids, cfg.val_fraction, cfg.seed)

    limited = limit is not None and limit < len(val_idx)
    if limited:
        val_idx = subsample(val_idx, limit, cfg.seed)

    val_texts = [texts[i] for i in val_idx]
    val_labels = label_ids[val_idx]
    pred_ids, confidence = predict(checkpoint, val_texts, cfg, batch_size)

    return build_report(
        val_texts,
        val_labels,
        pred_ids,
        confidence,
        {
            "checkpoint": checkpoint.name,
            "class_weights": cfg.class_weights,
            "seed": cfg.seed,
            "val_fraction": cfg.val_fraction,
            "corpus": cfg.corpus,
            "subsampled": limited,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--limit", type=int, help="evaluate only the first N validation rows, for a smoke check"
    )
    args = parser.parse_args(argv)

    reports = [analyse(path, args.batch_size, args.limit) for path in args.checkpoints]

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"wrote {args.out_json}")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_markdown(reports), encoding="utf-8")
        print(f"wrote {args.out_md}")
    if not (args.out_json or args.out_md):
        print(render_markdown(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
