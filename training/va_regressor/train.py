"""Train `va-xlmroberta-large`: valence-arousal regression for stages 6A/6B.

Produces a checkpoint that src/vea/stages/intensity_{ru,en}.py can load
unchanged. The contract those modules impose, read from their source:

    model = AutoModelForSequenceClassification.from_pretrained(path, trust_remote_code=True)
    predictions = torch.sigmoid(outputs.logits)
    valence = predictions[0, 0]
    arousal = predictions[0, 1]

Three consequences drive the design here:

1. `num_labels = 2`, index 0 valence, index 1 arousal. The pipeline does not
   read `id2label`, so a swapped order is undetectable at runtime - it produces
   a plausible timeline that is simply wrong. The order is written into the
   config and asserted by training/verify_checkpoint.py.
2. Inference applies **sigmoid**. So the loss is MSE between `sigmoid(logits)`
   and targets in [0, 1], not MSE on raw logits. Training raw regression and
   then squashing at inference pushes every prediction toward 0.5, which makes
   the video-adaptive median threshold in stage 6A meaningless - and raises no
   error anywhere.
3. The same checkpoint scores English in stage 6B, so the encoder stays
   multilingual (XLM-RoBERTa) and evaluation is reported per language.

Usage:

    # 1. Prove the loop runs, in ~1 minute, with no corpus and no download:
    python training/va_regressor/train.py --dataset smoke --base-model <small-model> \
        --output models/va-smoke --epochs 1

    # 2. The real run:
    python training/va_regressor/train.py \
        --dataset data/emobank.csv --text-col text \
        --valence-col V --arousal-col A --va-min 1 --va-max 5 \
        --output models/xlmroberta-large-va

    # 3. Confirm the checkpoint satisfies the pipeline contract:
    python training/verify_checkpoint.py va models/xlmroberta-large-va

Corpus choice is yours and is the biggest reproducibility gap in the project:
the archive recorded only a citation (Mendes & Martins, multilingual VA
prediction) and never which data or hyperparameters were used. Whatever you
pick, `--dataset`, `--va-min/--va-max` and the seed all land in
`train_config.json` next to the weights so the run is reconstructable.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger("va_train")

# Index order is load-bearing; see the module docstring.
LABELS = ["valence", "arousal"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Everything needed to reproduce a run. Serialised beside the weights."""

    dataset: str
    output: str
    base_model: str = "xlm-roberta-large"
    text_col: str = "text"
    valence_col: str = "valence"
    arousal_col: str = "arousal"
    lang_col: str | None = None
    # Source annotation scale, mapped linearly onto [0, 1]. EmoBank is 1-5,
    # ANEW-style corpora are 1-9, some are -1..1. Getting this wrong is
    # invisible downstream, so it is required to be explicit.
    va_min: float = 1.0
    va_max: float = 9.0
    learning_rate: float = 1e-5
    batch_size: int = 32
    eval_batch_size: int = 64
    epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    # Training pads dynamically to the longest item in the batch and truncates
    # here. VA corpora are sentence-level, so 128 wastes far less compute than
    # the 512 inference uses; position embeddings are untouched either way.
    max_length: int = 128
    val_fraction: float = 0.15
    seed: int = 42
    num_workers: int = 8
    bf16: bool = True
    limit: int | None = None
    device: str | None = None
    metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pure helpers - no torch, so they are unit-testable without a GPU stack
# ---------------------------------------------------------------------------


def normalize_targets(values: np.ndarray, src_min: float, src_max: float) -> np.ndarray:
    """Map an annotation scale linearly onto [0, 1].

    Values outside [src_min, src_max] are clipped rather than rescaled, so one
    mis-parsed row cannot shift the whole distribution.

    Raises:
        ValueError: src_max <= src_min, which would divide by zero or invert
            the scale silently.
    """
    if src_max <= src_min:
        raise ValueError(f"va_max ({src_max}) must be greater than va_min ({src_min})")
    scaled = (np.asarray(values, dtype=np.float64) - src_min) / (src_max - src_min)
    return np.clip(scaled, 0.0, 1.0)


def regression_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """Correlation first, error second.

    Pearson/Spearman lead because MSE looks acceptable for a model that has
    only learned the training mean, and stage 6A thresholds on the *ordering*
    of arousal within a video, not on absolute error. `pred_std` catches the
    collapsed head directly.
    """
    from scipy import stats

    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)

    out = {
        "mse": float(np.mean((pred - true) ** 2)),
        "mae": float(np.mean(np.abs(pred - true))),
        "pred_std": float(np.std(pred)),
        "true_std": float(np.std(true)),
    }
    # Correlation is undefined for a constant vector; report NaN rather than
    # letting scipy warn and return nan silently.
    if out["pred_std"] < 1e-8 or out["true_std"] < 1e-8:
        out["pearson"] = float("nan")
        out["spearman"] = float("nan")
    else:
        out["pearson"] = float(stats.pearsonr(pred, true).statistic)
        out["spearman"] = float(stats.spearmanr(pred, true).statistic)
    return out


def summarize(metrics: dict[str, dict[str, float]]) -> float:
    """Model-selection score: mean Pearson r over valence and arousal.

    NaN (a collapsed head) sorts below any real correlation.
    """
    values = [metrics[dim].get("pearson", float("nan")) for dim in LABELS]
    values = [v for v in values if not math.isnan(v)]
    return float(np.mean(values)) if values else -1.0


def build_smoke_frame(n_per_template: int = 60, seed: int = 0):
    """Synthetic bilingual VA data, for proving the loop end to end.

    Not a substitute for a corpus - the templates are trivially separable, so
    high correlation here means only that gradients flow, the sigmoid objective
    behaves, and the saved checkpoint satisfies the pipeline contract. Use it to
    debug plumbing before spending GPU hours.
    """
    import pandas as pd

    rng = random.Random(seed)
    # (template, language, valence, arousal) on the target 1-9 scale.
    templates = [
        ("This is wonderful news, I am thrilled {}", "en", 8.5, 8.0),
        ("Какая прекрасная новость, я в восторге {}", "ru", 8.5, 8.0),
        ("A quiet pleasant afternoon reading {}", "en", 7.0, 2.5),
        ("Тихий приятный день за книгой {}", "ru", 7.0, 2.5),
        ("I am furious about this outrage {}", "en", 1.5, 8.5),
        ("Я в ярости от этого безобразия {}", "ru", 1.5, 8.5),
        ("Everything feels hopeless and empty {}", "en", 1.5, 2.0),
        ("Всё кажется безнадёжным и пустым {}", "ru", 1.5, 2.0),
        ("The report was filed on Tuesday {}", "en", 5.0, 4.0),
        ("Отчёт был подан во вторник {}", "ru", 5.0, 4.0),
    ]
    rows = []
    for text, lang, valence, arousal in templates:
        for _ in range(n_per_template):
            filler = " ".join(
                rng.choice(["today", "again", "here", "now", "", "really"]) for _ in range(2)
            )
            rows.append(
                {
                    "text": text.format(filler).strip(),
                    "lang": lang,
                    "valence": valence + rng.uniform(-0.4, 0.4),
                    "arousal": arousal + rng.uniform(-0.4, 0.4),
                }
            )
    rng.shuffle(rows)
    return pd.DataFrame(rows)


def load_va_frame(cfg: TrainConfig):
    """Load the corpus into a DataFrame with text/valence/arousal columns.

    Accepts, in order of how it is detected:
        "smoke"              synthetic data (see build_smoke_frame)
        "hf:<dataset_id>"    a Hugging Face dataset, train split
        <path>               .csv / .tsv / .parquet / .json / .jsonl
    """
    import pandas as pd

    spec = cfg.dataset
    if spec == "smoke":
        frame = build_smoke_frame(seed=cfg.seed)
        logger.info("using synthetic smoke data: %d rows", len(frame))
    elif spec.startswith("hf:"):
        from datasets import load_dataset

        dataset_id = spec[3:]
        logger.info("loading Hugging Face dataset %s", dataset_id)
        frame = load_dataset(dataset_id, split="train").to_pandas()
    else:
        path = Path(spec)
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. Pass a corpus file, 'hf:<dataset_id>', or 'smoke'."
            )
        suffix = path.suffix.lower()
        if suffix == ".csv":
            frame = pd.read_csv(path)
        elif suffix in {".tsv", ".tab"}:
            frame = pd.read_csv(path, sep="\t")
        elif suffix == ".parquet":
            frame = pd.read_parquet(path)
        elif suffix in {".json", ".jsonl"}:
            frame = pd.read_json(path, lines=suffix == ".jsonl")
        else:
            raise ValueError(f"unsupported corpus format: {suffix}")
        logger.info("loaded %d rows from %s", len(frame), path)

    required = [cfg.text_col, cfg.valence_col, cfg.arousal_col]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(
            f"corpus is missing column(s) {missing}. Available: {list(frame.columns)}. "
            "Set --text-col / --valence-col / --arousal-col to match."
        )

    frame = frame.dropna(subset=required).copy()
    frame[cfg.text_col] = frame[cfg.text_col].astype(str).str.strip()
    frame = frame[frame[cfg.text_col].str.len() > 0]
    if cfg.limit:
        frame = frame.head(cfg.limit)
    if len(frame) < 50:
        raise ValueError(f"only {len(frame)} usable rows; too few to train or evaluate")
    return frame.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None) -> str:
    import torch

    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


class VADataset:
    """Tokenise lazily; the collator pads each batch to its own longest item."""

    def __init__(self, texts, targets, langs, tokenizer, max_length: int):
        self.texts = list(texts)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.langs = list(langs)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoded = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_length)
        encoded["labels"] = self.targets[idx]
        return encoded


def make_collator(tokenizer):
    import torch

    def collate(features):
        labels = torch.tensor(np.stack([f.pop("labels") for f in features]), dtype=torch.float32)
        batch = tokenizer.pad(features, return_tensors="pt")
        batch["labels"] = labels
        return batch

    return collate


def evaluate(model, loader, device: str, langs: list[str]) -> dict:
    """Predict over a loader and report metrics overall and per language."""
    import torch

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            # Exactly the inference transform from intensity_ru.py.
            out = torch.sigmoid(model(**batch).logits).float().cpu().numpy()
            preds.append(out)
            trues.append(labels.numpy())

    pred = np.concatenate(preds)
    true = np.concatenate(trues)

    report: dict = {}
    for i, dim in enumerate(LABELS):
        report[dim] = regression_metrics(pred[:, i], true[:, i])

    # Per language, because stage 6B reuses this checkpoint on English. A model
    # that only orders Russian correctly makes the cross-language comparison
    # meaningless while looking fine on an aggregate score.
    unique = sorted(set(langs))
    if len(unique) > 1:
        langs_arr = np.asarray(langs)
        per_lang: dict = {}
        for lang in unique:
            mask = langs_arr == lang
            if mask.sum() < 20:
                continue
            per_lang[lang] = {
                dim: regression_metrics(pred[mask, i], true[mask, i])
                for i, dim in enumerate(LABELS)
            }
        if per_lang:
            report["per_language"] = per_lang
    return report


def log_report(report: dict, prefix: str = "") -> None:
    for dim in LABELS:
        m = report[dim]
        logger.info(
            "%s%-8s pearson %.4f  spearman %.4f  mse %.4f  mae %.4f  pred_std %.4f",
            prefix,
            dim,
            m["pearson"],
            m["spearman"],
            m["mse"],
            m["mae"],
            m["pred_std"],
        )
    for lang, dims in report.get("per_language", {}).items():
        for dim in LABELS:
            logger.info(
                "%s  [%s] %-8s pearson %.4f  mse %.4f",
                prefix,
                lang,
                dim,
                dims[dim]["pearson"],
                dims[dim]["mse"],
            )


def train(cfg: TrainConfig) -> dict:
    # Corpus first, deliberately: a wrong --text-col or a missing file is the
    # most common mistake here, and validating before torch and transformers
    # load turns a 15-second failure into an immediate one.
    frame = load_va_frame(cfg)

    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info("device: %s", device)
    texts = frame[cfg.text_col].tolist()
    targets = np.stack(
        [
            normalize_targets(frame[cfg.valence_col].to_numpy(), cfg.va_min, cfg.va_max),
            normalize_targets(frame[cfg.arousal_col].to_numpy(), cfg.va_min, cfg.va_max),
        ],
        axis=1,
    ).astype(np.float32)
    langs = (
        frame[cfg.lang_col].astype(str).tolist()
        if cfg.lang_col and cfg.lang_col in frame.columns
        else ["all"] * len(frame)
    )

    logger.info(
        "targets normalised from [%.2f, %.2f] to [0, 1]: valence mean %.3f std %.3f, "
        "arousal mean %.3f std %.3f",
        cfg.va_min,
        cfg.va_max,
        targets[:, 0].mean(),
        targets[:, 0].std(),
        targets[:, 1].mean(),
        targets[:, 1].std(),
    )

    # Plain shuffled split. Stratification is not meaningful for continuous
    # targets; the seed is recorded so the split is reproducible.
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(texts))
    n_val = max(20, int(len(texts) * cfg.val_fraction))
    val_idx, train_idx = order[:n_val], order[n_val:]
    logger.info("split: %d train, %d validation", len(train_idx), len(val_idx))

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model,
        num_labels=2,
        problem_type="regression",
        id2label={i: name for i, name in enumerate(LABELS)},
        label2id={name: i for i, name in enumerate(LABELS)},
    ).to(device)

    collate = make_collator(tokenizer)
    train_loader = DataLoader(
        VADataset(
            [texts[i] for i in train_idx],
            targets[train_idx],
            [langs[i] for i in train_idx],
            tokenizer,
            cfg.max_length,
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=cfg.num_workers,
        pin_memory=(device != "cpu"),
        drop_last=False,
    )
    val_langs = [langs[i] for i in val_idx]
    val_loader = DataLoader(
        VADataset(
            [texts[i] for i in val_idx],
            targets[val_idx],
            val_langs,
            tokenizer,
            cfg.max_length,
        ),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=cfg.num_workers,
        pin_memory=(device != "cpu"),
    )

    # No weight decay on biases or LayerNorm - standard for transformer
    # fine-tuning; decaying them costs a little accuracy for nothing.
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if ("bias" in name or "LayerNorm" in name) else decay).append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
    )

    total_steps = max(1, len(train_loader) * cfg.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * cfg.warmup_ratio), total_steps
    )

    use_bf16 = cfg.bf16 and device != "cpu" and torch.cuda.is_bf16_supported()
    logger.info(
        "steps: %d total (%d per epoch), bf16: %s", total_steps, len(train_loader), use_bf16
    )

    output = Path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)

    best_score = -math.inf
    best_report: dict = {}
    history = []
    started = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(
                device_type="cuda" if device != "cpu" else "cpu",
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                logits = model(**batch).logits
                # THE objective decision: squash first, then MSE. This matches
                # torch.sigmoid(outputs.logits) in intensity_{ru,en}.py. Training
                # MSE on raw logits instead would make inference squash all
                # predictions toward 0.5 with no error raised anywhere.
                loss = torch.nn.functional.mse_loss(torch.sigmoid(logits.float()), labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            running += loss.item() * labels.size(0)
            seen += labels.size(0)
            if step % 50 == 0 or step == len(train_loader):
                logger.info(
                    "epoch %d step %d/%d  train_mse %.5f  lr %.2e",
                    epoch,
                    step,
                    len(train_loader),
                    running / seen,
                    scheduler.get_last_lr()[0],
                )

        report = evaluate(model, val_loader, device, val_langs)
        score = summarize(report)
        logger.info("epoch %d validation (mean pearson %.4f):", epoch, score)
        log_report(report, prefix="  ")
        history.append({"epoch": epoch, "train_mse": running / seen, "score": score, **report})

        if score > best_score:
            best_score, best_report = score, report
            model.save_pretrained(output)
            tokenizer.save_pretrained(output)
            logger.info("epoch %d is the best so far; checkpoint written to %s", epoch, output)
        else:
            logger.info("epoch %d did not improve on %.4f; checkpoint kept", epoch, best_score)

    elapsed = time.time() - started
    cfg.metrics = {
        "best_mean_pearson": best_score,
        "best_epoch_report": best_report,
        "history": history,
        "train_seconds": round(elapsed, 1),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
    }
    (output / "train_config.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("finished in %.1f min; best mean pearson %.4f", elapsed / 60, best_score)

    # A collapsed head is the failure mode that still looks like a successful
    # run, so say so loudly rather than leaving it to be discovered later.
    for dim in LABELS:
        if best_report[dim]["pred_std"] < 0.05:
            logger.error(
                "%s predictions have std %.4f - the head has collapsed. Stage 6A's "
                "median threshold will not discriminate. Check --va-min/--va-max and "
                "the learning rate before using this checkpoint.",
                dim,
                best_report[dim]["pred_std"],
            )
    return cfg.metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the valence-arousal regressor for pipeline stages 6A/6B.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", required=True, help="corpus path, 'hf:<dataset_id>', or 'smoke'")
    p.add_argument("--output", required=True, help="checkpoint directory to write")
    p.add_argument("--base-model", default=TrainConfig.base_model)
    p.add_argument("--text-col", default=TrainConfig.text_col)
    p.add_argument("--valence-col", default=TrainConfig.valence_col)
    p.add_argument("--arousal-col", default=TrainConfig.arousal_col)
    p.add_argument("--lang-col", default=None, help="optional column for per-language metrics")
    p.add_argument("--va-min", type=float, default=TrainConfig.va_min, help="source scale minimum")
    p.add_argument("--va-max", type=float, default=TrainConfig.va_max, help="source scale maximum")
    p.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--eval-batch-size", type=int, default=TrainConfig.eval_batch_size)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--max-length", type=int, default=TrainConfig.max_length)
    p.add_argument("--val-fraction", type=float, default=TrainConfig.val_fraction)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    p.add_argument("--limit", type=int, default=None, help="use only the first N rows")
    p.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:N")
    p.add_argument("--no-bf16", action="store_true", help="disable bf16 autocast")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    cfg = TrainConfig(
        dataset=args.dataset,
        output=args.output,
        base_model=args.base_model,
        text_col=args.text_col,
        valence_col=args.valence_col,
        arousal_col=args.arousal_col,
        lang_col=args.lang_col,
        va_min=args.va_min,
        va_max=args.va_max,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        max_length=args.max_length,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
        limit=args.limit,
        device=args.device,
        bf16=not args.no_bf16,
    )
    try:
        train(cfg)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 2
    print(
        f"\nNext: python training/verify_checkpoint.py va {cfg.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
