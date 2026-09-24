"""Train `emotion-en-deberta`: 7-class English emotion classification for stage 7B.

Produces a checkpoint that src/vea/stages/emotion_en.py can load unchanged. The
contract that module imposes, read from its source:

    model = AutoModelForSequenceClassification.from_pretrained(path)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_id = torch.argmax(probabilities, dim=-1).item()
    predicted_emotion = self.id2label[predicted_id]

Five consequences drive the design here:

1. `num_labels = 7` and **`id2label` is load-bearing**. Unlike the VA regressor,
   the label names come out of the checkpoint config, and anything
   visualize.normalize_emotion_label() does not recognise is silently rewritten
   to "neutral". So the names are written lowercase, from LABELS, and asserted
   by training/verify_checkpoint.py.
2. Inference applies **softmax**, so the loss is cross-entropy on raw logits.
   The model card claims a "Linear layer (768 -> 7) with sigmoid activation"
   and "7-dimensional sigmoid probabilities"; the shipped inference code uses
   softmax. For argmax the two agree, so the predicted label is unaffected -
   but the reported per-class confidence is not, and stage 8 drops predictions
   below 0.25 confidence. Cross-entropy makes those confidences mean what the
   pipeline reads them as. See docs/PROVENANCE.md.
3. The corpus text is **already normalised** by the twelve-step cleaner, and is
   not cleaned again here. `check_corpus_is_preclean` asserts that on a sample,
   so a future edit to vea.text_clean that would desynchronise training from
   inference fails here rather than silently at inference.
4. The 9,151 synthetic Disgust rows are gone, and that class's published
   precision was 0.6064 against recall 0.9157 - the augmented class was
   *over*-predicted. Class weighting would reproduce that, so it is off by
   default. `--class-weights balanced` is there to be measured against, not
   assumed. Model selection uses macro F1, which resists the majority classes
   without touching the loss.
5. **Any evaluation on dair-ai/emotion is in-distribution, not external.** The
   corpus's 349,057 rows labelled `ISEAR` are dair-ai/emotion (CARER)
   mislabelled upstream, so they are 83% of the training data. The model card
   reports "external validation" on 2,000 CARER samples at 92.55% accuracy,
   *higher* than its own held-out 89.95% - which is what contamination looks
   like. `carer_exposure` is reported on every run.

Usage:

    # 1. Prove the loop runs, in ~1 minute, with no corpus and no big download:
    python training/emotion_en_deberta/train.py --corpus smoke \
        --base-model <small-model> --output models/emotion-en-smoke --epochs 1

    # 2. The real run (the corpus is committed, so --corpus is not needed):
    python training/emotion_en_deberta/train.py --output models/emotion-en-deberta

    # 3. Confirm the checkpoint satisfies the pipeline contract:
    python training/verify_checkpoint.py emotion-en models/emotion-en-deberta

    # Or, on the server, with GPU selection and tmux:
    scripts/train_emotion_en.sh --tmux

Then switch the second entry of `stage_7b_english_emotion` in
src/vea/pipeline.py from `emotion-en-emoberta` to `emotion-en-deberta`, and
re-run any video whose results you intend to report. Until you do, the
"DeBERTa" column in every output CSV is EmoBERTa - see docs/PROVENANCE.md.

The base model defaults to `microsoft/deberta-v3-base`, which is what the card's
"DeBERTa-V2-Base" means: vocabulary 128,100, a SentencePiece tokenizer, 12
layers, hidden 768, ~183M parameters and relative p2c/c2p attention are all
v3-base's numbers, and v3 checkpoints load through the `DebertaV2` classes in
transformers, which is where the "V2" comes from. DeBERTa-v1-base has a 50,265
BPE vocabulary and ~140M parameters, so it is not the model described.

Card-derived defaults where the card is specific (3 epochs, max length 512, 15%
validation), conventional where it is not (batch 16 for its "small batch size",
lr 2e-5, 6% warmup).
Everything that defines a run lands in `train_config.json` beside the weights.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_prep import DEFAULT_CORPUS, TARGET_LABELS, load_corpus, validate  # noqa: E402

logger = logging.getLogger("emotion_en_train")

# Index order is written into id2label and read back at inference. Alphabetical,
# so it is reproducible from nothing but this list.
LABELS = list(TARGET_LABELS)
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}

# The corpus's `source` value for the misattributed dair-ai/emotion block.
# See docs/PROVENANCE.md section 7 and corpora/README.md.
CARER_SOURCE_LABEL = "ISEAR"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Everything needed to reproduce a run. Serialised beside the weights."""

    output: str
    corpus: str = str(DEFAULT_CORPUS)
    base_model: str = "microsoft/deberta-v3-base"
    text_col: str = "text"
    label_col: str = "label"
    learning_rate: float = 2e-5  # not recorded by the card; standard for a base model
    batch_size: int = 16  # the card says "small batch size (to limit compute)"
    eval_batch_size: int = 128
    epochs: int = 3  # the card is explicit
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    # The card is explicit, and it is what emotion_en.py passes at inference.
    # Truncation only: the collator pads each batch to its own longest item, and
    # the corpus's longest row is 178 whitespace tokens, so 512 costs nothing
    # here while guaranteeing training and inference truncate identically.
    max_length: int = 512
    # 15%, because the card's own per-class validation supports are 15% of the
    # build record's class counts, to the row, in all seven classes. Keeping the
    # fraction makes the held-out numbers comparable to the published ones.
    val_fraction: float = 0.15
    class_weights: str = "none"
    label_smoothing: float = 0.0
    seed: int = 42
    num_workers: int = 8
    bf16: bool = True
    limit: int | None = None
    device: str | None = None
    skip_corpus_validation: bool = False
    metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pure helpers - no torch, so they are unit-testable without a GPU stack
# ---------------------------------------------------------------------------


def encode_labels(names) -> np.ndarray:
    """Map label names to ids, case-insensitively.

    Raises:
        ValueError: on any name outside the seven-class scheme. Tolerating one
            would put it in the training set under whatever id it happened to
            get, and the pipeline would then rewrite that class to "neutral" at
            inference with no error raised anywhere.
    """
    lowered = [str(n).strip().lower() for n in names]
    unknown = sorted({n for n in lowered if n not in LABEL_TO_ID})
    if unknown:
        raise ValueError(f"labels outside the 7-class scheme: {unknown}. Expected {LABELS}.")
    return np.asarray([LABEL_TO_ID[n] for n in lowered], dtype=np.int64)


def compute_class_weights(label_ids: np.ndarray, scheme: str) -> np.ndarray | None:
    """Per-class loss weights, or None for unweighted cross-entropy.

    Schemes:
        none      no weighting. The default, and the one comparable to the
                  published run. See consequence 4 in the module docstring.
        balanced  n / (k * count), the sklearn convention. Full inverse
                  frequency: on this corpus that is 11x on Neutral against Joy,
                  and 29x on Disgust.
        sqrt      the square root of `balanced`, renormalised to mean 1. Damped,
                  for when `balanced` destabilises the rare classes.

    Raises:
        ValueError: on an unknown scheme, or a class absent from the split.
    """
    if scheme == "none":
        return None
    counts = np.bincount(label_ids, minlength=len(LABELS)).astype(np.float64)
    if (counts == 0).any():
        absent = [LABELS[i] for i in np.flatnonzero(counts == 0)]
        raise ValueError(f"cannot weight classes absent from the training split: {absent}")

    balanced = len(label_ids) / (len(LABELS) * counts)
    if scheme == "balanced":
        weights = balanced
    elif scheme == "sqrt":
        weights = np.sqrt(balanced)
    else:
        raise ValueError(f"unknown class-weight scheme {scheme!r}; use none, balanced or sqrt")
    return (weights / weights.mean()).astype(np.float32)


def classification_metrics(pred_ids: np.ndarray, true_ids: np.ndarray) -> dict:
    """Per-class precision/recall/F1 plus macro, micro and weighted summaries.

    Computed from a confusion matrix with numpy rather than sklearn, so the
    tests for it run in the dev group, which deliberately has neither sklearn
    nor torch. Macro F1 leads because accuracy on this corpus is 64% decided by
    Joy and Sadness alone: a model answering only those two scores 0.64.

    Raises:
        ValueError: on mismatched shapes, which otherwise broadcasts into a
            plausible-looking confusion matrix.
    """
    k = len(LABELS)
    pred_ids = np.asarray(pred_ids, dtype=np.int64)
    true_ids = np.asarray(true_ids, dtype=np.int64)
    if pred_ids.shape != true_ids.shape:
        raise ValueError(f"shape mismatch: {pred_ids.shape} predictions, {true_ids.shape} truths")

    # confusion[t, p]: rows are truth, columns are prediction.
    confusion = np.zeros((k, k), dtype=np.int64)
    np.add.at(confusion, (true_ids, pred_ids), 1)

    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    correct = np.diag(confusion)

    precision = np.where(predicted > 0, correct / np.maximum(predicted, 1), 0.0)
    recall = np.where(support > 0, correct / np.maximum(support, 1), 0.0)
    denominator = precision + recall
    f1 = np.where(denominator > 0, 2 * precision * recall / np.maximum(denominator, 1e-12), 0.0)

    total = int(support.sum())
    # Macro averages over the classes actually present, so a split that happens
    # to miss a rare class is not silently scored as 0 on it.
    present = support > 0
    report = {
        "accuracy": float(correct.sum() / total) if total else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "macro_precision": float(precision[present].mean()) if present.any() else 0.0,
        "macro_recall": float(recall[present].mean()) if present.any() else 0.0,
        "weighted_f1": float((f1 * support).sum() / total) if total else 0.0,
        "n": total,
        "per_class": {
            LABELS[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "predicted": int(predicted[i]),
            }
            for i in range(k)
        },
        "confusion": confusion.tolist(),
    }
    # Micro F1 equals accuracy for single-label multi-class. Reported because
    # the card reports both, and their being equal there (0.8995) is one of the
    # few internally consistent things in it.
    report["micro_f1"] = report["accuracy"]
    return report


def grouped_stratified_split(
    texts: list[str], label_ids: np.ndarray, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split rows into train/validation, stratified by class, grouped by text.

    Stratification matters because Disgust is 1.2% of the reproducible corpus;
    an unstratified 15% split gives a Disgust estimate wide enough to make the
    card's 0.6064 precision unfalsifiable.

    Grouping matters because 444 texts appear more than once and 178 of them
    under two different labels, so a plain row-wise split can put the same
    string on both sides. That is only 362 rows, but the leak is free to avoid.

    Each unique text is assigned to one side using the label of its first
    occurrence, and every row sharing that text follows it.

    Returns:
        (train_indices, val_indices) into the original row order.

    Raises:
        ValueError: if val_fraction is not a proper fraction.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    first_label: dict[str, int] = {}
    rows_by_text: dict[str, list[int]] = {}
    for row, (text, label_id) in enumerate(zip(texts, label_ids, strict=True)):
        if text not in first_label:
            first_label[text] = int(label_id)
            rows_by_text[text] = []
        rows_by_text[text].append(row)

    by_class: dict[int, list[str]] = {}
    for text, label_id in first_label.items():
        by_class.setdefault(label_id, []).append(text)

    rng = np.random.default_rng(seed)
    val_texts: set[str] = set()
    for label_id in sorted(by_class):
        group = by_class[label_id]
        order = rng.permutation(len(group))
        # round(), not floor: the card's supports are 15% of each class count
        # rounded, which is how a stratified split hits an exact total. Capped
        # at len - 1 so no class is emptied out of the training side.
        n_val = min(len(group) - 1, max(1, round(len(group) * val_fraction)))
        val_texts.update(group[i] for i in order[:n_val])

    val_idx = np.asarray(
        sorted(row for text in val_texts for row in rows_by_text[text]), dtype=np.int64
    )
    train_idx = np.asarray(
        sorted(row for text, rows in rows_by_text.items() if text not in val_texts for row in rows),
        dtype=np.int64,
    )
    return train_idx, val_idx


def build_smoke_frame(n_per_template: int = 40, seed: int = 0):
    """Synthetic 7-class data in the corpus's schema, for proving the loop.

    Not a substitute for the corpus - the templates are trivially separable, so
    a high macro F1 here means only that gradients flow, the label order
    survives the round trip through id2label, and the saved checkpoint satisfies
    the pipeline contract. Use it to debug plumbing before spending GPU hours.

    Text is written in the cleaner's output convention (lower case, [CAPS]
    markers) so `check_corpus_is_preclean` passes on it too.
    """
    import pandas as pd

    rng = random.Random(seed)
    templates = [
        ("i am so happy and delighted about this", "joy"),
        ("what a wonderful and lovely day this is", "joy"),
        ("i feel hopeless and miserable and empty", "sadness"),
        ("everything is bleak and i am grieving", "sadness"),
        ("i am furious about what you did [CAPS] stop", "anger"),
        ("this makes me livid and enraged", "anger"),
        ("i am terrified of what happens next", "fear"),
        ("i am scared and my hands are shaking", "fear"),
        ("that is revolting and repulsive and vile", "disgust"),
        ("it is sickening and i am nauseated", "disgust"),
        ("wait i did not expect that at all", "surprise"),
        ("i am astonished this is unbelievable", "surprise"),
        ("the meeting is scheduled for tuesday at three", "neutral"),
        ("the report was filed and the file was closed", "neutral"),
    ]
    rows = []
    for text, label in templates:
        for _ in range(n_per_template):
            filler = " ".join(
                rng.choice(["today", "again", "here", "now", "really", ""]) for _ in range(2)
            )
            # Single-spaced deliberately: the cleaner's last step is
            # collapse_whitespace, so a double space here would make the smoke
            # corpus fail check_corpus_is_preclean and mislead whoever is
            # debugging the loop.
            body = " ".join(f"{text} {filler}".split())
            rows.append(
                {
                    "text": body,
                    "label": label.capitalize(),
                    "source": "smoke",
                    "labels_source": f"['{label}']",
                    "token_count": len(body.split()),
                }
            )
    rng.shuffle(rows)
    return pd.DataFrame(rows)


def check_corpus_is_preclean(texts: list[str], sample: int = 3000, seed: int = 0) -> float:
    """Fraction of a sample already in `vea.text_clean.clean_text`'s output form.

    The corpus was built with that cleaner and is not cleaned again here, so
    training and inference agree only as long as the cleaner is unchanged. The
    measured baseline is 99.9%: the shortfall is the `[TAG]` placeholder, which
    the vendored cleaner folds to `[tag]` and the published build left
    uppercase. A materially lower figure means the cleaner has drifted and the
    corpus must be rebuilt before this checkpoint is trustworthy.

    Returns:
        The idempotence rate in [0, 1].
    """
    from vea.text_clean import clean_text

    rng = random.Random(seed)
    picked = texts if len(texts) <= sample else rng.sample(texts, sample)
    unchanged = sum(1 for t in picked if clean_text(t) == t)
    return unchanged / len(picked) if picked else 1.0


def carer_exposure(sources) -> float:
    """Share of rows drawn from the misattributed dair-ai/emotion block.

    Reported on every run because the model card's headline external-validation
    result is 2,000 CARER samples, and CARER is most of the training data. An
    "external" set that beats the internal one by 2.6 points is the shape of a
    contaminated benchmark, not of good generalisation.
    """
    values = [str(s) for s in sources]
    if not values:
        return 0.0
    return sum(1 for s in values if s == CARER_SOURCE_LABEL) / len(values)


def load_frame(cfg: TrainConfig):
    """Load and check the corpus, before torch and transformers are imported.

    Raises:
        ValueError: if the corpus diverges from the build record, or lacks the
            configured columns, or is too small to split.
    """
    if cfg.corpus == "smoke":
        frame = build_smoke_frame(seed=cfg.seed)
        logger.info("using synthetic smoke data: %d rows", len(frame))
    else:
        frame = load_corpus(cfg.corpus)
        logger.info("loaded %d rows from %s", len(frame), cfg.corpus)
        if cfg.skip_corpus_validation:
            logger.warning("corpus validation skipped by request")
        else:
            problems = validate(frame)
            if problems:
                raise ValueError(
                    "corpus does not match docs/dataset_build.md:\n  - "
                    + "\n  - ".join(problems)
                    + "\n\nFix the corpus, or pass --skip-corpus-validation and state "
                    "in your report that the published metrics are not comparable."
                )

    for column in (cfg.text_col, cfg.label_col):
        if column not in frame.columns:
            raise ValueError(
                f"corpus is missing column {column!r}. Available: {list(frame.columns)}."
            )

    frame = frame.dropna(subset=[cfg.text_col, cfg.label_col]).copy()
    frame[cfg.text_col] = frame[cfg.text_col].astype(str)
    frame = frame[frame[cfg.text_col].str.strip().str.len() > 0]
    if cfg.limit:
        # Sampled, not head(): the corpus is ordered by source, so head() would
        # take MELD only and never see a single ISEAR or Crowdflower row.
        frame = frame.sample(n=min(cfg.limit, len(frame)), random_state=cfg.seed)
    if len(frame) < 4 * len(LABELS):
        raise ValueError(f"only {len(frame)} usable rows; too few to train and evaluate")
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


class EmotionDataset:
    """Tokenise lazily; the collator pads each batch to its own longest item."""

    def __init__(self, texts, label_ids, tokenizer, max_length: int):
        self.texts = list(texts)
        self.label_ids = np.asarray(label_ids, dtype=np.int64)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoded = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_length)
        encoded["labels"] = int(self.label_ids[idx])
        return encoded


def make_collator(tokenizer):
    import torch

    def collate(features):
        labels = torch.tensor([f.pop("labels") for f in features], dtype=torch.long)
        batch = tokenizer.pad(features, return_tensors="pt")
        batch["labels"] = labels
        return batch

    return collate


def evaluate(model, loader, device: str) -> dict:
    """Predict over a loader and report the full classification report.

    Also reports the card's calibration evidence: mean softmax confidence on
    correct against incorrect predictions (the card claims 0.8873 against
    0.4275). Stage 8 discards predictions below 0.25 confidence, so a narrow
    gap changes the timeline, not just the table.
    """
    import torch

    model.eval()
    preds, trues, confidences = [], [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            # Exactly the inference transform from emotion_en.py.
            probabilities = torch.nn.functional.softmax(model(**batch).logits.float(), dim=-1)
            top = torch.max(probabilities, dim=-1)
            preds.append(top.indices.cpu().numpy())
            confidences.append(top.values.cpu().numpy())
            trues.append(labels.numpy())

    pred_ids = np.concatenate(preds)
    true_ids = np.concatenate(trues)
    confidence = np.concatenate(confidences)

    report = classification_metrics(pred_ids, true_ids)
    hit = pred_ids == true_ids
    report["confidence"] = {
        "correct_mean": float(confidence[hit].mean()) if hit.any() else float("nan"),
        "incorrect_mean": float(confidence[~hit].mean()) if (~hit).any() else float("nan"),
        "below_stage8_threshold": float((confidence < 0.25).mean()),
    }
    return report


def log_report(report: dict, prefix: str = "") -> None:
    logger.info(
        "%saccuracy %.4f  macro_f1 %.4f  weighted_f1 %.4f  (n=%d)",
        prefix,
        report["accuracy"],
        report["macro_f1"],
        report["weighted_f1"],
        report["n"],
    )
    logger.info(
        "%s%-9s %8s %8s %8s %8s %10s",
        prefix,
        "class",
        "prec",
        "recall",
        "f1",
        "support",
        "predicted",
    )
    for name in LABELS:
        m = report["per_class"][name]
        logger.info(
            "%s%-9s %8.4f %8.4f %8.4f %8d %10d",
            prefix,
            name,
            m["precision"],
            m["recall"],
            m["f1"],
            m["support"],
            m["predicted"],
        )
    conf = report.get("confidence")
    if conf:
        logger.info(
            "%sconfidence: %.4f when correct, %.4f when wrong, %.2f%% below stage 8's 0.25",
            prefix,
            conf["correct_mean"],
            conf["incorrect_mean"],
            100 * conf["below_stage8_threshold"],
        )


def train(cfg: TrainConfig) -> dict:
    # Corpus first, deliberately: a corpus that does not match the build record
    # invalidates every comparison to the published metrics, and finding that
    # out after three epochs is three epochs too late.
    frame = load_frame(cfg)
    texts = frame[cfg.text_col].tolist()
    label_ids = encode_labels(frame[cfg.label_col])

    preclean_rate = check_corpus_is_preclean(texts, seed=cfg.seed)
    logger.info(
        "corpus is already in the cleaner's output form for %.2f%% of a sample; "
        "no cleaning is applied here (inference applies clean_text to raw text)",
        100 * preclean_rate,
    )
    if preclean_rate < 0.95:
        logger.error(
            "only %.1f%% of sampled rows survive clean_text unchanged, against a "
            "measured baseline of 99.9%%. vea.text_clean has drifted from the cleaner "
            "that built this corpus, so training and inference will normalise "
            "differently. Rebuild the corpus or revert the cleaner.",
            100 * preclean_rate,
        )

    if "source" in frame.columns:
        exposure = carer_exposure(frame["source"])
        if exposure > 0:
            logger.warning(
                "%.1f%% of training rows are dair-ai/emotion (labelled 'ISEAR' upstream). "
                "Do not report an evaluation on dair-ai/emotion or CARER as external - "
                "it is in-distribution. See docs/PROVENANCE.md section 7.",
                100 * exposure,
            )

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

    train_idx, val_idx = grouped_stratified_split(texts, label_ids, cfg.val_fraction, cfg.seed)
    logger.info("split: %d train, %d validation (grouped by text)", len(train_idx), len(val_idx))

    train_labels = label_ids[train_idx]
    weights = compute_class_weights(train_labels, cfg.class_weights)
    counts = np.bincount(train_labels, minlength=len(LABELS))
    for i, name in enumerate(LABELS):
        logger.info(
            "  %-9s %7d train rows (%5.2f%%)%s",
            name,
            counts[i],
            100 * counts[i] / len(train_labels),
            f"  weight {weights[i]:.3f}" if weights is not None else "",
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model,
        num_labels=len(LABELS),
        problem_type="single_label_classification",
        # Load-bearing: emotion_en.py reads these back, and the pipeline rewrites
        # any name it does not recognise to "neutral" without raising.
        id2label={i: name for i, name in enumerate(LABELS)},
        label2id=dict(LABEL_TO_ID),
    ).to(device)

    collate = make_collator(tokenizer)
    train_loader = DataLoader(
        EmotionDataset([texts[i] for i in train_idx], train_labels, tokenizer, cfg.max_length),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=cfg.num_workers,
        pin_memory=(device != "cpu"),
        drop_last=False,
    )
    val_loader = DataLoader(
        EmotionDataset([texts[i] for i in val_idx], label_ids[val_idx], tokenizer, cfg.max_length),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=cfg.num_workers,
        pin_memory=(device != "cpu"),
    )

    loss_fn = torch.nn.CrossEntropyLoss(
        weight=None if weights is None else torch.tensor(weights, device=device),
        label_smoothing=cfg.label_smoothing,
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
        "steps: %d total (%d per epoch), bf16: %s, class weights: %s",
        total_steps,
        len(train_loader),
        use_bf16,
        cfg.class_weights,
    )

    output = Path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)

    best_score = -1.0
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
                # Cross-entropy on raw logits, which is softmax + NLL - the same
                # softmax emotion_en.py applies at inference. The card's claim of
                # a sigmoid head would mean seven independent binary problems;
                # for a single-label argmax task that changes the reported
                # confidences without changing the answer, and the shipped
                # inference code is the contract that matters.
                loss = loss_fn(logits.float(), labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            running += loss.item() * labels.size(0)
            seen += labels.size(0)
            if step % 200 == 0 or step == len(train_loader):
                logger.info(
                    "epoch %d step %d/%d  train_loss %.5f  lr %.2e",
                    epoch,
                    step,
                    len(train_loader),
                    running / seen,
                    scheduler.get_last_lr()[0],
                )

        report = evaluate(model, val_loader, device)
        score = report["macro_f1"]
        logger.info("epoch %d validation:", epoch)
        log_report(report, prefix="  ")
        history.append({"epoch": epoch, "train_loss": running / seen, **report})

        if score > best_score:
            best_score, best_report = score, report
            model.save_pretrained(output)
            tokenizer.save_pretrained(output)
            logger.info("epoch %d is the best so far; checkpoint written to %s", epoch, output)
        else:
            logger.info(
                "epoch %d did not improve on macro F1 %.4f; checkpoint kept", epoch, best_score
            )

    elapsed = time.time() - started
    cfg.metrics = {
        "best_macro_f1": best_score,
        "best_epoch_report": best_report,
        "history": history,
        "train_seconds": round(elapsed, 1),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "label_order": LABELS,
        "preclean_rate": preclean_rate,
        "corpus_caveats": [
            "9,151 synthetic Disgust rows from the published training set could not be "
            "rebuilt (2.1% of rows, 64% of the Disgust class), so these metrics are not "
            "directly comparable to the card's 0.8995 accuracy / 0.8127 macro F1.",
            "83% of the corpus is dair-ai/emotion, mislabelled 'ISEAR' upstream. Any "
            "evaluation on dair-ai/emotion or CARER is in-distribution, not external.",
        ],
    }
    (output / "train_config.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("finished in %.1f min; best macro F1 %.4f", elapsed / 60, best_score)

    # A class the model never predicts is the failure mode that still looks like
    # a successful run, because accuracy barely moves when Neutral (3.1% of the
    # corpus) is abandoned entirely.
    for name in LABELS:
        m = best_report["per_class"][name]
        if m["support"] and m["predicted"] == 0:
            logger.error(
                "%s was never predicted on the validation set (%d true rows). The head "
                "has abandoned the class; try --class-weights balanced.",
                name,
                m["support"],
            )
        elif m["support"] and m["recall"] < 0.2:
            logger.warning(
                "%s recall is %.3f on %d rows - stage 7B's ensemble will effectively "
                "never see this class from this model.",
                name,
                m["recall"],
                m["support"],
            )
    return cfg.metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the 7-class English emotion classifier for pipeline stage 7B.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output", required=True, help="checkpoint directory to write")
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="corpus path or 'smoke'")
    p.add_argument("--base-model", default=TrainConfig.base_model)
    p.add_argument("--text-col", default=TrainConfig.text_col)
    p.add_argument("--label-col", default=TrainConfig.label_col)
    p.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--eval-batch-size", type=int, default=TrainConfig.eval_batch_size)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--max-length", type=int, default=TrainConfig.max_length)
    p.add_argument("--val-fraction", type=float, default=TrainConfig.val_fraction)
    p.add_argument(
        "--class-weights",
        choices=["none", "balanced", "sqrt"],
        default=TrainConfig.class_weights,
        help="loss weighting; the module docstring says why 'none' is the default",
    )
    p.add_argument("--label-smoothing", type=float, default=TrainConfig.label_smoothing)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    p.add_argument("--limit", type=int, default=None, help="train on a random N rows")
    p.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:N")
    p.add_argument("--no-bf16", action="store_true", help="disable bf16 autocast")
    p.add_argument(
        "--skip-corpus-validation",
        action="store_true",
        help="train on a corpus that does not match the build record (say so in your report)",
    )
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
        output=args.output,
        corpus=args.corpus,
        base_model=args.base_model,
        text_col=args.text_col,
        label_col=args.label_col,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        max_length=args.max_length,
        val_fraction=args.val_fraction,
        class_weights=args.class_weights,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
        num_workers=args.num_workers,
        limit=args.limit,
        device=args.device,
        bf16=not args.no_bf16,
        skip_corpus_validation=args.skip_corpus_validation,
    )
    try:
        train(cfg)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 2
    print(
        f"\nNext: python training/verify_checkpoint.py emotion-en {cfg.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
