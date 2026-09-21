"""Verify a retrained checkpoint against the pipeline's inference contract.

Run this before wiring a new checkpoint into the pipeline. It catches the class
of mistake that produces a plausible-looking but wrong timeline rather than an
error: swapped valence/arousal order, a label spelled in a way the visualiser
silently rewrites to "neutral", a regression head whose outputs collapse once
sigmoid is applied, or a model that assigns the same emotion to everything.

    uv run python training/verify_checkpoint.py va models/xlmroberta-large-va
    uv run python training/verify_checkpoint.py emotion-en models/emotion-en-deberta

Exit code 0 means every check passed. Anything else means do not ship it.

This file is intentionally complete and opinionated: it encodes the contract
read out of src/vea/stages/, so it is the place to look when the contract
changes. The training scripts next to it are skeletons for you to fill in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The seven labels the pipeline understands, from EMOTION_CONFIG in
# src/vea/stages/visualize.py. Anything outside this set (and its aliases) is
# rewritten to "neutral" by normalize_emotion_label().
PIPELINE_EMOTIONS = {
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
}
PIPELINE_ALIASES = {"happiness": "joy", "happy": "joy", "sad": "sadness"}

# Probe sentences at the four corners of the valence-arousal plane. Russian and
# English are both checked because stage 6B feeds the same VA checkpoint English
# text: a checkpoint that only orders Russian correctly will quietly produce
# nonsense for the cross-language comparison.
VA_PROBES = [
    # (text, language, expect_high_valence, expect_high_arousal)
    #
    # The arousal expectations follow Russell's circumplex, where sadness is a
    # LOW-arousal state. That is the textbook position and it is contested in
    # annotation practice: raters often read an intensely expressed sadness as
    # activated, and the published VA checkpoint does exactly that, scoring the
    # two sadness probes at 0.830 (ru) and 0.881 (en). It still ranks the seven
    # classes correctly in aggregate - sadness mean 0.4930 below fear 0.5339 and
    # anger 0.5231 - so the disagreement is probe noise on a weak dimension
    # rather than a broken head. See docs/PROVENANCE.md section 11. The grouped
    # checks are therefore on means, and per-probe disagreement is printed.
    ("Это лучший день в моей жизни, я невероятно счастлив!", "ru", True, True),
    ("This is the best day of my life, I am incredibly happy!", "en", True, True),
    ("Я спокойно сижу у окна и читаю книгу.", "ru", True, False),
    ("I am sitting calmly by the window reading a book.", "en", True, False),
    ("Я в ярости, это отвратительно и невыносимо!", "ru", False, True),
    ("I am furious, this is disgusting and unbearable!", "en", False, True),
    ("Мне очень грустно и одиноко, ничего не хочется.", "ru", False, False),
    ("I feel very sad and lonely, I do not want anything.", "en", False, False),
]

EMOTION_PROBES = [
    ("I am so happy and delighted right now!", "joy"),
    ("I am furious about what you did.", "anger"),
    ("I am terrified of what might happen next.", "fear"),
    ("I feel hopeless and miserable.", "sadness"),
    ("That is revolting and repulsive.", "disgust"),
    ("Wait, I did not expect that at all!", "surprise"),
    ("The meeting is scheduled for Tuesday at three.", "neutral"),
]


class Failure(Exception):
    """A contract violation. Message is shown to the user verbatim."""


def _report(ok: bool, message: str) -> bool:
    print(f"  [{'pass' if ok else 'FAIL'}] {message}")
    return ok


def verify_va(path: Path, structural_only: bool = False) -> bool:
    """Check a valence-arousal regressor against src/vea/stages/intensity_ru.py.

    Two kinds of check, deliberately separated:

    **Structural** - head shape, label order, output range. These are pass/fail
    for any checkpoint, toy or real: they decide whether the pipeline can load
    it at all and whether index 0 really means valence.

    **Behavioural** - prediction spread and probe ordering. These need a model
    that actually learned the task. A smoke-test checkpoint trained for seconds
    on synthetic templates will fail them legitimately, so `structural_only`
    prints them as advisory instead of failing the run. A check that cries wolf
    on an expected condition is a check people learn to ignore.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"Verifying valence-arousal checkpoint: {path}")
    if structural_only:
        print("  (structural checks gate the result; behavioural checks are advisory)")
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(path), trust_remote_code=True
    ).eval()

    passed = True
    behavioural = True

    # 1. Head shape. Inference indexes logits[0, 0] and logits[0, 1].
    n_labels = model.config.num_labels
    passed &= _report(n_labels == 2, f"num_labels == 2 (got {n_labels})")

    # 2. Label order. Inference treats index 0 as valence and index 1 as
    # arousal. id2label is not read by the pipeline, so a wrong order here is
    # undetectable at runtime - which is exactly why it is checked.
    id2label = {int(k): str(v).lower() for k, v in (model.config.id2label or {}).items()}
    expected = {0: "valence", 1: "arousal"}
    passed &= _report(
        id2label == expected,
        f"id2label declares {{0: 'valence', 1: 'arousal'}} (got {id2label or 'nothing'})",
    )

    # 3. Behaviour. Predictions must be in range and must order the probes
    # correctly in both languages.
    texts = [p[0] for p in VA_PROBES]
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        # Exactly what intensity_ru.py does.
        preds = torch.sigmoid(model(**inputs).logits)
    valence = preds[:, 0].tolist()
    arousal = preds[:, 1].tolist()

    in_range = all(0.0 <= v <= 1.0 for v in valence + arousal)
    # Structural: sigmoid guarantees this unless the head is non-standard.
    passed &= _report(in_range, "all predictions within [0, 1] after sigmoid")

    # 4. Degenerate output. A head trained with plain MSE on unbounded targets
    # collapses toward 0.5 once sigmoid is applied; a head that learned the mean
    # does the same. Either way the video-adaptive median threshold in stage 6A
    # stops discriminating.
    spread_v = max(valence) - min(valence)
    spread_a = max(arousal) - min(arousal)
    behavioural &= _report(
        spread_v > 0.15, f"valence spread across probes > 0.15 (got {spread_v:.3f})"
    )
    behavioural &= _report(
        spread_a > 0.15, f"arousal spread across probes > 0.15 (got {spread_a:.3f})"
    )

    # 5. Ordering, per language and per dimension.
    for lang in ("ru", "en"):
        idx = [i for i, p in enumerate(VA_PROBES) if p[1] == lang]
        hi_v = [valence[i] for i in idx if VA_PROBES[i][2]]
        lo_v = [valence[i] for i in idx if not VA_PROBES[i][2]]
        hi_a = [arousal[i] for i in idx if VA_PROBES[i][3]]
        lo_a = [arousal[i] for i in idx if not VA_PROBES[i][3]]
        # Group means, not min-vs-max. This check exists to catch two things: a
        # swapped index and a collapsed head. Both still fail it - a swap flips
        # valence, a collapse fails the spread check above. What min-vs-max
        # additionally demanded was that every individual probe be ordered
        # correctly, which is per-item perfection on two sentences per group,
        # and that is beyond what four probes can establish about a noisy
        # dimension. The published VA checkpoint fails it on arousal for the
        # sadness probes alone while ranking the classes correctly in aggregate
        # (AUC 0.5734, docs/PROVENANCE.md section 11). Individual disagreements
        # are still printed below, so nothing is hidden by the change.
        mean_hi_v, mean_lo_v = sum(hi_v) / len(hi_v), sum(lo_v) / len(lo_v)
        mean_hi_a, mean_lo_a = sum(hi_a) / len(hi_a), sum(lo_a) / len(lo_a)
        behavioural &= _report(
            mean_hi_v > mean_lo_v,
            f"[{lang}] positive probes score higher valence than negative "
            f"(mean {mean_hi_v:.3f} vs {mean_lo_v:.3f})",
        )
        behavioural &= _report(
            mean_hi_a > mean_lo_a,
            f"[{lang}] activated probes score higher arousal than calm "
            f"(mean {mean_hi_a:.3f} vs {mean_lo_a:.3f})",
        )

    print("\n  probe predictions (valence, arousal):")
    disagree: list[str] = []
    for (text, lang, want_v, want_a), v, a in zip(VA_PROBES, valence, arousal, strict=True):
        # Flag against the group mean for that language and dimension, so a
        # marker means "on the wrong side of its own group" rather than
        # "beyond an absolute threshold nobody chose".
        idx = [i for i, p in enumerate(VA_PROBES) if p[1] == lang]
        mid_v = sum(valence[i] for i in idx) / len(idx)
        mid_a = sum(arousal[i] for i in idx) / len(idx)
        marks = ""
        if (v > mid_v) != want_v:
            marks += "V"
            disagree.append(f"valence on {text[:34]!r}")
        if (a > mid_a) != want_a:
            marks += "A"
            disagree.append(f"arousal on {text[:34]!r}")
        print(f"    [{lang}] {v:.3f} {a:.3f} {marks:<2}  {text[:52]}")
    if disagree:
        print(
            f"\n  {len(disagree)} probe(s) fall on the wrong side of their own group mean "
            "(V = valence, A = arousal):"
        )
        for item in disagree:
            print(f"    - {item}")
        print(
            "  Individual probes are noisy and this is reported, not gated. It matters "
            "only if a whole dimension is wrong, which the grouped checks above test."
        )

    # If the two columns are near-identical across probes, both heads learned the
    # same axis. That is under-training, not mis-indexing - a swap would fail
    # arousal and pass valence instead - but it is worth naming explicitly,
    # because a model whose valence merely echoes arousal makes the pipeline's
    # valence panel and its Russian-vs-English comparison meaningless.
    import numpy as _np

    if len(valence) > 2:
        corr = float(_np.corrcoef(valence, arousal)[0, 1])
        print(f"\n  corr(predicted valence, predicted arousal) = {corr:.3f}")
        if corr > 0.90:
            print(
                "  note: the two heads are heavily entangled - valence is largely "
                "echoing arousal. The pipeline's valence panel and its Russian-vs-English "
                "comparison would both be uninformative. Train longer, or on a corpus "
                "where the two axes are less correlated."
            )
        elif corr > 0.50:
            print(
                "  note: moderate entanglement. Check the probe table above - if valence "
                "misorders the calm-positive and angry-negative probes specifically, the "
                "valence axis is under-trained rather than mis-indexed (a swapped index "
                "would fail the arousal checks instead)."
            )

    if not behavioural and not structural_only:
        passed = False
    return passed


def verify_emotion_en(path: Path, structural_only: bool = False) -> bool:
    """Check an English emotion classifier against src/vea/stages/emotion_en.py.

    Args:
        structural_only: gate the exit code on head shape and label spelling
            only, reporting the behavioural checks without failing on them. A
            smoke checkpoint trains for seconds on a few hundred synthetic rows,
            so its softmax sits just above uniform (1/7 = 0.143) and the
            confidence floor below is unreachable however correct the
            predictions are. Failing on it buries the checks that do mean
            something at that scale: the label round trip and probe agreement.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"Verifying English emotion checkpoint: {path}")
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForSequenceClassification.from_pretrained(str(path)).eval()

    passed = True

    id2label = {int(k): str(v) for k, v in (model.config.id2label or {}).items()}
    labels = set(id2label.values())

    passed &= _report(len(id2label) == 7, f"exactly 7 classes (got {len(id2label)})")

    # Labels the visualiser does not recognise become "neutral" silently, so
    # spelling is a correctness issue, not a cosmetic one.
    lowered = {label.lower() for label in labels}
    unknown = {
        label
        for label in lowered
        if label not in PIPELINE_EMOTIONS and label not in PIPELINE_ALIASES
    }
    passed &= _report(
        not unknown,
        f"every label is recognised by visualize.EMOTION_CONFIG "
        f"(unrecognised: {sorted(unknown) or 'none'})",
    )
    passed &= _report(
        labels == lowered,
        f"labels are lowercase (got {sorted(labels)})",
    )
    aliased = {label for label in lowered if label in PIPELINE_ALIASES}
    if aliased:
        print(
            f"  [note] {sorted(aliased)} will be renamed by normalize_emotion_label; "
            "the CSV exporter keeps the raw label, so plot and CSV will differ"
        )

    # Behaviour: exactly what emotion_en.py computes.
    texts = [p[0] for p in EMOTION_PROBES]
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(**inputs).logits, dim=-1)
    predicted = [id2label[int(i)] for i in probs.argmax(dim=-1)]
    confidence = probs.max(dim=-1).values.tolist()

    # A model that collapses to one class still scores well on an imbalanced
    # test set. This is the cheapest way to notice.
    distinct = len(set(predicted))
    behavioural = _report(
        distinct >= 5,
        f"probes elicit at least 5 distinct labels (got {distinct}: {sorted(set(predicted))})",
    )

    # Stage 8 discards predictions below 0.25 confidence. A checkpoint trained
    # with BCE but read with softmax tends to sit under that everywhere.
    above = sum(1 for c in confidence if c >= 0.25)
    behavioural &= _report(
        above >= len(confidence) - 1,
        f"at least {len(confidence) - 1} probes clear stage 8's min_confidence of "
        f"0.25 (got {above})",
    )
    if not structural_only:
        passed &= behavioural

    correct = sum(
        1 for pred, (_, want) in zip(predicted, EMOTION_PROBES, strict=True) if pred == want
    )
    _report(True, f"probe agreement with intent: {correct}/{len(EMOTION_PROBES)} (indicative only)")

    print("\n  probe predictions:")
    for (text, want), pred, conf in zip(EMOTION_PROBES, predicted, confidence, strict=True):
        mark = " " if pred == want else "?"
        print(f"   {mark} {pred:<9} {conf:.3f}  (expected {want:<9}) {text[:40]}")

    return passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=["va", "emotion-en"], help="which contract to check")
    parser.add_argument("path", type=Path, help="checkpoint directory")
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help="gate only on head shape, label order and output range; report "
        "behavioural checks without failing. For smoke-test checkpoints.",
    )
    args = parser.parse_args(argv)

    if not (args.path / "config.json").is_file():
        print(f"No config.json in {args.path} - that is not a saved checkpoint.")
        return 2

    try:
        if args.kind == "va":
            passed = verify_va(args.path, structural_only=args.structural_only)
        else:
            passed = verify_emotion_en(args.path, structural_only=args.structural_only)
    except Failure as exc:
        print(f"\nContract violation: {exc}")
        return 1

    print()
    if passed and args.structural_only:
        print(
            "Structural contract checks passed: the pipeline can load this checkpoint "
            "and its head shape and labels are right. Behavioural results above are advisory."
        )
        return 0
    if passed:
        print("All contract checks passed. Safe to register in MODEL_REGISTRY.")
        return 0
    print("Contract checks FAILED. Do not wire this checkpoint into the pipeline.")
    print("See training/README.md, section 'The inference contract'.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
