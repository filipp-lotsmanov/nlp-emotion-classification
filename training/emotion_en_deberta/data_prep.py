"""Load and validate the English emotion corpus for `emotion-en-deberta`.

This was a rebuild skeleton until the built corpus turned up. It no longer
rebuilds anything: `corpora/super_emotion_clean.csv.gz` is the real thing, and
it reproduces every checkable figure in `docs/dataset_build.md`. What is left to
do is prove that on every run, because the corpus is the one input to training
that nothing downstream can validate.

    uv run python training/emotion_en_deberta/data_prep.py

Exit code 0 means the file matches the build record and is safe to train on.
Exit code 1 lists what diverged, and then the published metrics in
docs/model_cards/emotion_en_deberta.md are not comparable to whatever you train.

`training/emotion_en_deberta/train.py` calls `load_corpus` and `validate`
directly, so a corpus problem fails before any GPU time is spent.

Upstream: https://huggingface.co/datasets/cirimus/super-emotion (CC BY-SA 4.0).
That licence propagates to this file and to any model trained on it - see
docs/LICENSING.md before redistributing either.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = REPO / "corpora" / "super_emotion_clean.csv.gz"

# Class counts from docs/dataset_build.md (the build record), NOT from the model
# card's "Dataset Details" table - that table is wrong. Proof: 15% of each count
# below equals the model card's own held-out validation support for that class,
# within one row, for all seven classes, and sums to exactly 64,250. The card's
# dataset table matches none of them and sums to 433,387 rather than the 428,331
# the card itself states. See docs/PROVENANCE.md.
EXPECTED_CLASS_COUNTS = {
    "joy": 147_869,
    "sadness": 125_615,
    "anger": 57_963,
    "fear": 53_351,
    "surprise": 15_816,
    "disgust": 14_316,
    "neutral": 13_401,
}
EXPECTED_TOTAL = 428_331  # sums to the table above

# 9,151 synthetic Disgust rows were written for the original project and the file
# did not survive. The delivered corpus therefore stops at 419,180 rows with
# 5,165 Disgust, and that is the correct outcome, not a cleaning failure. Both
# figures are checked separately so an honest corpus is not reported as broken.
SYNTHETIC_DISGUST_ROWS = 9_151
REPRODUCIBLE_TOTAL = EXPECTED_TOTAL - SYNTHETIC_DISGUST_ROWS  # 419,180
REPRODUCIBLE_DISGUST = EXPECTED_CLASS_COUNTS["disgust"] - SYNTHETIC_DISGUST_ROWS  # 5,165

# The delivered schema. It is not the column list the model card implies
# (`labels`, `labels_str`, `text_length`); the card described a working frame
# that was never shipped. The file that exists is what the checks follow.
EXPECTED_COLUMNS = ["text", "label", "source", "labels_source", "token_count"]

# Label names must match what the pipeline understands, because
# normalize_emotion_label() in src/vea/stages/visualize.py rewrites anything
# else to "neutral". It lowercases first, so the corpus's Title-cased names are
# accepted - but the checkpoint's id2label is written lowercase anyway, since
# "anything else" failing silently to "neutral" is the failure mode here.
# Note the card writes "happiness" in its dataset table but "joy" in its metrics
# table - the pipeline wants "joy".
TARGET_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# docs/dataset_build.md, "The labels", rule 3: rows annotated as both disgust and
# neutral end up Neutral. The record puts that at 33 rows and says it is written
# down "because it is exactly the sort of detail that silently moves a class
# count" - so it is checked, for exactly that reason.
DISGUST_ANNOTATED_ROWS = REPRODUCIBLE_DISGUST + 33

# Per-source row counts, measured from the delivered file. These are not in the
# build record, which reports only pre-filter counts, so they pin the file rather
# than verify it. They exist because the register they describe is the corpus's
# main limitation: 97.2% Twitter against 2.6% television dialogue.
EXPECTED_SOURCE_COUNTS = {
    "ISEAR": 349_057,
    "Crowdflower": 34_416,
    "TwitterEmotion": 15_183,
    "MELD": 10_836,
    "SemEval": 8_741,
    "GoEmotions": 947,
}


def load_corpus(path: Path | str = DEFAULT_CORPUS):
    """Read the corpus into a DataFrame. Handles `.csv` and `.csv.gz` alike.

    Raises:
        FileNotFoundError: with the git-lfs-free reason it might be missing.
    """
    import pandas as pd

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. The corpus is committed gzipped at "
            f"{DEFAULT_CORPUS.relative_to(REPO)}; if this is a shallow or partial "
            "clone, fetch it before training."
        )
    frame = pd.read_csv(path, encoding="utf-8")
    return frame


def validate(frame) -> list[str]:
    """Compare the corpus to the published build record, and explain gaps.

    Two legitimate outcomes:
      428,331 rows - the published set, only possible with the synthetic Disgust
                     file, which nobody has.
      419,180 rows - the faithful corpus; every class matches except Disgust,
                     which lands at 5,165 because 9,151 rows cannot be rebuilt.

    Anything else is a build difference, and then the model card's metrics are
    not comparable to whatever you train.

    Returns:
        A list of problems, empty if the corpus matches.
    """
    problems: list[str] = []

    missing_columns = [c for c in EXPECTED_COLUMNS if c not in frame.columns]
    if missing_columns:
        problems.append(f"missing columns: {missing_columns} (have {list(frame.columns)})")
        return problems

    nulls = {c: int(frame[c].isna().sum()) for c in EXPECTED_COLUMNS}
    if any(nulls.values()):
        problems.append(f"null values present: { {k: v for k, v in nulls.items() if v} }")

    counts = frame["label"].str.lower().value_counts().to_dict()
    unexpected = set(counts) - set(TARGET_LABELS)
    if unexpected:
        problems.append(f"labels outside the 7-class scheme: {sorted(unexpected)}")

    total = len(frame)
    if total == EXPECTED_TOTAL:
        print(f"{total:,} rows: the published set (synthetic Disgust included).")
        expected = dict(EXPECTED_CLASS_COUNTS)
    elif total == REPRODUCIBLE_TOTAL:
        print(
            f"{total:,} rows: the faithful corpus "
            f"({REPRODUCIBLE_TOTAL:,} = {EXPECTED_TOTAL:,} minus "
            f"{SYNTHETIC_DISGUST_ROWS:,} unreproducible synthetic Disgust rows)."
        )
        expected = dict(EXPECTED_CLASS_COUNTS)
        expected["disgust"] = REPRODUCIBLE_DISGUST
    else:
        problems.append(
            f"row count {total:,} is neither the published {EXPECTED_TOTAL:,} nor the "
            f"reproducible {REPRODUCIBLE_TOTAL:,}; the build differs from the record"
        )
        expected = dict(EXPECTED_CLASS_COUNTS)

    # Exact, not tolerant. The record claims the build reproduces "to the row",
    # and it does; a tolerance here would only hide a real divergence.
    for label, want in expected.items():
        got = counts.get(label, 0)
        if got != want:
            problems.append(f"class {label!r}: {got:,} rows, record says {want:,}")

    problems.extend(_check_label_rules(frame))
    problems.extend(_check_sources(frame))
    return problems


def _check_label_rules(frame) -> list[str]:
    """Rules 2 and 3 of the label collapse, from docs/dataset_build.md.

    Rule 2 restores a row to Disgust when its source annotation says disgust,
    overruling the upstream seven-class mapping. Rule 3 applies the Neutral
    restoration afterwards, so a row annotated both ends up Neutral.

    These are checked because they are the two places where the published class
    counts came from a deliberate choice rather than from the data, which makes
    them the two places a silent change would be hardest to notice.
    """
    problems: list[str] = []

    annotations = frame["labels_source"].map(_parse_annotations)
    annotated_disgust = annotations.map(lambda names: "disgust" in names)
    n_annotated = int(annotated_disgust.sum())
    if n_annotated != DISGUST_ANNOTATED_ROWS:
        problems.append(
            f"{n_annotated:,} rows carry 'disgust' in labels_source; the record's "
            f"rules imply {DISGUST_ANNOTATED_ROWS:,} ({REPRODUCIBLE_DISGUST:,} "
            "restored to Disgust plus 33 that rule 3 sends to Neutral)"
        )

    labelled_disgust = frame["label"].str.lower() == "disgust"
    unrestored = int((labelled_disgust & ~annotated_disgust).sum())
    if unrestored:
        problems.append(
            f"{unrestored:,} rows are labelled Disgust without a 'disgust' source "
            "annotation; rule 2 says every Disgust row comes from one"
        )

    overridden = frame.loc[annotated_disgust & ~labelled_disgust, "label"].str.lower()
    not_neutral = overridden[overridden != "neutral"]
    if len(not_neutral):
        problems.append(
            f"{len(not_neutral):,} disgust-annotated rows are labelled "
            f"{sorted(set(not_neutral))}; rule 3 allows only Neutral"
        )

    if "love" in {name for names in annotations for name in names}:
        problems.append("'love' survives in labels_source; the drop-love step removed all of them")

    return problems


def _check_sources(frame) -> list[str]:
    """Per-source counts, and the register they add up to."""
    problems: list[str] = []
    got = frame["source"].value_counts().to_dict()

    if set(got) != set(EXPECTED_SOURCE_COUNTS):
        problems.append(
            f"source set changed: {sorted(got)} against {sorted(EXPECTED_SOURCE_COUNTS)}"
        )
        return problems

    for source, want in EXPECTED_SOURCE_COUNTS.items():
        if got[source] != want:
            problems.append(f"source {source!r}: {got[source]:,} rows, expected {want:,}")

    # Stated on every successful run, because it is the finding the report and
    # the model card both get wrong. The corpus is not a mix of questionnaire
    # narratives and television dialogue; it is Twitter.
    twitter = {"ISEAR", "Crowdflower", "TwitterEmotion", "SemEval"}
    n_twitter = sum(v for k, v in got.items() if k in twitter)
    print(
        f"register: {n_twitter / len(frame):.1%} Twitter-derived, "
        f"{got['MELD'] / len(frame):.2%} television dialogue (MELD), "
        f"{got['GoEmotions'] / len(frame):.2%} Reddit. The 'ISEAR' block is "
        "dair-ai/emotion mislabelled upstream - see docs/PROVENANCE.md section 7."
    )
    return problems


def _parse_annotations(value) -> list[str]:
    """`labels_source` is a stringified Python list, e.g. ``"['anger', 'disgust']"``.

    A malformed entry yields an empty list rather than raising, so validation
    reports every problem it found instead of dying on the first bad row.
    """
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return []
    return [str(x).lower() for x in parsed] if isinstance(parsed, (list, tuple)) else []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help="corpus .csv or .csv.gz (default: the committed one)",
    )
    args = parser.parse_args(argv)

    try:
        frame = load_corpus(args.corpus)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    print(f"{args.corpus}: {len(frame):,} rows, columns {list(frame.columns)}")
    problems = validate(frame)

    if problems:
        print("\nCorpus does NOT match the published build record:")
        for problem in problems:
            print(f"  - {problem}")
        print(
            "\nYou may still train on this, but the published metrics "
            "(accuracy 0.8995, macro F1 0.8127) are then not comparable to yours. "
            "Report your own numbers and state the dataset difference."
        )
        return 1

    print("\nCorpus matches the published build record.")
    print(
        f"Caveat to carry into any result: {SYNTHETIC_DISGUST_ROWS:,} synthetic Disgust "
        f"rows ({SYNTHETIC_DISGUST_ROWS / EXPECTED_TOTAL:.1%} of the published set, "
        f"{SYNTHETIC_DISGUST_ROWS / EXPECTED_CLASS_COUNTS['disgust']:.0%} of that class) "
        "are absent and cannot be rebuilt."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
