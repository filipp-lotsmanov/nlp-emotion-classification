"""Tests for the English emotion trainer and its corpus.

Split in two. `TestCorpus*` reads the committed corpus and pins the figures the
rest of the project reasons about - the build record's class counts, the
already-cleaned text, the misattributed source block. `Test*` beyond that covers
the pure helpers in the trainer, which are deliberately torch-free so they run
in CI's fast job.

Nothing here loads a model, downloads anything, or needs a GPU.
"""

from __future__ import annotations

import ast
import gzip
import hashlib
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO, load_training_module

# Loaded by path rather than by import name: both trainers are called train.py,
# so `from train import ...` resolves to whichever test module ran first. See
# tests/conftest.py.
_prep = load_training_module("emotion_en_deberta", "data_prep")
_train = load_training_module("emotion_en_deberta")

DEFAULT_CORPUS = _prep.DEFAULT_CORPUS
EXPECTED_CLASS_COUNTS = _prep.EXPECTED_CLASS_COUNTS
EXPECTED_COLUMNS = _prep.EXPECTED_COLUMNS
EXPECTED_SOURCE_COUNTS = _prep.EXPECTED_SOURCE_COUNTS
EXPECTED_TOTAL = _prep.EXPECTED_TOTAL
REPRODUCIBLE_DISGUST = _prep.REPRODUCIBLE_DISGUST
REPRODUCIBLE_TOTAL = _prep.REPRODUCIBLE_TOTAL
SYNTHETIC_DISGUST_ROWS = _prep.SYNTHETIC_DISGUST_ROWS
TARGET_LABELS = _prep.TARGET_LABELS
load_corpus = _prep.load_corpus
validate = _prep.validate

CARER_SOURCE_LABEL = _train.CARER_SOURCE_LABEL
LABELS = _train.LABELS
build_smoke_frame = _train.build_smoke_frame
carer_exposure = _train.carer_exposure
check_corpus_is_preclean = _train.check_corpus_is_preclean
classification_metrics = _train.classification_metrics
compute_class_weights = _train.compute_class_weights
encode_labels = _train.encode_labels
grouped_stratified_split = _train.grouped_stratified_split

RUSSIAN_CORPUS = REPO / "corpora" / "ru_izard_emotions.csv.gz"
VISUALIZE = REPO / "src" / "vea" / "stages" / "visualize.py"


def _emotion_config_keys() -> set[str]:
    """Keys of `EMOTION_CONFIG` in visualize.py, read without importing it."""
    tree = ast.parse(VISUALIZE.read_text(encoding="utf-8"), filename=str(VISUALIZE))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "EMOTION_CONFIG" for t in node.targets
        ):
            assert isinstance(node.value, ast.Dict)
            return {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    raise AssertionError(f"EMOTION_CONFIG not found at module level in {VISUALIZE}")


@pytest.fixture(scope="module")
def corpus():
    if not DEFAULT_CORPUS.is_file():
        pytest.skip(f"{DEFAULT_CORPUS} is not present")
    return load_corpus(DEFAULT_CORPUS)


# ---------------------------------------------------------------------------
# The corpus itself
# ---------------------------------------------------------------------------


class TestCorpusMatchesTheBuildRecord:
    """docs/dataset_build.md is the authority; this is the file it describes.

    Every figure here was measured against the delivered corpus, not copied
    from the record, and they agree. Pinning them means a re-delivered or
    re-built corpus that differs fails here rather than producing a checkpoint
    whose metrics are quietly incomparable to the published ones.
    """

    def test_it_validates_clean(self, corpus):
        assert validate(corpus) == []

    def test_the_row_count_is_the_reproducible_build(self, corpus):
        assert len(corpus) == REPRODUCIBLE_TOTAL == 419_180

    def test_the_arithmetic_of_the_missing_synthetic_rows(self):
        assert REPRODUCIBLE_TOTAL + SYNTHETIC_DISGUST_ROWS == EXPECTED_TOTAL
        assert EXPECTED_CLASS_COUNTS["disgust"] == REPRODUCIBLE_DISGUST + SYNTHETIC_DISGUST_ROWS
        assert sum(EXPECTED_CLASS_COUNTS.values()) == EXPECTED_TOTAL

    def test_the_schema_is_the_delivered_one(self, corpus):
        # Not the model card's column list. The card names `labels`,
        # `labels_str` and `text_length`, which describe a working frame that
        # was never shipped.
        assert list(corpus.columns) == EXPECTED_COLUMNS

    def test_every_class_count_matches_exactly(self, corpus):
        counts = corpus["label"].str.lower().value_counts().to_dict()
        expected = dict(EXPECTED_CLASS_COUNTS)
        expected["disgust"] = REPRODUCIBLE_DISGUST
        assert counts == expected

    def test_the_labels_are_the_seven_the_pipeline_understands(self, corpus):
        assert sorted(corpus["label"].str.lower().unique()) == TARGET_LABELS

    def test_there_are_no_nulls(self, corpus):
        assert corpus.isna().sum().sum() == 0

    def test_the_card_supports_are_fifteen_percent_of_these_counts(self):
        # The cross-check that makes the build record authoritative over the
        # card's dataset table: the card's own per-class validation supports are
        # 15% of the record's class counts, rounded, in all seven classes, and
        # they sum to exactly 64,250. Neither number is derivable from the
        # other. See docs/PROVENANCE.md.
        card_supports = {
            "anger": 8_695,
            "disgust": 2_147,
            "fear": 8_003,
            "joy": 22_181,
            "neutral": 2_010,
            "sadness": 18_842,
            "surprise": 2_372,
        }
        assert sum(card_supports.values()) == 64_250
        assert round(EXPECTED_TOTAL * 0.15) == 64_250
        # Within one row, not exactly: rounding each class independently gives
        # 64,248, so a stratified split hitting 64,250 has to hand the two
        # leftover rows to two classes. Joy and Anger are the two that are one
        # above their own rounded share, and every other class is exact - which
        # is a tighter fit than "approximately 15%" would be.
        off_by_one = []
        for label, support in card_supports.items():
            delta = support - round(EXPECTED_CLASS_COUNTS[label] * 0.15)
            assert delta in (0, 1), f"{label}: card {support}, 15% is {delta} away"
            if delta:
                off_by_one.append(label)
        assert sorted(off_by_one) == ["anger", "joy"]

    def test_the_card_dataset_table_does_not_match_and_does_not_self_sum(self):
        # The defect this project must not quote from: the card's dataset table
        # sums to 433,387 while the card itself states 428,331, and its class
        # names are misassigned against the descending order of its own counts.
        card_table = [149_321, 127_866, 63_532, 54_041, 16_075, 13_401, 9_151]
        assert sum(card_table) == 433_387 != EXPECTED_TOTAL
        assert sorted(card_table, reverse=True) != sorted(
            EXPECTED_CLASS_COUNTS.values(), reverse=True
        )


class TestCorpusLabelRules:
    """Rules 2 and 3 of the label collapse, from docs/dataset_build.md."""

    def test_rule_two_restores_exactly_the_published_disgust_count(self, corpus):
        annotated = corpus["labels_source"].str.contains("'disgust'")
        labelled = corpus["label"].str.lower() == "disgust"
        assert int((annotated & labelled).sum()) == REPRODUCIBLE_DISGUST == 5_165
        # Every Disgust row comes from a disgust source annotation.
        assert int((labelled & ~annotated).sum()) == 0

    def test_rule_three_sends_thirty_three_rows_to_neutral(self, corpus):
        annotated = corpus["labels_source"].str.contains("'disgust'")
        overridden = corpus.loc[annotated & (corpus["label"] != "Disgust"), "label"]
        assert len(overridden) == 33
        assert set(overridden) == {"Neutral"}

    def test_drop_love_left_nothing_behind(self, corpus):
        assert not corpus["labels_source"].str.contains("love").any()


class TestCorpusProvenance:
    """The register, and the source block that is not what it says it is."""

    def test_the_source_counts_are_pinned(self, corpus):
        assert corpus["source"].value_counts().to_dict() == EXPECTED_SOURCE_COUNTS

    def test_the_corpus_is_overwhelmingly_twitter(self, corpus):
        twitter = {"ISEAR", "Crowdflower", "TwitterEmotion", "SemEval"}
        share = corpus["source"].isin(twitter).mean()
        assert share > 0.97, f"{share:.1%} - the register claim in the report depends on this"

    def test_television_dialogue_is_a_rounding_error(self, corpus):
        # The record frames GoEmotions' exclusion as protecting 13,708 lines of
        # television dialogue. MELD survives at 10,836 rows, 2.6% of the corpus.
        assert (corpus["source"] == "MELD").mean() < 0.03

    def test_the_isear_block_is_not_isear(self, corpus):
        # Published ISEAR is 7,666 punctuated questionnaire narratives over
        # seven classes including disgust, shame and guilt. This block is
        # 349,057 unpunctuated "i feel" tweets over five classes: it is
        # dair-ai/emotion (CARER). See docs/PROVENANCE.md section 7.
        block = corpus[corpus["source"] == CARER_SOURCE_LABEL]
        assert len(block) == 349_057
        assert not block["text"].str.contains(r"[.,!?]", regex=True).any()
        assert block["text"].str.contains(r"\bfeel|\bfelt", regex=True).mean() > 0.98
        assert sorted(block["label"].unique()) == [
            "Anger",
            "Fear",
            "Joy",
            "Sadness",
            "Surprise",
        ]

    def test_the_rest_of_the_corpus_is_punctuated(self, corpus):
        # The contrast is the evidence: the ISEAR block's 0% punctuation is a
        # property of that corpus, not of the cleaner.
        other = corpus[corpus["source"] != CARER_SOURCE_LABEL]
        assert other["text"].str.contains(r"[.,!?]", regex=True).mean() > 0.6

    def test_disgust_comes_mostly_from_semeval_not_goemotions(self, corpus):
        # The record says GoEmotions "is kept for disgust" because "disgust is
        # the class the other five corpora barely have". SemEval supplies four
        # times as much.
        by_source = corpus.loc[corpus["label"] == "Disgust", "source"].value_counts()
        assert by_source["SemEval"] == 3_936
        assert by_source["GoEmotions"] == 914
        assert by_source["SemEval"] > 4 * by_source["GoEmotions"]


class TestCorpusTextIsAlreadyClean:
    def test_the_cleaner_is_idempotent_on_the_corpus(self, corpus):
        # The trainer relies on this: it does not clean, because cleaning again
        # would move ~0.1% of rows away from what the published model saw.
        rate = check_corpus_is_preclean(corpus["text"].tolist(), sample=2000, seed=7)
        assert rate > 0.99, (
            f"only {rate:.2%} of sampled rows survive clean_text unchanged. "
            "vea.text_clean has drifted from the cleaner that built this corpus."
        )

    def test_the_known_shortfall_is_the_tag_placeholder(self):
        from vea.text_clean import clean_text

        # The one systematic disagreement, documented in corpora/README.md.
        # mark_shouting protects a placeholder only when it is a whole token, so
        # a bare [TAG] survives but one glued to punctuation is lowercased - and
        # the published build kept the uppercase form. Affects ~0.1% of rows and
        # no realistic pipeline input, since [TAG] is a Twitter-mention artefact.
        assert clean_text("[TAG] hello") == "[TAG] hello"
        assert clean_text("[TAG]._.; hi") == "[tag]._.; hi"

    def test_the_caps_marker_is_present_in_the_corpus(self, corpus):
        # mark_shouting is the strongest single predictor in the error analysis,
        # so its token has to actually be in the training data.
        assert corpus["text"].str.contains(r"\[CAPS\]", regex=True).sum() == 5_003

    def test_token_count_is_whitespace_not_subwords(self, corpus):
        # Pinned so nobody reaches for it to set a tokeniser max_length.
        sample = corpus.sample(5000, random_state=0)
        approx = sample["text"].str.split().str.len()
        assert sample["token_count"].corr(approx) > 0.999


class TestCorpusIntegrity:
    """Digests over the decompressed bytes; gzip output is not byte-stable."""

    @staticmethod
    def _inner_digest(path: Path) -> str:
        with gzip.open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    def test_english_corpus_digest(self):
        if not DEFAULT_CORPUS.is_file():
            pytest.skip(f"{DEFAULT_CORPUS} is not present")
        assert (
            self._inner_digest(DEFAULT_CORPUS)
            == "eeff9a8c3c613bfd8cff5f5f53bbcb4336aea06522c0ac6952bc630aad175d6b"
        )

    def test_russian_corpus_digest(self):
        if not RUSSIAN_CORPUS.is_file():
            pytest.skip(f"{RUSSIAN_CORPUS} is not present")
        assert (
            self._inner_digest(RUSSIAN_CORPUS)
            == "b4481810e890d79d9f47a601310a6fc8d875750a13461714828d7deaca276da1"
        )


@pytest.fixture(scope="module")
def russian():
    if not RUSSIAN_CORPUS.is_file():
        pytest.skip(f"{RUSSIAN_CORPUS} is not present")
    return load_corpus(RUSSIAN_CORPUS)


class TestRussianCorpus:
    def test_shape_and_labels(self, russian):
        assert len(russian) == 24_766
        assert list(russian.columns) == ["text", "label", "source"]
        assert sorted(russian["label"].str.lower().unique()) == TARGET_LABELS

    def test_it_is_raw_not_cleaned(self, russian):
        # So a model trained on it must be served raw text, which is why
        # MODEL_REGISTRY["emotion-ru-finetuned"].preprocess stays None.
        assert not russian["text"].str.contains(r"\[CAPS\]", regex=True).any()
        assert (
            russian["text"].str.replace(r"\[[A-Z]+\]", "", regex=True).str.contains(r"[A-Z]").sum()
            == 2_553
        )

    def test_the_russian_model_declares_no_text_transform(self):
        from vea.config import MODEL_REGISTRY

        assert MODEL_REGISTRY["emotion-ru-finetuned"].preprocess is None

    def test_the_untranslated_fragments_are_pinned(self, russian):
        # 0.94% of rows have no Cyrillic at all - untranslated English left over
        # from the machine translation upstream. Documented, not fixed here.
        no_cyrillic = ~russian["text"].str.contains(r"[а-яА-ЯёЁ]", regex=True)
        assert int(no_cyrillic.sum()) == 234


# ---------------------------------------------------------------------------
# Trainer helpers
# ---------------------------------------------------------------------------


class TestLabelEncoding:
    def test_the_order_is_alphabetical_and_matches_the_pipeline(self):
        assert LABELS == TARGET_LABELS == sorted(TARGET_LABELS)

    def test_every_label_is_one_the_pipeline_recognises(self):
        # visualize.py imports matplotlib, which the fast CI job deliberately
        # does not install, so EMOTION_CONFIG's keys are read by parsing the
        # source rather than by importing it - the same trade
        # tests/test_source_hygiene.py makes for pipeline.py.
        #
        # This is the check that matters most in this file: a label name outside
        # EMOTION_CONFIG is rewritten to "neutral" by normalize_emotion_label()
        # with nothing raised anywhere, so a typo in LABELS would silently
        # collapse a whole class in the timeline.
        assert set(LABELS) == _emotion_config_keys()

    def test_it_accepts_the_corpus_title_case(self):
        assert encode_labels(["Joy", "Disgust", "neutral"]).tolist() == [3, 1, 4]

    def test_an_unknown_label_is_rejected_rather_than_mapped(self):
        # Silently tolerating one would put it in training under an arbitrary id
        # and the pipeline would rewrite that class to "neutral" at inference.
        with pytest.raises(ValueError, match="outside the 7-class scheme"):
            encode_labels(["joy", "love"])


class TestClassWeights:
    def test_none_means_no_weighting(self):
        assert compute_class_weights(np.array([0, 1, 2, 3, 4, 5, 6]), "none") is None

    def test_balanced_is_inverse_frequency_normalised_to_mean_one(self):
        ids = np.array([0] * 90 + [1] * 10 + list(range(2, 7)))
        weights = compute_class_weights(ids, "balanced")
        assert weights is not None
        assert weights[0] < weights[1] < weights[2]
        assert pytest.approx(float(weights.mean()), abs=1e-6) == 1.0

    def test_sqrt_is_damped_relative_to_balanced(self):
        ids = np.array([0] * 90 + [1] * 10 + list(range(2, 7)))
        balanced = compute_class_weights(ids, "balanced")
        sqrt = compute_class_weights(ids, "sqrt")
        spread = max(balanced) / min(balanced)
        assert 1.0 < max(sqrt) / min(sqrt) < spread

    def test_an_absent_class_is_an_error_not_an_infinity(self):
        with pytest.raises(ValueError, match="absent from the training split"):
            compute_class_weights(np.array([0, 1, 2, 3, 4, 5]), "balanced")

    def test_an_unknown_scheme_is_rejected(self):
        with pytest.raises(ValueError, match="unknown class-weight scheme"):
            compute_class_weights(np.arange(7), "focal")


class TestClassificationMetrics:
    def test_a_perfect_prediction(self):
        ids = np.arange(7)
        report = classification_metrics(ids, ids)
        assert report["accuracy"] == 1.0
        assert report["macro_f1"] == 1.0
        assert report["micro_f1"] == report["accuracy"]

    def test_majority_only_scores_well_on_accuracy_and_badly_on_macro_f1(self):
        # The reason macro F1 selects the checkpoint: on this corpus's balance,
        # answering Joy to everything scores 0.35 accuracy and 0.07 macro F1.
        true = np.array([3] * 35 + [5] * 29 + [0] * 14 + [2] * 12 + [6] * 4 + [1] * 3 + [4] * 3)
        pred = np.full_like(true, 3)
        report = classification_metrics(pred, true)
        assert report["accuracy"] == pytest.approx(0.35)
        assert report["macro_f1"] < 0.08
        assert report["per_class"]["neutral"]["predicted"] == 0

    def test_precision_and_recall_are_not_transposed(self):
        # One true joy predicted as neutral: joy loses recall, neutral loses
        # precision. Getting this backwards is invisible in aggregate scores.
        true = np.array([3, 3, 3, 4])
        pred = np.array([3, 3, 4, 4])
        report = classification_metrics(pred, true)
        assert report["per_class"]["joy"]["recall"] == pytest.approx(2 / 3)
        assert report["per_class"]["joy"]["precision"] == 1.0
        assert report["per_class"]["neutral"]["precision"] == pytest.approx(0.5)
        assert report["per_class"]["neutral"]["recall"] == 1.0

    def test_confusion_rows_are_truth_and_columns_are_prediction(self):
        report = classification_metrics(np.array([4]), np.array([3]))
        confusion = report["confusion"]
        assert confusion[3][4] == 1
        assert confusion[4][3] == 0

    def test_an_absent_class_does_not_drag_macro_f1_to_zero(self):
        ids = np.array([0, 0, 1, 1])
        report = classification_metrics(ids, ids)
        assert report["macro_f1"] == 1.0

    def test_weighted_f1_respects_support(self):
        true = np.array([3] * 9 + [4])
        pred = np.array([3] * 9 + [3])
        report = classification_metrics(pred, true)
        assert report["weighted_f1"] > report["macro_f1"]

    def test_mismatched_shapes_are_rejected(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            classification_metrics(np.arange(3), np.arange(4))


class TestGroupedStratifiedSplit:
    @staticmethod
    def _frame(n_per_class: int = 40):
        texts, labels = [], []
        for label_id in range(7):
            for i in range(n_per_class):
                texts.append(f"class {label_id} item {i}")
                labels.append(label_id)
        return texts, np.asarray(labels, dtype=np.int64)

    def test_it_partitions_every_row_exactly_once(self):
        texts, labels = self._frame()
        train_idx, val_idx = grouped_stratified_split(texts, labels, 0.15, 42)
        assert sorted([*train_idx, *val_idx]) == list(range(len(texts)))

    def test_every_class_appears_on_both_sides(self):
        texts, labels = self._frame()
        train_idx, val_idx = grouped_stratified_split(texts, labels, 0.15, 42)
        assert set(labels[train_idx]) == set(range(7))
        assert set(labels[val_idx]) == set(range(7))

    def test_the_validation_fraction_is_hit_per_class(self):
        texts, labels = self._frame(n_per_class=100)
        _, val_idx = grouped_stratified_split(texts, labels, 0.15, 42)
        counts = np.bincount(labels[val_idx], minlength=7)
        assert counts.tolist() == [15] * 7

    def test_a_duplicated_text_never_straddles_the_split(self):
        # The leak this exists to prevent: 178 corpus texts carry two different
        # labels, so a row-wise split can train and validate on the same string.
        texts = [f"unique {i}" for i in range(140)]
        labels = list(np.arange(140) % 7)
        texts += ["shared string", "shared string"]
        labels += [1, 0]
        train_idx, val_idx = grouped_stratified_split(
            texts, np.asarray(labels, dtype=np.int64), 0.15, 42
        )
        train_texts = {texts[i] for i in train_idx}
        val_texts = {texts[i] for i in val_idx}
        assert not train_texts & val_texts
        # Both rows of the shared text landed together.
        shared_rows = {140, 141}
        assert shared_rows <= set(train_idx.tolist()) or shared_rows <= set(val_idx.tolist())

    def test_it_is_deterministic_for_a_seed_and_not_for_different_seeds(self):
        texts, labels = self._frame()
        a = grouped_stratified_split(texts, labels, 0.15, 1)
        b = grouped_stratified_split(texts, labels, 0.15, 1)
        c = grouped_stratified_split(texts, labels, 0.15, 2)
        assert np.array_equal(a[1], b[1])
        assert not np.array_equal(a[1], c[1])

    def test_no_class_is_emptied_out_of_training(self):
        texts = ["only anger row", "a", "b", "c", "d", "e", "f"]
        labels = np.asarray([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
        train_idx, _ = grouped_stratified_split(texts, labels, 0.9, 0)
        assert set(labels[train_idx]) == set(range(7))

    def test_an_impossible_fraction_is_rejected(self):
        texts, labels = self._frame()
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="val_fraction"):
                grouped_stratified_split(texts, labels, bad, 42)


class TestSmokeFrame:
    def test_it_has_the_corpus_schema(self):
        frame = build_smoke_frame()
        assert list(frame.columns) == EXPECTED_COLUMNS

    def test_it_covers_all_seven_classes(self):
        frame = build_smoke_frame()
        assert sorted(frame["label"].str.lower().unique()) == TARGET_LABELS

    def test_its_text_is_already_in_the_cleaner_convention(self):
        # So --corpus smoke does not trip the preclean warning and mislead
        # someone debugging the loop.
        frame = build_smoke_frame()
        assert check_corpus_is_preclean(frame["text"].tolist()) == 1.0

    def test_it_is_deterministic(self):
        assert build_smoke_frame(seed=3).equals(build_smoke_frame(seed=3))

    def test_it_is_too_small_to_be_mistaken_for_the_corpus(self):
        assert len(build_smoke_frame()) < 1000


class TestCarerExposure:
    def test_it_measures_the_misattributed_block(self):
        assert carer_exposure([CARER_SOURCE_LABEL] * 3 + ["MELD"]) == 0.75

    def test_the_smoke_corpus_has_no_exposure(self):
        assert carer_exposure(build_smoke_frame()["source"]) == 0.0

    def test_an_empty_input_is_zero_not_a_division_error(self):
        assert carer_exposure([]) == 0.0

    def test_the_real_corpus_is_mostly_carer(self, corpus):
        assert carer_exposure(corpus["source"]) > 0.83


class TestCorpusValidationCatchesDivergence:
    def test_a_missing_column_is_reported(self, corpus):
        problems = validate(corpus.drop(columns=["labels_source"]))
        assert any("missing columns" in p for p in problems)

    def test_a_wrong_row_count_is_reported(self, corpus):
        problems = validate(corpus.head(1000))
        assert any("neither the published" in p for p in problems)

    def test_a_relabelled_row_is_reported(self, corpus):
        tampered = corpus.copy()
        tampered.loc[tampered.index[0], "label"] = "Joy"
        assert validate(tampered)

    def test_an_out_of_scheme_label_is_reported(self, corpus):
        tampered = corpus.copy()
        tampered.loc[tampered.index[0], "label"] = "Love"
        assert any("outside the 7-class scheme" in p for p in validate(tampered))


class TestRussianModelLabelsAreRecognised:
    """Every label stage 7A can emit must resolve to one of the seven classes.

    The first full pipeline run hit the failure this guards: the hub checkpoint
    `Djacon/rubert-tiny2-russian-emotion-detection` emitted `enthusiasm` on 45
    of 311 segments, `EMOTION_CONFIG` did not list it, and
    `normalize_emotion_label` rewrote every one to "neutral" without raising.
    14.5% of a video, silently relabelled to the opposite end of the valence
    axis. See docs/PROVENANCE.md section 13.
    """

    def test_enthusiasm_maps_to_joy_not_neutral(self):
        # Not a guess: the build record for that model's own training corpus
        # states "collapse by priority; enthusiasm becomes Joy", 1,115 rows.
        from vea.config import canonicalize_emotion

        assert canonicalize_emotion("enthusiasm") == "joy"

    def test_the_observed_russian_labels_are_all_recognised(self):
        # Every label the Russian model produced on the first full run. Any of
        # these unrecognised means that share of a video is silently relabelled.
        from vea.config import EMOTION_CLASSES, canonicalize_emotion

        observed = {"neutral", "happiness", "enthusiasm", "anger", "fear", "surprise", "sadness"}
        for label in observed:
            assert canonicalize_emotion(label) in EMOTION_CLASSES, label


class TestOneEmotionVocabulary:
    """`vea.config` is the only place that decides what a label means.

    Three modules used to decide independently and did not agree, which on the
    first full pipeline run produced two wrong published numbers: a Neutral
    proportion inflated by 45 relabelled segments, and stages 8 and 9 reporting
    different ensemble agreement for the same 311 segments (37 disagreements
    against 42). See docs/PROVENANCE.md section 13.
    """

    def test_canonicalising_covers_every_label_the_run_produced(self):
        from vea.config import canonicalize_emotion

        observed = {
            "neutral",
            "happiness",
            "enthusiasm",
            "anger",
            "fear",
            "surprise",
            "sadness",
            "disgust",
            "joy",
        }
        for label in observed:
            assert canonicalize_emotion(label) in _emotion_config_keys()

    def test_enthusiasm_and_happiness_are_one_class(self):
        # The 5-segment gap between stage 8 and stage 9 was exactly this: the
        # Russian model says `enthusiasm`, the English models say `joy`, and a
        # raw string comparison called that a disagreement.
        from vea.config import canonicalize_emotion

        assert canonicalize_emotion("enthusiasm") == canonicalize_emotion("happiness") == "joy"

    def test_an_unknown_label_raises_rather_than_defaulting(self):
        # Defaulting to "neutral" is what hid `enthusiasm`: it yields a
        # plausible timeline, a plausible CSV, and no error anywhere.
        from vea.config import canonicalize_emotion

        with pytest.raises(ValueError, match="unrecognised emotion label"):
            canonicalize_emotion("guilt")
        assert canonicalize_emotion("guilt", default="neutral") == "neutral"

    def test_the_visualiser_recognises_exactly_the_config_vocabulary(self):
        """`EMOTION_CONFIG`'s aliases are derived from `EMOTION_ALIASES`.

        Read by parsing, since visualize.py imports matplotlib and the fast CI
        job does not install it. A literal alias list here would be a second
        source of truth, which is the defect this class exists to prevent.
        """
        from vea.config import EMOTION_ALIASES, EMOTION_CLASSES

        source = VISUALIZE.read_text(encoding="utf-8")
        assert "from vea.config import EMOTION_ALIASES" in source
        assert '"aliases": ["happiness"' not in source, (
            "visualize.py has gone back to a literal alias list; derive it from "
            "vea.config.EMOTION_ALIASES instead"
        )
        assert set(_emotion_config_keys()) == set(EMOTION_CLASSES)
        assert set(EMOTION_ALIASES.values()) <= set(EMOTION_CLASSES)

    def test_the_exporter_uses_the_shared_vocabulary(self):
        export = (REPO / "src" / "vea" / "stages" / "export.py").read_text(encoding="utf-8")
        assert "from vea.config import canonicalize_emotion, display_emotion" in export
        assert '"joy": "Happiness"' not in export, (
            "export.py has gone back to its own display map; it must use vea.config.display_emotion"
        )
        # The vote must canonicalise, or synonyms count as disagreement again.
        assert "_vote_label(emotion_ru)" in export
        assert "_vote_label(emotion_en_distil)" in export
        assert "_vote_label(emotion_en_deberta)" in export

    def test_display_names_cover_all_seven_classes(self):
        from vea.config import EMOTION_CLASSES, EMOTION_DISPLAY_NAMES

        assert set(EMOTION_DISPLAY_NAMES) == set(EMOTION_CLASSES)
