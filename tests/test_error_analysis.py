"""The reporting half of training/emotion_en_deberta/error_analysis.py.

Only the prediction step needs torch, so everything here runs in the dev
group: arrays in, report out. The numbers are small enough to verify by hand,
which is the point - a report generator that is itself unverified would be the
same failure the script exists to fix.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from conftest import REPO

sys.path.insert(0, str(REPO / "training" / "emotion_en_deberta"))

import error_analysis as ea  # noqa: E402

A, D, F, J, N, S, U = range(7)  # the fixed LABELS order


def test_the_label_order_is_the_pipelines():
    assert ea.LABELS == ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


class TestConfusion:
    def test_rows_are_truth_and_columns_are_prediction(self):
        true_ids = np.array([A, A, A, J])
        pred_ids = np.array([A, D, D, J])
        matrix = ea.confusion(true_ids, pred_ids)

        assert matrix[A, A] == 1
        assert matrix[A, D] == 2, "two angers predicted as disgust belong at [anger, disgust]"
        assert matrix[D, A] == 0, "and nothing at the transpose"
        assert matrix[J, J] == 1
        assert matrix.sum() == 4

    def test_a_perfect_model_is_diagonal(self):
        ids = np.array([A, D, F, J, N, S, U])
        matrix = ea.confusion(ids, ids)
        assert (matrix == np.eye(7, dtype=np.int64)).all()


class TestTopConfusions:
    def test_it_names_where_a_classs_errors_went(self):
        # 10 neutrals: 4 right, 5 called joy, 1 called fear.
        true_ids = np.array([N] * 10)
        pred_ids = np.array([N] * 4 + [J] * 5 + [F])
        top = ea.top_confusions(ea.confusion(true_ids, pred_ids))

        assert [e["predicted"] for e in top["neutral"]] == ["joy", "fear"]
        assert top["neutral"][0]["count"] == 5
        assert top["neutral"][0]["share_of_class"] == pytest.approx(0.5)
        assert top["neutral"][1]["count"] == 1

    def test_a_class_with_no_errors_lists_none(self):
        ids = np.array([A, A, A])
        assert ea.top_confusions(ea.confusion(ids, ids))["anger"] == []

    def test_an_absent_class_does_not_divide_by_zero(self):
        ids = np.array([A, A])
        top = ea.top_confusions(ea.confusion(ids, ids))
        assert top["surprise"] == []


class TestLengthEffects:
    def test_it_separates_correct_from_incorrect(self):
        texts = ["one two three four", "five six", "a b c d e f", "g"]
        hit = np.array([True, False, True, False])
        out = ea.length_effects(texts, hit)

        assert out["correct_mean_words"] == pytest.approx((4 + 6) / 2)
        assert out["incorrect_mean_words"] == pytest.approx((2 + 1) / 2)
        assert out["correct_mean_chars"] == pytest.approx((18 + 11) / 2)

    def test_one_empty_group_reports_rather_than_dividing(self):
        out = ea.length_effects(["a", "b"], np.array([True, True]))
        assert "note" in out
        assert "correct_mean_words" not in out


class TestConfidenceEffects:
    def test_the_threshold_arithmetic(self):
        # Two kept correct, one kept wrong, one dropped wrong.
        confidence = np.array([0.9, 0.8, 0.6, 0.1])
        hit = np.array([True, True, False, False])
        out = ea.confidence_effects(confidence, hit)

        assert out["correct_mean"] == pytest.approx(0.85)
        assert out["incorrect_mean"] == pytest.approx(0.35)
        assert out["below_stage8_threshold"] == pytest.approx(0.25), "one row of four"
        assert out["accuracy_above_threshold"] == pytest.approx(2 / 3)
        # Of the two errors, one is confident enough for stage 8 to draw it.
        assert out["share_of_errors_above_threshold"] == pytest.approx(0.5)

    def test_a_flawless_run_reports_no_surviving_errors(self):
        out = ea.confidence_effects(np.array([0.9, 0.9]), np.array([True, True]))
        assert out["share_of_errors_above_threshold"] == 0.0
        assert np.isnan(out["incorrect_mean"])


class TestBuildReport:
    @staticmethod
    def _report() -> dict:
        texts = ["a short one", "considerably longer text here", "mid length text", "tiny"]
        true_ids = np.array([A, J, N, N])
        pred_ids = np.array([A, J, N, J])
        confidence = np.array([0.95, 0.91, 0.72, 0.31])
        return ea.build_report(
            texts, true_ids, pred_ids, confidence, {"checkpoint": "demo", "class_weights": "none"}
        )

    def test_the_headline_counts(self):
        report = self._report()
        assert report["n"] == 4
        assert report["errors"] == 1
        assert report["error_rate"] == pytest.approx(0.25)
        assert report["metrics"]["accuracy"] == pytest.approx(0.75)

    def test_metadata_is_carried_through(self):
        report = self._report()
        assert report["checkpoint"] == "demo"
        assert report["class_weights"] == "none"

    def test_it_is_json_serialisable(self):
        """numpy types leak into a dict and break json.dump three steps later."""
        json.dumps(self._report())


class TestRenderMarkdown:
    @staticmethod
    def _reports(count: int) -> list[dict]:
        texts = ["one two three", "four five", "six seven eight nine", "ten"]
        true_ids = np.array([A, J, N, N])
        out = []
        for i in range(count):
            pred_ids = np.array([A, J, N, J if i == 0 else N])
            out.append(
                ea.build_report(
                    texts,
                    true_ids,
                    pred_ids,
                    np.array([0.9, 0.9, 0.8, 0.4]),
                    {"checkpoint": f"ckpt-{i}", "class_weights": "none" if i == 0 else "balanced"},
                )
            )
        return out

    def test_one_checkpoint_renders(self):
        text = ea.render_markdown(self._reports(1))
        assert text.startswith("# Error analysis")
        assert "ckpt-0" in text
        assert text.endswith("\n")

    def test_two_checkpoints_share_the_comparison_tables(self):
        text = ea.render_markdown(self._reports(2))
        for name in ("ckpt-0", "ckpt-1"):
            assert name in text
        for heading in ("### Precision", "### Recall", "### F1"):
            assert heading in text
        # Every class gets a row, including ones with no support.
        for label in ea.LABELS:
            assert f"| {label} |" in text

    def test_it_states_that_these_are_in_distribution_numbers(self):
        """The caveat is the most important line in the document."""
        text = ea.render_markdown(self._reports(1))
        assert "upper" in text and "bound" in text
        assert "Twitter" in text

    def test_it_tells_the_reader_how_to_regenerate_it(self):
        text = ea.render_markdown(self._reports(1))
        assert "error_analysis.py" in text
        assert "--out-json" in text


class TestConfigFrom:
    def test_unknown_recorded_fields_are_dropped(self, tmp_path):
        """The trainer writes `metrics` into the same file; it is not a config field."""
        (tmp_path / "train_config.json").write_text(
            json.dumps(
                {
                    "output": str(tmp_path),
                    "seed": 7,
                    "val_fraction": 0.2,
                    "class_weights": "balanced",
                    "metrics": {"best_macro_f1": 0.5},
                    "a_field_added_next_year": True,
                }
            ),
            encoding="utf-8",
        )
        cfg = ea.config_from(tmp_path)
        assert cfg.seed == 7
        assert cfg.val_fraction == pytest.approx(0.2)
        assert cfg.class_weights == "balanced"

    def test_a_missing_record_says_so(self, tmp_path):
        with pytest.raises(SystemExit, match="did that run finish"):
            ea.config_from(tmp_path)


class TestSubsample:
    """`--limit` must sample the split, not slice the front of it.

    The first version took `val_idx[:limit]`. On the real corpus that is 25.9%
    Neutral against 3.2% in the full split, and it reported accuracy 0.7610
    for a checkpoint measured at 0.9202 - a number that looks like a broken
    model and is a broken subset.
    """

    def test_it_does_not_take_the_head(self):
        val_idx = np.arange(1000)
        chosen = ea.subsample(val_idx, 50, seed=42)
        assert len(chosen) == 50
        assert not np.array_equal(chosen, val_idx[:50])
        assert chosen.max() > 200, "a sample of 1000 should reach beyond the first fifth"

    def test_it_preserves_a_representative_mix(self):
        """The head of a source-ordered split is one source; a sample is not."""
        # 900 of class 0 then 100 of class 1, as the corpus is blocked.
        labels = np.array([0] * 900 + [1] * 100)
        val_idx = np.arange(1000)

        head_share = labels[val_idx[:100]].mean()
        sample_share = labels[ea.subsample(val_idx, 100, seed=0)].mean()

        assert head_share == 0.0, "the head misses the second source entirely"
        assert sample_share == pytest.approx(0.1, abs=0.06)

    def test_it_is_reproducible_for_a_given_seed(self):
        val_idx = np.arange(500)
        assert np.array_equal(ea.subsample(val_idx, 40, 7), ea.subsample(val_idx, 40, 7))
        assert not np.array_equal(ea.subsample(val_idx, 40, 7), ea.subsample(val_idx, 40, 8))

    def test_it_stays_sorted(self):
        chosen = ea.subsample(np.arange(300), 30, seed=1)
        assert np.array_equal(chosen, np.sort(chosen))

    def test_a_limit_past_the_end_returns_everything(self):
        val_idx = np.arange(10)
        assert np.array_equal(ea.subsample(val_idx, 999, seed=1), val_idx)

    def test_it_only_ever_returns_real_indices(self):
        val_idx = np.array([3, 9, 27, 81, 243])
        chosen = ea.subsample(val_idx, 3, seed=5)
        assert set(chosen.tolist()) <= set(val_idx.tolist())
        assert len(set(chosen.tolist())) == 3, "no duplicates"


class TestCorpusIsLoadedOnce:
    """Two checkpoints on the same corpus must not read it twice.

    Loading and validating 419,180 gzipped rows dominated a run whose GPU work
    is about a minute per model, and the duplicated banner in the log was the
    visible symptom.
    """

    @staticmethod
    def _cfg(tmp_path, corpus: str):
        from train import TrainConfig

        return TrainConfig(output=str(tmp_path), corpus=corpus)

    def test_a_repeated_config_hits_the_cache(self, tmp_path, monkeypatch):
        import pandas as pd

        calls = []

        def fake_load_frame(cfg):
            calls.append(cfg.corpus)
            return pd.DataFrame({"text": ["a", "b"], "label": ["joy", "anger"]})

        monkeypatch.setattr(ea, "load_frame", fake_load_frame)
        monkeypatch.setattr(ea, "_CORPUS_CACHE", {})

        cfg = self._cfg(tmp_path, "one.csv.gz")
        first = ea.corpus_for(cfg)
        second = ea.corpus_for(cfg)

        assert calls == ["one.csv.gz"], "the second call must not reload"
        assert first[0] == second[0] == ["a", "b"]

    def test_a_different_corpus_is_loaded_separately(self, tmp_path, monkeypatch):
        import pandas as pd

        calls = []

        def fake_load_frame(cfg):
            calls.append(cfg.corpus)
            return pd.DataFrame({"text": ["a"], "label": ["joy"]})

        monkeypatch.setattr(ea, "load_frame", fake_load_frame)
        monkeypatch.setattr(ea, "_CORPUS_CACHE", {})

        ea.corpus_for(self._cfg(tmp_path, "one.csv.gz"))
        ea.corpus_for(self._cfg(tmp_path, "two.csv.gz"))

        assert calls == ["one.csv.gz", "two.csv.gz"]


class TestThresholdSweep:
    """Stage 8's 0.25 gate keeps 99.97% of predictions and 99.6% of errors.

    Measured, it is not quality control. The sweep exists so the number can be
    chosen from a curve rather than inherited.
    """

    # Four correct at high confidence, two wrong at middling confidence.
    CONF = np.array([0.99, 0.95, 0.80, 0.55, 0.65, 0.30])
    HIT = np.array([True, True, True, True, False, False])

    def _row(self, threshold: float) -> dict:
        rows = ea.threshold_sweep(self.CONF, self.HIT)
        return next(r for r in rows if r["threshold"] == pytest.approx(threshold))

    def test_it_reports_every_declared_threshold(self):
        rows = ea.threshold_sweep(self.CONF, self.HIT)
        assert [r["threshold"] for r in rows] == list(ea.SWEEP)

    def test_stage_8s_gate_is_first(self):
        assert ea.SWEEP[0] == 0.25, "the table should open with what the pipeline does today"

    def test_a_gate_below_everything_changes_nothing(self):
        """The real finding in miniature: 0.25 is under every prediction here."""
        row = self._row(0.25)
        assert row["coverage"] == 1.0, "the lowest confidence in the fixture is 0.30"
        assert row["accuracy_kept"] == pytest.approx(4 / 6)
        assert row["errors_removed"] == 0.0
        assert row["correct_removed"] == 0.0

    def test_a_useful_gate_removes_errors_faster_than_correct_answers(self):
        row = self._row(0.60)
        # Keeps 0.99, 0.95, 0.80, 0.65 -> three correct and one wrong.
        assert row["coverage"] == pytest.approx(4 / 6)
        assert row["accuracy_kept"] == pytest.approx(3 / 4)
        assert row["errors_removed"] == pytest.approx(0.5), "the 0.30 error goes"
        assert row["correct_removed"] == pytest.approx(0.25), "and one correct with it"

    def test_a_high_gate_costs_more_than_it_saves(self):
        row = self._row(0.95)
        assert row["errors_removed"] == 1.0
        assert row["correct_removed"] == pytest.approx(0.5), "half the correct answers too"

    def test_a_flawless_run_removes_no_errors(self):
        rows = ea.threshold_sweep(np.array([0.9, 0.9]), np.array([True, True]))
        assert all(r["errors_removed"] == 0.0 for r in rows)
