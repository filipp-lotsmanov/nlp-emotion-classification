"""Tests for the VA trainer's pure logic.

Deliberately covers only what can be checked without torch, transformers or a
model download: target scaling, metric computation, model selection, and corpus
loading/validation. Those are also where a silent, plausible-looking error is
most likely - a wrong scale mapping or a collapsed head produces a timeline that
renders perfectly and is wrong.

The training loop itself needs a GPU and a corpus; `scripts/train_va.sh --smoke`
is the end-to-end check for that.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import load_training_module

# Loaded by path rather than by import name: both trainers are called train.py,
# so `from train import ...` resolves to whichever test module ran first. See
# tests/conftest.py.
_train = load_training_module("va_regressor")

LABELS = _train.LABELS
TrainConfig = _train.TrainConfig
build_smoke_frame = _train.build_smoke_frame
load_va_frame = _train.load_va_frame
normalize_targets = _train.normalize_targets
regression_metrics = _train.regression_metrics
summarize = _train.summarize


class TestNormalizeTargets:
    def test_maps_declared_scale_onto_unit_interval(self):
        out = normalize_targets(np.array([1.0, 5.0, 9.0]), 1.0, 9.0)
        assert out.tolist() == pytest.approx([0.0, 0.5, 1.0])

    def test_handles_a_one_to_five_scale(self):
        # EmoBank is 1-5; using the 1-9 default on it would compress everything
        # into the lower half of the range and nothing would error.
        out = normalize_targets(np.array([1.0, 3.0, 5.0]), 1.0, 5.0)
        assert out.tolist() == pytest.approx([0.0, 0.5, 1.0])

    def test_handles_a_signed_scale(self):
        out = normalize_targets(np.array([-1.0, 0.0, 1.0]), -1.0, 1.0)
        assert out.tolist() == pytest.approx([0.0, 0.5, 1.0])

    def test_clips_rather_than_rescaling_outliers(self):
        # One mis-parsed row must not shift the whole distribution.
        out = normalize_targets(np.array([-5.0, 5.0, 99.0]), 1.0, 9.0)
        assert out[0] == 0.0
        assert out[2] == 1.0
        assert out[1] == pytest.approx(0.5)

    def test_rejects_an_inverted_or_degenerate_scale(self):
        with pytest.raises(ValueError, match="must be greater than"):
            normalize_targets(np.array([1.0]), 9.0, 1.0)
        with pytest.raises(ValueError, match="must be greater than"):
            normalize_targets(np.array([1.0]), 5.0, 5.0)


class TestRegressionMetrics:
    def test_perfect_prediction(self):
        true = np.linspace(0, 1, 50)
        m = regression_metrics(true, true)
        assert m["pearson"] == pytest.approx(1.0)
        assert m["spearman"] == pytest.approx(1.0)
        assert m["mse"] == pytest.approx(0.0)

    def test_inverted_prediction_is_negatively_correlated(self):
        true = np.linspace(0, 1, 50)
        m = regression_metrics(1 - true, true)
        assert m["pearson"] == pytest.approx(-1.0)

    def test_collapsed_head_is_reported_not_hidden(self):
        # The failure mode that matters: a head that predicts the mean has a
        # respectable MSE but zero information. pred_std makes it visible and
        # correlation is undefined rather than silently nan-from-scipy.
        true = np.linspace(0, 1, 100)
        pred = np.full_like(true, 0.5)
        m = regression_metrics(pred, true)
        assert m["pred_std"] == pytest.approx(0.0)
        assert math.isnan(m["pearson"])
        assert m["mse"] < 0.1  # looks fine on error alone

    def test_mae_and_mse_are_distinct(self):
        m = regression_metrics(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        assert m["mse"] == pytest.approx(0.5)
        assert m["mae"] == pytest.approx(0.5)


class TestSummarize:
    def test_averages_pearson_across_dimensions(self):
        report = {"valence": {"pearson": 0.8}, "arousal": {"pearson": 0.6}}
        assert summarize(report) == pytest.approx(0.7)

    def test_a_fully_collapsed_report_sorts_below_any_real_score(self):
        collapsed = {dim: {"pearson": float("nan")} for dim in LABELS}
        real = {dim: {"pearson": -0.9} for dim in LABELS}
        assert summarize(collapsed) < summarize(real)

    def test_ignores_a_single_nan_dimension(self):
        report = {"valence": {"pearson": 0.5}, "arousal": {"pearson": float("nan")}}
        assert summarize(report) == pytest.approx(0.5)


class TestSmokeFrame:
    def test_is_bilingual_and_has_the_expected_columns(self):
        frame = build_smoke_frame(n_per_template=5, seed=1)
        assert {"text", "lang", "valence", "arousal"} <= set(frame.columns)
        assert set(frame["lang"]) == {"en", "ru"}

    def test_targets_span_the_va_plane(self):
        # If the synthetic data did not cover all four quadrants, a passing
        # smoke run would prove much less than it appears to.
        frame = build_smoke_frame(n_per_template=5, seed=1)
        assert frame["valence"].min() < 3 and frame["valence"].max() > 7
        assert frame["arousal"].min() < 3 and frame["arousal"].max() > 7

    def test_is_deterministic_for_a_seed(self):
        a = build_smoke_frame(n_per_template=4, seed=7)
        b = build_smoke_frame(n_per_template=4, seed=7)
        assert a["text"].tolist() == b["text"].tolist()


class TestLoadVaFrame:
    def _cfg(self, dataset: str, **kw) -> TrainConfig:
        return TrainConfig(dataset=dataset, output="unused", **kw)

    def test_loads_the_smoke_dataset(self):
        frame = load_va_frame(self._cfg("smoke"))
        assert len(frame) > 100

    def test_loads_a_csv_and_reports_a_missing_column_usefully(self, tmp_path):
        import pandas as pd

        path = tmp_path / "va.csv"
        pd.DataFrame({"sentence": ["a"] * 60, "V": [5.0] * 60, "A": [5.0] * 60}).to_csv(
            path, index=False
        )

        frame = load_va_frame(
            self._cfg(str(path), text_col="sentence", valence_col="V", arousal_col="A")
        )
        assert len(frame) == 60

        with pytest.raises(ValueError, match="missing column"):
            load_va_frame(self._cfg(str(path)))  # default column names

    def test_drops_null_and_empty_text_rows(self, tmp_path):
        import pandas as pd

        path = tmp_path / "va.csv"
        rows = {
            "text": ["ok"] * 60 + ["", "   ", None],
            "valence": [5.0] * 63,
            "arousal": [5.0] * 63,
        }
        pd.DataFrame(rows).to_csv(path, index=False)
        assert len(load_va_frame(self._cfg(str(path)))) == 60

    def test_refuses_a_corpus_too_small_to_evaluate(self, tmp_path):
        import pandas as pd

        path = tmp_path / "tiny.csv"
        pd.DataFrame({"text": ["a", "b"], "valence": [1.0, 2.0], "arousal": [1.0, 2.0]}).to_csv(
            path, index=False
        )
        with pytest.raises(ValueError, match="too few"):
            load_va_frame(self._cfg(str(path)))

    def test_missing_file_names_the_accepted_forms(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="hf:"):
            load_va_frame(self._cfg(str(tmp_path / "nope.csv")))

    def test_unsupported_extension_is_rejected(self, tmp_path):
        path = tmp_path / "corpus.xlsx"
        path.write_text("not really a spreadsheet", encoding="utf-8")
        with pytest.raises(ValueError, match="unsupported corpus format"):
            load_va_frame(self._cfg(str(path)))

    def test_limit_truncates(self, tmp_path):
        import pandas as pd

        path = tmp_path / "va.csv"
        pd.DataFrame(
            {
                "text": [f"row {i}" for i in range(500)],
                "valence": [5.0] * 500,
                "arousal": [5.0] * 500,
            }
        ).to_csv(path, index=False)
        assert len(load_va_frame(self._cfg(str(path), limit=120))) == 120


class TestContractConstants:
    def test_label_order_matches_what_inference_indexes(self):
        # intensity_ru.py reads predictions[0, 0] as valence and [0, 1] as
        # arousal. If this list is ever reordered, every timeline silently
        # swaps its two panels.
        assert LABELS == ["valence", "arousal"]
