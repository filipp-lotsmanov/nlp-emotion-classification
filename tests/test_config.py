"""Tests for the settings and model registry.

These run without torch, transformers or any model download, so they work in CI
on a CPU-only runner.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import REPO
from vea.config import (
    MODEL_REGISTRY,
    TRANSLATION_MODELS,
    MissingCheckpointError,
    ModelSpec,
    Settings,
    apply_device_setting,
    missing_checkpoints,
    resolve_model,
    translation_model_ref,
)


class TestSettings:
    def test_defaults_are_repo_relative(self):
        settings = Settings.from_env(env={})
        assert settings.data_dir.name == "data"
        assert settings.models_dir.name == "models"
        assert settings.device == "auto"

    def test_env_overrides(self):
        settings = Settings.from_env(
            env={
                "VEA_DATA_DIR": "/scratch/vea-data",
                "VEA_MODELS_DIR": "/scratch/weights",
                "VEA_DEVICE": "cuda:1",
                "VEA_LOG_LEVEL": "debug",
            }
        )
        # Compare Path objects, not their strings: str(Path("/scratch/x")) is
        # "\\scratch\\x" on Windows, so a string comparison here asserts the
        # host's separator rather than the behaviour under test.
        assert settings.data_dir == Path("/scratch/vea-data")
        assert settings.models_dir == Path("/scratch/weights")
        assert settings.device == "cuda:1"
        assert settings.log_level == "DEBUG"

    def test_video_dir_matches_original_layout(self):
        # The stage modules glob for files inside "video-{id}" directories, so
        # this naming is load-bearing, not cosmetic.
        settings = Settings.from_env(env={"VEA_DATA_DIR": "/tmp/d"})
        assert settings.video_dir("mqGSkDFeLEo").name == "video-mqGSkDFeLEo"


class TestDeviceSetting:
    """`VEA_DEVICE` used to be read, printed by `vea config`, and then ignored:
    every stage defaulted to cuda-if-available whatever it said."""

    @staticmethod
    def _configs() -> dict:
        # The shapes DEFAULT_CONFIGS uses: a flat stage, and 7B's list of models.
        return {
            "stage_4": {"device": None, "model_size": "large-v3"},
            "stage_7b": {"models": [{"config": {"device": None}}, {"config": {"device": None}}]},
            "stage_1": {"base_output": "downloads"},
        }

    def test_an_explicit_setting_reaches_every_stage(self):
        configs = self._configs()
        assert apply_device_setting(configs, "cpu") == "cpu"
        assert configs["stage_4"]["device"] == "cpu"
        assert [m["config"]["device"] for m in configs["stage_7b"]["models"]] == ["cpu", "cpu"]

    def test_stages_without_a_device_are_not_given_one(self):
        configs = self._configs()
        apply_device_setting(configs, "cpu")
        assert "device" not in configs["stage_1"]

    def test_a_device_the_caller_set_is_kept(self):
        configs = self._configs()
        configs["stage_4"]["device"] = "cuda:1"
        apply_device_setting(configs, "cpu")
        assert configs["stage_4"]["device"] == "cuda:1"

    def test_auto_leaves_each_stage_to_choose(self):
        # Resolving "auto" here would pin stage 5B to the first GPU instead of
        # the one with the most free memory.
        configs = self._configs()
        assert apply_device_setting(configs, "auto") is None
        assert configs == self._configs()


class TestRegistry:
    def test_every_spec_is_fully_declared(self):
        for name, spec in MODEL_REGISTRY.items():
            assert isinstance(spec, ModelSpec), name
            assert spec.kind in {"hub", "local"}, name
            assert spec.ref, name
            assert spec.purpose, name
            assert spec.provenance, name

    def test_local_refs_are_relative(self):
        # A local ref is joined onto settings.models_dir. An absolute ref (the
        # original code had Windows paths such as A:\git\...) would silently
        # escape the configured model directory.
        for name, spec in MODEL_REGISTRY.items():
            if spec.is_local:
                assert not spec.ref.startswith(("/", "\\")), name
                assert ":" not in spec.ref, name

    def test_hub_models_resolve_to_their_id(self):
        assert resolve_model("nllb-3.3b") == "facebook/nllb-200-3.3B"
        assert (
            resolve_model("emotion-en-distilroberta")
            == "j-hartmann/emotion-english-distilroberta-base"
        )

    def test_unknown_model_lists_the_alternatives(self):
        with pytest.raises(KeyError, match="Unknown model"):
            resolve_model("xlmroberta-large-va")  # old hardcoded path, not a key

    def test_missing_local_checkpoint_names_the_training_script(self, tmp_path):
        settings = Settings.from_env(env={"VEA_MODELS_DIR": str(tmp_path)})
        with pytest.raises(MissingCheckpointError) as excinfo:
            resolve_model("emotion-en-deberta", settings=settings)
        message = str(excinfo.value)
        assert "training/emotion_en_deberta/train.py" in message
        assert str(tmp_path) in message

    def test_present_local_checkpoint_resolves_to_a_path(self, tmp_path):
        spec = MODEL_REGISTRY["emotion-en-deberta"]
        checkpoint = tmp_path / spec.ref
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text("{}", encoding="utf-8")

        settings = Settings.from_env(env={"VEA_MODELS_DIR": str(tmp_path)})
        assert resolve_model("emotion-en-deberta", settings=settings) == str(checkpoint)

    def test_a_directory_without_config_json_counts_as_missing(self, tmp_path):
        # An interrupted download or a partially copied checkpoint leaves the
        # directory in place. Existence of the directory alone is not enough.
        spec = MODEL_REGISTRY["va-xlmroberta-large"]
        (tmp_path / spec.ref).mkdir(parents=True)
        settings = Settings.from_env(env={"VEA_MODELS_DIR": str(tmp_path)})

        assert "va-xlmroberta-large" in missing_checkpoints(settings)
        with pytest.raises(MissingCheckpointError):
            resolve_model("va-xlmroberta-large", settings=settings)


class TestMissingCheckpoints:
    def test_reports_all_local_models_on_an_empty_models_dir(self, tmp_path):
        settings = Settings.from_env(env={"VEA_MODELS_DIR": str(tmp_path)})
        missing = missing_checkpoints(settings)
        local = {n for n, s in MODEL_REGISTRY.items() if s.is_local}
        assert set(missing) == local

    def test_never_reports_hub_models(self, tmp_path):
        settings = Settings.from_env(env={"VEA_MODELS_DIR": str(tmp_path)})
        for name in missing_checkpoints(settings):
            assert MODEL_REGISTRY[name].is_local


class TestTranslationModelSelection:
    """Stage 5B must be able to load something other than 17 GB of weights.

    Three NLLB sizes were declared in the registry and two were unreachable:
    the 3.3B Hub id was written straight into DEFAULT_CONFIGS. Same defect
    class as `--structural-only` and `serve --reload` - declared, documented,
    and wired to nothing.
    """

    def test_the_default_is_unchanged(self):
        assert Settings().translation_model == "nllb-3.3b"
        assert translation_model_ref(Settings()) == "facebook/nllb-200-3.3B"

    @pytest.mark.parametrize(
        ("name", "ref"),
        [
            ("nllb-3.3b", "facebook/nllb-200-3.3B"),
            ("nllb-1.3b", "facebook/nllb-200-1.3B"),
            ("nllb-600m", "facebook/nllb-200-distilled-600M"),
        ],
    )
    def test_every_declared_size_resolves(self, name, ref):
        assert translation_model_ref(Settings(translation_model=name)) == ref

    def test_the_environment_variable_is_read(self):
        settings = Settings.from_env({"VEA_TRANSLATION_MODEL": "nllb-600m"})
        assert translation_model_ref(settings) == "facebook/nllb-200-distilled-600M"

    def test_an_unknown_name_raises_rather_than_downloading_the_default(self):
        with pytest.raises(ValueError, match="not a translation model"):
            translation_model_ref(Settings(translation_model="nllb-200m"))

    def test_every_selectable_model_is_in_the_registry(self):
        for name in TRANSLATION_MODELS:
            assert name in MODEL_REGISTRY, f"{name} is selectable but not declared"
            assert MODEL_REGISTRY[name].kind == "hub"

    def test_the_pipeline_default_config_uses_the_selector(self):
        """Not the literal id, or the variable would do nothing."""
        source = (REPO / "src" / "vea" / "pipeline.py").read_text(encoding="utf-8")
        block = source[source.index('"stage_5b_translation"') :][:600]
        assert "translation_model_ref()" in block
        assert "facebook/nllb-200-3.3B" not in block
