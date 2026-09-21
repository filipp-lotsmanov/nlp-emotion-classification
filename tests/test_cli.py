"""CLI tests.

``vea config`` and ``vea models`` must work with no torch and no weights - that
is the point of the preflight check, and it is what CI exercises.
"""

from __future__ import annotations

import pytest

from vea.cli import main


@pytest.fixture
def empty_models_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("VEA_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("VEA_DATA_DIR", str(tmp_path / "data"))
    return tmp_path


def test_config_reports_resolved_settings(empty_models_dir, capsys):
    assert main(["config"]) == 0
    out = capsys.readouterr().out
    assert "VEA_MODELS_DIR" in out
    assert str(empty_models_dir) in out


def test_models_exits_nonzero_when_checkpoints_are_missing(empty_models_dir, capsys):
    # Non-zero is intentional: a setup script or CI job can gate on it.
    assert main(["models"]) == 1
    out = capsys.readouterr().out
    assert "MISSING" in out
    assert "va-xlmroberta-large" in out
    assert "training/README.md" in out


def test_models_exits_zero_once_checkpoints_exist(empty_models_dir, capsys):
    from vea.config import MODEL_REGISTRY

    for spec in MODEL_REGISTRY.values():
        if spec.is_local:
            path = empty_models_dir / spec.ref
            path.mkdir(parents=True)
            (path / "config.json").write_text("{}", encoding="utf-8")

    assert main(["models"]) == 0
    assert "All declared local checkpoints are present." in capsys.readouterr().out


def test_run_refuses_to_start_without_weights(empty_models_dir, capsys):
    # Must fail before importing the pipeline, so this test needs no torch.
    assert main(["run", "https://www.youtube.com/watch?v=mqGSkDFeLEo"]) == 1
    out = capsys.readouterr().out
    assert "Refusing to start" in out
    assert "--allow-missing-models" in out


def test_run_ignores_a_checkpoint_no_stage_loads(empty_models_dir):
    """Only the models a stage names can block a run.

    `emotion-ru-finetuned` is in the registry for its provenance; stage 7A loads
    a hub checkpoint. Refusing over it was a real defect - the message listed it
    and then said stage 7A would run fine without it.
    """
    from vea.config import MODEL_REGISTRY, STAGE_MODELS

    for name, spec in MODEL_REGISTRY.items():
        if spec.is_local and name in STAGE_MODELS:
            path = empty_models_dir / spec.ref
            path.mkdir(parents=True, exist_ok=True)
            (path / "config.json").write_text("{}", encoding="utf-8")

    # Every stage model is present and only the unreferenced one is absent, so
    # the preflight must pass. It then fails on the URL check below rather than
    # on weights, which is what proves it got past the checkpoint gate without
    # needing torch installed to find out.
    assert main(["run", "not-a-youtube-url"]) == 2


class TestUrlValidation:
    """The URL is checked before the pipeline - and its models - are loaded.

    The orchestrator initialises every model before stage 1 runs, so an
    unparseable URL used to cost a full NLLB-3.3B load first. These all return
    2, and none of them import torch.
    """

    @pytest.fixture(autouse=True)
    def _weights_present(self, empty_models_dir):
        from vea.config import MODEL_REGISTRY

        for spec in MODEL_REGISTRY.values():
            if spec.is_local:
                path = empty_models_dir / spec.ref
                path.mkdir(parents=True, exist_ok=True)
                (path / "config.json").write_text("{}", encoding="utf-8")

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.youtube.com/watch?v=REAL_ID",  # the placeholder, pasted literally
            "https://www.youtube.com/watch?v=<youtube-url>",
            "https://www.youtube.com/watch?v=tooshort",
            "https://vimeo.com/123456789",
            "not-a-url-at-all",
            "",
        ],
    )
    def test_unparseable_urls_are_refused(self, url, capsys):
        assert main(["run", url]) == 2
        assert "Not a YouTube URL this pipeline can parse" in capsys.readouterr().out

    def test_a_placeholder_says_so(self, capsys):
        main(["run", "https://www.youtube.com/watch?v=REAL_ID"])
        assert "looks like a placeholder" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.youtube.com/watch?v=mqGSkDFeLEo",
            "https://youtu.be/mqGSkDFeLEo",
            "https://www.youtube.com/embed/mqGSkDFeLEo",
            "https://www.youtube.com/watch?t=42&v=mqGSkDFeLEo",
        ],
    )
    def test_real_urls_get_past_the_check(self, url):
        # They must not be rejected as unparseable. Whether the run then
        # succeeds needs torch and a network, so it is not asserted here - only
        # that the exit code is not the URL rejection.
        from vea.config import extract_youtube_id

        assert extract_youtube_id(url) == "mqGSkDFeLEo"


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit):
        main(["frobnicate"])
