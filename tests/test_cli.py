"""CLI tests.

``vea config`` and ``vea models`` must work with no torch and no weights - that
is the point of the preflight check, and it is what CI exercises.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from vea.cli import _force_utf8_output, main


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


class TestConsoleEncoding:
    """A cp1252 console must not be able to kill a run.

    The pipeline banner prints `1 -> 2 -> ...` with U+2192, and from stage 4
    onward it prints Russian transcript text. Neither is encodable in cp1252,
    which is what `sys.stdout` is on a Western-European Windows install, so the
    first one reached ended the run with UnicodeEncodeError after the download
    and transcription had already been paid for.

    Reproduced here with PYTHONIOENCODING rather than a real Windows console,
    so it runs on all three platforms: the bug is the stream's encoding, not
    the operating system.
    """

    ARROW = "→"
    CYRILLIC = "Привет"  # "Privet"

    @staticmethod
    def _run(code: str, encoding: str) -> subprocess.CompletedProcess:
        # sys.executable, not "python": on Windows a bare `python` can resolve
        # to the launcher stub rather than to this environment's interpreter.
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            env={**os.environ, "PYTHONIOENCODING": encoding},
            check=False,
        )

    def test_the_stage_arrow_survives_a_cp1252_stream(self):
        code = (
            "from vea.cli import _force_utf8_output\n"
            "_force_utf8_output()\n"
            f"print('1 {self.ARROW} 2')\n"
        )
        result = self._run(code, "cp1252")
        assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
        assert self.ARROW.encode("utf-8") in result.stdout

    def test_russian_transcript_text_survives_a_cp1252_stream(self):
        code = (
            "from vea.cli import _force_utf8_output\n"
            "_force_utf8_output()\n"
            f"print('{self.CYRILLIC}')\n"
        )
        result = self._run(code, "cp1252")
        assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
        assert self.CYRILLIC.encode("utf-8") in result.stdout

    def test_without_the_fix_the_same_print_fails(self):
        """The guard above is only meaningful if the bug is real."""
        result = self._run(f"print('1 {self.ARROW} 2')", "cp1252")
        assert result.returncode != 0
        assert b"UnicodeEncodeError" in result.stderr

    def test_a_stream_without_reconfigure_is_tolerated(self, monkeypatch):
        """pytest's capture replaces the streams with objects that lack it."""
        monkeypatch.setattr(sys, "stdout", object())
        monkeypatch.setattr(sys, "stderr", object())
        _force_utf8_output()
