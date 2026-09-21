"""Static guards against the import-time side effects that broke portability.

The archive had four stage modules setting ``CUDA_VISIBLE_DEVICES = '5'`` and
thirteen calling ``logging.basicConfig`` at import time. Both are invisible in
normal use and both are easy to reintroduce, so they are checked here by AST
scan rather than by running the pipeline. No heavy dependency is imported: the
source is parsed, never executed.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "vea"
PIPELINE = SRC / "pipeline.py"
SOURCE_FILES = sorted(SRC.rglob("*.py"))

# Files the encoding guard applies to: the package, the tests, and the training
# entry points written for this repository. It deliberately excludes
# training/{baselines,translation,emotion_ru,interpretability}, which are the
# group's migrated coursework - seven of those files do perform unencoded text
# I/O, and rewriting them for style is the same trade this repo already declined
# in the ruff per-file-ignores.
#
# Runtime detection via -W error::EncodingWarning was tried here and removed: it
# fires inside third-party packages (dill, pulled in by datasets) and is silent
# when those are absent, so it failed for reasons unrelated to this repository
# while missing the case it exists for - which was a test file of ours that read
# source without an encoding and broke only on Windows.
MAINTAINED_TRAINING = (
    REPO / "training" / "va_regressor",
    REPO / "training" / "emotion_en_deberta",
)
OWNED_FILES = sorted(
    [p for d in (SRC, REPO / "tests", *MAINTAINED_TRAINING) for p in d.rglob("*.py")]
    + [REPO / "training" / "verify_checkpoint.py"]
)


def _module_level_nodes(tree: ast.Module):
    """Statements that execute at import time (module body, not inside defs)."""
    return tree.body


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_there_are_source_files_to_check():
    # Guards against the glob silently matching nothing and every test passing.
    assert len(SOURCE_FILES) >= 14


def test_no_top_level_directory_shadows_a_dependency():
    """A repo-root directory must not be named after an importable dependency.

    Python puts the current directory first on `sys.path`, so a directory at the
    repo root shadows any installed distribution of the same name for anything
    run from there - and because a directory without `__init__.py` is a valid
    namespace package, the import *succeeds* and fails later on the attribute:

        ImportError: cannot import name 'load_dataset' from 'datasets'
                     (unknown location)

    That is what a `datasets/` directory here did to Hugging Face `datasets`,
    which `training/va_regressor/train.py` needs for `--dataset hf:<id>` and the
    baselines need for GoEmotions. The corpora live in `corpora/` for this
    reason. The check is name-based rather than import-based so it works in the
    fast CI job, where most of these packages are not installed.
    """
    reserved = {
        "datasets",
        "transformers",
        "torch",
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "matplotlib",
        "whisper",
        "typer",
        "rich",
        "yaml",
    }
    offenders = sorted(
        path.name for path in REPO.iterdir() if path.is_dir() and path.name in reserved
    )
    assert not offenders, (
        f"repo-root directories shadow importable packages: {offenders}. "
        "Rename them - a same-named directory becomes a namespace package and "
        "silently wins over the installed distribution for anything run from "
        "the repository root."
    )


def _writes_to_environ(tree: ast.Module, key: str) -> bool:
    """True if the module assigns to ``os.environ[key]`` or setdefault()s it.

    Checked against the AST rather than the raw text, so documentation that
    mentions the anti-pattern (config.py explains why it was removed) does not
    trip the guard.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == key
                ):
                    return True
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"setdefault", "putenv"}
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == key
        ):
            return True
    return False


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_no_module_sets_cuda_visible_devices(path: Path):
    assert not _writes_to_environ(_parse(path), "CUDA_VISIBLE_DEVICES"), (
        f"{path.name} pins CUDA_VISIBLE_DEVICES. Device selection belongs in "
        "vea.config.resolve_device(), driven by VEA_DEVICE."
    )


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_only_the_cli_configures_logging(path: Path):
    tree = _parse(path)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "basicConfig"
    ]
    if path.name == "config.py":
        # configure_logging() is the one sanctioned caller, inside a function.
        assert len(calls) == 1
        return
    assert not calls, (
        f"{path.name} calls logging.basicConfig. Library modules must only ever "
        "call logging.getLogger(__name__); the entry point configures handlers."
    )


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_no_module_exits_the_interpreter_at_import_time(path: Path):
    """``sys.exit`` inside a module body turns a bad install into a silent exit."""
    tree = _parse(path)
    for node in _module_level_nodes(tree):
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                for stmt in ast.walk(ast.Module(body=handler.body, type_ignores=[])):
                    if (
                        isinstance(stmt, ast.Call)
                        and isinstance(stmt.func, ast.Attribute)
                        and stmt.func.attr == "exit"
                    ):
                        pytest.fail(
                            f"{path.name} calls sys.exit() from an import-time "
                            "except block; let the ImportError propagate."
                        )


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_no_absolute_developer_paths(path: Path):
    """The archive carried `A:\\git\\...` and `C:\\Users\\...` paths in 12 files."""
    source = path.read_text(encoding="utf-8")
    for marker in ("C:\\Users", "A:\\git", "/home/y2a", "/content/drive"):
        assert marker not in source, f"{path.name} contains a machine-specific path: {marker}"


@pytest.mark.parametrize("path", OWNED_FILES, ids=lambda p: str(p.name))
def test_text_file_io_declares_an_encoding(path: Path):
    """Text I/O must name its encoding, because the default is host-dependent.

    ``open(p)``, ``p.read_text()`` and ``p.write_text(s)`` use
    ``locale.getpreferredencoding()`` - UTF-8 on the Linux server this pipeline
    was developed on, but cp1252 on a Western-European Windows install. Since
    every transcription, segment and emotion JSON file this pipeline writes
    contains Russian text, an unencoded read is a guaranteed
    UnicodeDecodeError on Windows and nowhere else.

    The stage modules already got this right - all 56 of their I/O calls pass
    ``encoding`` - which is precisely the kind of correctness that erodes
    silently under later edits, so it is pinned here. It covers tests/ and
    training/ too: the first version of the VA trainer's tests broke exactly
    this way and the failure only appeared on Windows.
    """
    tree = _parse(path)
    offenders: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        is_builtin_open = isinstance(node.func, ast.Name) and node.func.id == "open"
        is_path_text_io = isinstance(node.func, ast.Attribute) and node.func.attr in {
            "read_text",
            "write_text",
        }
        if not (is_builtin_open or is_path_text_io):
            continue

        keywords = {kw.arg for kw in node.keywords}
        if "encoding" in keywords:
            continue

        # Binary mode needs no encoding: open(p, "rb").
        if is_builtin_open and len(node.args) > 1:
            mode = node.args[1]
            if isinstance(mode, ast.Constant) and "b" in str(mode.value):
                continue

        name = "open" if is_builtin_open else node.func.attr
        offenders.append(f"line {node.lineno}: {name}()")

    assert not offenders, (
        f"{path.name} performs text I/O without an explicit encoding: "
        f"{', '.join(offenders)}. Pass encoding='utf-8' - the default is the "
        "host locale, which breaks on Windows for Cyrillic content."
    )


@lru_cache(maxsize=1)
def usable_bash() -> str | None:
    """A bash that actually runs a command, or None.

    Not `shutil.which("bash")`. On a Windows runner that finds
    C:\\Windows\\System32\\bash.exe - the WSL launcher - which, with no
    distribution installed, prints a UTF-16 notice and exits 1. A test that
    skipped on "bash is absent" therefore ran there and failed on output it
    never asked for. Probing behaviour catches that, and any other broken
    bash, without naming a platform.
    """
    path = shutil.which("bash")
    if path is None:
        return None
    try:
        probe = subprocess.run([path, "-c", "exit 7"], capture_output=True, timeout=30)
    except OSError:
        return None
    return path if probe.returncode == 7 else None


class TestPipelineWiring:
    """The orchestrator must only name models that the registry declares.

    ``vea.pipeline`` imports torch, so its config is read by parsing the source
    instead of importing the module. That keeps this check runnable in CI.
    """

    @staticmethod
    def _declared_model_keys() -> list[str]:
        tree = _parse(PIPELINE)
        keys: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values, strict=False):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "model"
                    and isinstance(value, ast.Constant)
                ):
                    keys.append(value.value)
        return keys

    def test_found_the_stage_model_declarations(self):
        # Five: 6A, 6B, 7A, and the two 7B ensemble members.
        assert len(self._declared_model_keys()) == 5

    def test_every_stage_model_is_in_the_registry(self):
        from vea.config import MODEL_REGISTRY

        for key in self._declared_model_keys():
            assert key in MODEL_REGISTRY, (
                f"pipeline.py declares model {key!r}, which is not in MODEL_REGISTRY"
            )

    def test_stage_models_matches_what_the_pipeline_declares(self):
        """`config.STAGE_MODELS` mirrors pipeline.py's `"model"` keys.

        It is hand-maintained, because importing vea.pipeline pulls in torch and
        the CLI has to stay cheap. So it is pinned here instead.

        What it decides: `vea run` refuses to start when a checkpoint a stage
        needs is absent, and ignores one no stage loads. Getting this set wrong
        either blocks a runnable pipeline - which it did, over
        `emotion-ru-finetuned`, a model stage 7A does not load - or starts a run
        that dies at stage 6.
        """
        from vea.config import MODEL_REGISTRY, STAGE_MODELS

        assert set(self._declared_model_keys()) == STAGE_MODELS, (
            "STAGE_MODELS has drifted from pipeline.py. Declared there: "
            f"{sorted(set(self._declared_model_keys()))}; STAGE_MODELS: {sorted(STAGE_MODELS)}"
        )
        # A stage naming a model outside the registry is caught by the test
        # above; this pins the other direction, that STAGE_MODELS is a subset.
        assert set(MODEL_REGISTRY) >= STAGE_MODELS

    def test_a_declared_but_unreferenced_model_does_not_block_a_run(self):
        """`emotion-ru-finetuned` is in the registry for provenance only.

        Stage 7A loads the hub checkpoint `emotion-ru-rubert-tiny2` instead, so
        the group's own Russian model being absent must not stop a run. This is
        the specific false alarm the split exists to prevent.
        """
        from vea.config import MODEL_REGISTRY, STAGE_MODELS

        assert "emotion-ru-finetuned" in MODEL_REGISTRY
        assert "emotion-ru-finetuned" not in STAGE_MODELS
        assert MODEL_REGISTRY["emotion-ru-finetuned"].is_local
        assert "emotion-ru-rubert-tiny2" in STAGE_MODELS

    def test_no_stage_carries_a_literal_checkpoint_path(self):
        source = PIPELINE.read_text(encoding="utf-8")
        for stale in ("models/xlmroberta-large-va", "models/checkpoint-3600"):
            # Allowed in comments explaining the migration, not as a config value.
            for line in source.splitlines():
                stripped = line.strip()
                if stale in stripped and not stripped.startswith("#"):
                    pytest.fail(f"pipeline.py still uses a literal path: {stripped}")


class TestDockerfileCudaPatch:
    """The CUDA 12 patch must not break the environment it patches.

    An earlier version ran, inside torch's own virtualenv:

        uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12

    nvidia-cudnn-cu12 installs `nvidia/cudnn/lib/libcudnn.so.9`, which is the
    exact path nvidia-cudnn-cu13 already owns and torch links at import. The
    install replaced it, and every run of the image died before stage 1 with

        ImportError: libcudnn.so.9: cannot open shared object file

    on machines that were never going to touch CUDA. The build had verified
    both sonames were in the loader cache and never verified that torch could
    still be imported.
    """

    @staticmethod
    def _dockerfile() -> str:
        return (REPO / "Dockerfile").read_text(encoding="utf-8")

    @classmethod
    def _install_lines(cls) -> list[str]:
        """`uv pip install` lines, comments excluded.

        The comment block above the step quotes the broken command verbatim so
        nobody reinstates it, which is exactly the string these tests look for.
        Scanning comments would fail on the warning against the defect.
        """
        return [
            line.strip()
            for line in cls._dockerfile().splitlines()
            if not line.strip().startswith("#") and "uv pip install" in line
        ]

    def test_cudnn_is_not_installed_over_torchs_own(self):
        """cuDNN 9 is cuDNN 9 for both CUDA majors; torch already ships it."""
        for line in self._install_lines():
            assert "nvidia-cudnn" not in line, (
                "the Dockerfile installs nvidia-cudnn, which collides with "
                f"torch's own copy at nvidia/cudnn/lib/: {line}"
            )

    def test_any_cuda_wheel_goes_to_its_own_prefix(self):
        """--target, so nothing can replace a file the venv already owns."""
        for line in self._install_lines():
            if "nvidia-" in line:
                assert "--target" in line, f"CUDA wheels must install to their own prefix: {line}"

    def test_the_build_verifies_that_torch_still_imports(self):
        """The check that would have caught it, and now will."""
        text = self._dockerfile()
        assert 'python -c "import torch' in text, (
            "the app target must import torch during the build; a build that "
            "cannot import torch is not a build worth pushing"
        )

    def test_the_loader_config_covers_both_prefixes(self):
        text = self._dockerfile()
        assert "/opt/cuda12/nvidia/*/lib" in text
        assert "/opt/venv/lib/python3*/site-packages/nvidia/*/lib" in text


class TestSetupGuideStaysTrue:
    """SETUP.md is the first thing a new user reads, so its commands must work.

    Every failure listed in its troubleshooting table is one this project
    actually hit. The risk is not that the prose ages badly - it is that a flag
    or a filename changes and the guide keeps confidently naming the old one.
    """

    @staticmethod
    def _setup() -> str:
        return (REPO / "SETUP.md").read_text(encoding="utf-8")

    def test_every_script_it_names_exists(self):
        import re

        named = set(re.findall(r"scripts/[a-z0-9_]+\.sh", self._setup()))
        assert named, "the guide names no scripts; this test is looking in the wrong place"
        for script in sorted(named):
            assert (REPO / script).is_file(), f"SETUP.md tells the reader to run a missing {script}"

    def test_every_compose_file_it_names_exists(self):
        for name in ("compose.yaml", "compose.gpu.yaml"):
            assert name in self._setup(), f"{name} is not mentioned in SETUP.md"
            assert (REPO / name).is_file()

    def test_the_build_argument_it_documents_is_real(self):
        assert "CUDA_RUNTIME" in (REPO / "Dockerfile").read_text(encoding="utf-8")
        assert "CUDA_RUNTIME" in self._setup()

    def test_the_translation_variable_it_documents_is_real(self):
        from vea.config import TRANSLATION_MODELS

        text = self._setup()
        assert "VEA_TRANSLATION_MODEL" in text
        for name in TRANSLATION_MODELS:
            # Every selectable size should be discoverable from the guide.
            assert name in text, f"SETUP.md does not mention {name}"

    def test_the_checkpoint_directories_match_the_registry(self):
        from vea.config import MODEL_REGISTRY, STAGE_MODELS

        text = self._setup()
        for name in sorted(STAGE_MODELS):
            spec = MODEL_REGISTRY[name]
            if spec.is_local:
                assert spec.ref in text, (
                    f"SETUP.md must name the directory {spec.ref!r} that {name} is loaded from"
                )


class TestTrainingWillNotClobberAProvenCheckpoint:
    """An experiment must not overwrite the model in production use.

    `--output` defaults to the directory MODEL_REGISTRY resolves
    `emotion-en-deberta` to. The script's own documented example for a
    class-weight experiment omitted it, so following the documentation would
    have replaced a checkpoint measured at macro F1 0.8162 with an untested
    variant - after three hours of GPU, and discoverable only afterwards.
    """

    SCRIPT = REPO / "scripts" / "train_emotion_en.sh"

    def test_the_documented_experiment_names_an_output(self):
        header = self.SCRIPT.read_text(encoding="utf-8").split("set -euo pipefail")[0]
        for line in header.splitlines():
            if "--class-weights" in line:
                assert "--output" in line or line.strip().endswith("\\"), (
                    f"documented experiment would overwrite the live checkpoint: {line}"
                )

    @pytest.mark.skipif(usable_bash() is None, reason="no working bash on this runner")
    def test_it_refuses_an_existing_checkpoint(self, tmp_path):
        models = tmp_path / "models"
        (models / "emotion-en-deberta").mkdir(parents=True)
        (models / "emotion-en-deberta" / "config.json").write_text("{}", encoding="utf-8")

        result = subprocess.run(
            [usable_bash(), str(self.SCRIPT)],
            capture_output=True,
            text=True,
            env={**os.environ, "VEA_MODELS_DIR": str(models), "VEA_DATA_DIR": str(tmp_path)},
            cwd=REPO,
        )
        assert result.returncode == 3, (result.returncode, result.stdout, result.stderr)
        assert "already holds a checkpoint" in result.stderr

    @pytest.mark.skipif(usable_bash() is None, reason="no working bash on this runner")
    def test_force_and_a_different_output_both_get_past_it(self, tmp_path):
        """Past the guard, not past the GPU check - which is the next thing."""
        models = tmp_path / "models"
        (models / "emotion-en-deberta").mkdir(parents=True)
        (models / "emotion-en-deberta" / "config.json").write_text("{}", encoding="utf-8")
        env = {**os.environ, "VEA_MODELS_DIR": str(models), "VEA_DATA_DIR": str(tmp_path)}

        for args in (["--force"], ["--output", str(models / "emotion-en-deberta-balanced")]):
            result = subprocess.run(
                [usable_bash(), str(self.SCRIPT), *args],
                capture_output=True,
                text=True,
                env=env,
                cwd=REPO,
            )
            assert "already holds a checkpoint" not in result.stderr, args
            assert result.returncode != 3, args
