"""
Central configuration for the video emotion analysis pipeline.

This module replaces three anti-patterns that were spread across the original
stage modules:

1. ``os.environ['CUDA_VISIBLE_DEVICES'] = '5'`` at import time in four modules.
   On any machine with fewer than six GPUs this hides every GPU, so the stages
   silently fell back to CPU instead of failing loudly. Device selection now
   goes through :func:`resolve_device` and is controlled by ``VEA_DEVICE``.

2. Hardcoded relative checkpoint paths (``models/xlmroberta-large-va``,
   ``./models/checkpoint-3600``) buried in per-module ``MODEL_CONFIGS`` dicts.
   Weights are now declared once in :data:`MODEL_REGISTRY` and resolved through
   :func:`resolve_model`, which raises an actionable error when a checkpoint is
   absent rather than a bare ``OSError`` from ``from_pretrained``.

3. ``logging.basicConfig`` called at import time in 13 library modules, which
   made log configuration depend on import order. Only the CLI configures
   logging now, via :func:`configure_logging`.

Nothing here imports torch or transformers at module level, so the settings and
registry can be inspected (and unit-tested) without a GPU stack installed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

ModelKind = Literal["hub", "local"]

# Repository root, resolved from this file: src/vea/config.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]


class MissingCheckpointError(FileNotFoundError):
    """Raised when a locally trained checkpoint is declared but not present."""


@dataclass(frozen=True)
class ModelSpec:
    """Declaration of one model the pipeline can load.

    Attributes:
        kind: ``"hub"`` for a Hugging Face Hub id that downloads on first use,
            ``"local"`` for a checkpoint directory that must exist on disk.
        ref: Hub id, or path relative to ``settings.models_dir`` for local kinds.
        purpose: Which pipeline stage consumes this model.
        provenance: How the weights were produced. For local checkpoints this
            names the training entry point that rebuilds them.
        license: License of the weights, needed for the repository's own
            license compliance (see docs/LICENSING.md).
        preprocess: Name of the text transform this model's *training data* went
            through, or None for raw text. Declared per model, not per stage,
            because stage 7B runs an ensemble whose members were trained on
            different text conventions: applying one member's normalisation to
            the other would create a fresh train/serve skew rather than remove
            one. Resolved by :func:`resolve_preprocess`.
    """

    kind: ModelKind
    ref: str
    purpose: str
    provenance: str
    license: str = "unknown"
    preprocess: str | None = None

    @property
    def is_local(self) -> bool:
        return self.kind == "local"


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Every model the pipeline touches is declared here exactly once. Stage modules
# must look models up by logical name instead of embedding paths.
#
# The two "local" entries are the checkpoints that are NOT in the archive. Until
# they are retrained (see training/README.md) stages 6A, 6B and the DeBERTa half
# of 7B cannot run, and resolve_model() will say so explicitly.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    # --- Stage 4: transcription -------------------------------------------
    "whisper-large-v3": ModelSpec(
        kind="hub",
        ref="large-v3",
        purpose="stage_4_transcription (faster-whisper)",
        provenance="Pretrained, downloaded by faster-whisper on first use.",
        license="MIT",
    ),
    # --- Stage 5A: semantic segmentation ----------------------------------
    "sentence-embeddings-multilingual": ModelSpec(
        kind="hub",
        ref="paraphrase-multilingual-MiniLM-L12-v2",
        purpose="stage_5a_scene_alignment (sentence-transformers)",
        provenance="Pretrained sentence-transformers model.",
        license="Apache-2.0",
    ),
    # --- Stage 5B: translation --------------------------------------------
    "nllb-3.3b": ModelSpec(
        kind="hub",
        ref="facebook/nllb-200-3.3B",
        purpose="stage_5b_translation (ru -> en)",
        provenance="Pretrained NLLB-200.",
        license="CC-BY-NC-4.0",
    ),
    "nllb-1.3b": ModelSpec(
        kind="hub",
        ref="facebook/nllb-200-1.3B",
        purpose="stage_5b_translation, lower-VRAM fallback",
        provenance="Pretrained NLLB-200.",
        license="CC-BY-NC-4.0",
    ),
    "nllb-600m": ModelSpec(
        kind="hub",
        ref="facebook/nllb-200-distilled-600M",
        purpose="stage_5b_translation, smallest fallback",
        provenance="Pretrained NLLB-200 distilled.",
        license="CC-BY-NC-4.0",
    ),
    # --- Stages 6A/6B: valence-arousal regression -------------------------
    "va-xlmroberta-large": ModelSpec(
        kind="local",
        ref="xlmroberta-base-va",
        purpose="stage_6a_russian_intensity and stage_6b_english_intensity",
        provenance=(
            "The published checkpoint, not a rebuild: gmendes9/multilingual_va_prediction "
            "(Mendes & Martins, ECIR 2023), XLM-RoBERTa trained on 34 psycho-linguistic "
            "datasets across 100 languages, so it reads Russian without translation. "
            "Fetch it with scripts/fetch_va_checkpoint.sh, which pins sha256 "
            "f75773cb738a8f279832b5dd8b24209c5b1c3c71d4eb09d97b2981ecc9041332. "
            "NOTE: this is the BASE checkpoint - the authors' large one is 2.09 GB and "
            "over GitHub's asset limit. The key name says 'large' because the archive's "
            "stage config did; the two were within noise of each other on the "
            "measurement that chose it. Its arousal output, which stage 6A thresholds "
            "on, separates this project's classes at AUC 0.5734 - see docs/PROVENANCE.md "
            "section 11 before reporting anything derived from it. "
            "training/va_regressor/train.py remains for training a replacement."
        ),
        license="MIT (mirror), see docs/LICENSING.md",
    ),
    # --- Stage 7A: Russian emotion ----------------------------------------
    "emotion-ru-rubert-tiny2": ModelSpec(
        kind="hub",
        ref="Djacon/rubert-tiny2-russian-emotion-detection",
        purpose="stage_7a_russian_emotion",
        provenance="Third-party pretrained model, used as-is.",
        license="MIT",
    ),
    "emotion-ru-finetuned": ModelSpec(
        kind="local",
        ref="emotion-ru-rubert",
        purpose="stage_7a_russian_emotion, group's own model (not wired in)",
        provenance=(
            "Trained by training/emotion_ru/train_single.py on the balanced "
            "ru-izard-emotions set built by training/emotion_ru/01_build_dataset.ipynb. "
            "Never referenced by the shipped pipeline - see docs/PROVENANCE.md."
        ),
        license="MIT (dataset: Djacon/ru-izard-emotions)",
    ),
    # --- Stage 7B: English emotion ----------------------------------------
    "emotion-en-distilroberta": ModelSpec(
        kind="hub",
        ref="j-hartmann/emotion-english-distilroberta-base",
        purpose="stage_7b_english_emotion, ensemble member 1",
        provenance="Third-party pretrained model (Hartmann 2022), used as-is.",
        license="MIT",
    ),
    "emotion-en-deberta": ModelSpec(
        kind="local",
        ref="emotion-en-deberta",
        purpose="stage_7b_english_emotion, ensemble member 2",
        provenance=(
            "MISSING FROM ARCHIVE. DeBERTa-V2-base fine-tuned on the cleaned "
            "super-emotion set; specified in docs/model_cards/emotion_en_deberta.md. "
            "Rebuild with: training/emotion_en_deberta/train.py"
        ),
        license="CC-BY-SA-4.0 (inherited from cirimus/super-emotion)",
        # Trained on the twelve-step normalised corpus, so its inputs must be
        # normalised the same way. The other ensemble member must NOT be.
        preprocess="super_emotion_v1",
    ),
    "emotion-en-emoberta": ModelSpec(
        kind="hub",
        ref="tae898/emoberta-large",
        purpose="stage_7b_english_emotion, stand-in used by the shipped pipeline",
        provenance=(
            "Third-party RoBERTa-large trained on MELD. The original orchestrator "
            "loaded this under the label 'deberta-finetuned', so published outputs "
            "named *_deberta-finetuned_* were NOT produced by the model card's "
            "DeBERTa checkpoint. See docs/PROVENANCE.md."
        ),
        license="MIT",
    ),
}


# ---------------------------------------------------------------------------
# Text transforms
# ---------------------------------------------------------------------------
# A model trained on normalised text must be fed normalised text. The transform
# is named here and attached to the model that needs it, so nothing else in the
# pipeline is affected.
#
# "super_emotion_v1" is the twelve-step cleaner the super-emotion training set
# was built with, vendored verbatim as vea.text_clean. It is deliberately not
# reimplemented: the step order is load-bearing (the build record pins it with a
# test), and it reproduces a known bug - `:/` is stripped as an emoticon before
# URLs are masked, so `https://x` arrives as `https/x` and never matches the URL
# pattern. That bug is in the training data, so it has to be in the inference
# path too. See docs/PROVENANCE.md.
TEXT_TRANSFORMS: dict[str, str] = {
    "super_emotion_v1": "vea.text_clean:clean_text",
}


def resolve_preprocess(name: str | None):
    """Return the text transform named by a :class:`ModelSpec`, or identity.

    Imported lazily by dotted path so that declaring a transform costs nothing
    until a model that uses one is actually loaded.

    Raises:
        KeyError: ``name`` is not a declared transform.
    """
    if name is None:
        return lambda text: text
    if name not in TEXT_TRANSFORMS:
        known = ", ".join(sorted(TEXT_TRANSFORMS)) or "(none declared)"
        raise KeyError(f"Unknown text transform {name!r}. Declared: {known}")

    import importlib

    module_path, _, attribute = TEXT_TRANSFORMS[name].partition(":")
    return getattr(importlib.import_module(module_path), attribute)


#: Registry keys stage 5B will accept, smallest last. Declared here rather than
#: derived from a name prefix so that adding an unrelated NLLB entry to the
#: registry cannot silently become a selectable translator.
TRANSLATION_MODELS = ("nllb-3.3b", "nllb-1.3b", "nllb-600m")


def translation_model_ref(settings: Settings | None = None) -> str:
    """The Hub id stage 5B should load, from ``VEA_TRANSLATION_MODEL``.

    The registry has declared three NLLB sizes since the rewrite and nothing
    could reach two of them: the 3.3B id was written into the pipeline's
    default config. That matters off the server - 3.3B is roughly 17 GB of
    weights and wants 8 GB of VRAM, so on a laptop the choice is this variable
    or no translation stage at all.

    Raises:
        ValueError: on a name that is not a declared translation model. The
            alternative is loading 17 GB because a name was misspelled.
    """
    settings = settings or get_settings()
    name = settings.translation_model
    if name not in TRANSLATION_MODELS:
        raise ValueError(
            f"VEA_TRANSLATION_MODEL={name!r} is not a translation model. "
            f"Choose one of: {', '.join(TRANSLATION_MODELS)}."
        )
    return MODEL_REGISTRY[name].ref


@dataclass(frozen=True)
class Settings:
    """Runtime settings, resolved from environment variables.

    All variables use the ``VEA_`` prefix so they cannot collide with the
    environment of other projects on a shared university server.
    """

    data_dir: Path = field(default_factory=lambda: REPO_ROOT / "data")
    models_dir: Path = field(default_factory=lambda: REPO_ROOT / "models")
    device: str = "auto"
    log_level: str = "INFO"
    translation_model: str = "nllb-3.3b"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        """Build settings from ``os.environ`` (or an explicit mapping, for tests).

        Recognised variables:
            VEA_DATA_DIR    Where per-video working directories are written.
            VEA_MODELS_DIR  Where locally trained checkpoints live.
            VEA_DEVICE      "auto" | "cpu" | "cuda" | "cuda:N".
            VEA_LOG_LEVEL   Standard logging level name.
            VEA_TRANSLATION_MODEL  Which NLLB size stage 5B loads.
        """
        env = os.environ if env is None else env
        defaults = cls()
        return cls(
            data_dir=Path(env.get("VEA_DATA_DIR", defaults.data_dir)).expanduser(),
            models_dir=Path(env.get("VEA_MODELS_DIR", defaults.models_dir)).expanduser(),
            device=env.get("VEA_DEVICE", defaults.device),
            log_level=env.get("VEA_LOG_LEVEL", defaults.log_level).upper(),
            translation_model=env.get("VEA_TRANSLATION_MODEL", defaults.translation_model),
        )

    def video_dir(self, video_id: str) -> Path:
        """Working directory for one video, matching the original layout."""
        return self.data_dir / f"video-{video_id}"


def get_settings() -> Settings:
    """Settings for the current process. Call once and pass down explicitly."""
    return Settings.from_env()


def resolve_model(name: str, settings: Settings | None = None) -> str:
    """Resolve a registry name to something ``from_pretrained`` accepts.

    Hub models resolve to their Hub id. Local checkpoints resolve to an absolute
    path under ``settings.models_dir`` and are checked for existence, so a
    missing checkpoint fails with a message that names the training script
    instead of a stack trace from inside transformers.

    Raises:
        KeyError: ``name`` is not declared in :data:`MODEL_REGISTRY`.
        MissingCheckpointError: a local checkpoint is declared but absent.
    """
    if name not in MODEL_REGISTRY:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model {name!r}. Declared models: {known}")

    spec = MODEL_REGISTRY[name]
    if not spec.is_local:
        return spec.ref

    settings = settings or get_settings()
    path = (settings.models_dir / spec.ref).resolve()
    if not (path / "config.json").is_file():
        raise MissingCheckpointError(
            f"Checkpoint for {name!r} not found at {path}.\n"
            f"  Purpose:    {spec.purpose}\n"
            f"  Provenance: {spec.provenance}\n"
            f"  Set VEA_MODELS_DIR if your weights live elsewhere, or run "
            f"`vea models` to see everything that is missing."
        )
    return str(path)


# The registry entries that pipeline stages actually name. Not every declared
# model is one: `emotion-ru-finetuned` is in the registry so its provenance is
# recorded, and stage 7A loads the hub checkpoint instead - so its absence
# blocks nothing and must not stop a run.
#
# This is a hand-maintained mirror of the `"model"` keys in vea.pipeline's stage
# config, kept here because importing that module pulls in torch and the CLI is
# meant to stay cheap. It cannot drift: test_source_hygiene.py parses pipeline.py
# and asserts the two agree.
STAGE_MODELS = frozenset(
    {
        "va-xlmroberta-large",
        "emotion-ru-rubert-tiny2",
        "emotion-en-distilroberta",
        "emotion-en-deberta",
    }
)


def missing_checkpoints(
    settings: Settings | None = None, *, required_only: bool = False
) -> dict[str, ModelSpec]:
    """Local checkpoints that are declared but absent on this machine.

    Used as a preflight check so a user learns the pipeline cannot complete
    before spending 20 minutes on stages 1-5.

    Args:
        required_only: Restrict the result to checkpoints a stage actually
            names, per `STAGE_MODELS`. ``vea run`` passes True, because
            refusing to start over a model no stage loads is a false alarm -
            and one this command used to raise, while its own message said the
            affected stage would run fine.
    """
    settings = settings or get_settings()
    missing = {}
    for name, spec in MODEL_REGISTRY.items():
        if not spec.is_local:
            continue
        if required_only and name not in STAGE_MODELS:
            continue
        if not (settings.models_dir / spec.ref / "config.json").is_file():
            missing[name] = spec
    return missing


# The seven classes, and every label a stage 7 model actually emits mapped onto
# them. This lives here because three modules had been deciding independently
# what a label means, and they did not agree:
#
#   * visualize.py    EMOTION_CONFIG aliases; anything unlisted -> "neutral"
#   * export.py       its own display map; anything unlisted -> .capitalize()
#   * export.py       its ensemble vote, on raw lowercased strings
#
# On the first full run that cost two wrong numbers. The Russian model emits
# `enthusiasm`, which visualize.py did not know, so 45 of 311 segments became
# Neutral on the timeline; and export.py's vote counted Russian `enthusiasm`
# against English `joy` as a disagreement when they are one class, which is why
# stages 8 and 9 reported different agreement for the same segments.
#
# Unknown labels raise rather than defaulting. Defaulting to "neutral" is what
# hid `enthusiasm` for this project's entire life: it produces a plausible
# timeline, a plausible CSV, and no error anywhere.
EMOTION_CLASSES = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")

#: Label as a model emits it -> canonical class. `enthusiasm` maps to joy on the
#: authority of the build record for the Russian model's own training corpus,
#: which records "collapse by priority; enthusiasm becomes Joy".
EMOTION_ALIASES = {
    "happiness": "joy",
    "happy": "joy",
    "enthusiasm": "joy",
    "sad": "sadness",
}

#: Canonical class -> the name written to the CSV and drawn on the timeline.
#: "joy" displays as "Happiness" because the original project's outputs did.
EMOTION_DISPLAY_NAMES = {
    "anger": "Anger",
    "disgust": "Disgust",
    "fear": "Fear",
    "joy": "Happiness",
    "neutral": "Neutral",
    "sadness": "Sadness",
    "surprise": "Surprise",
}


def canonicalize_emotion(label: str, *, default: str | None = None) -> str:
    """One of `EMOTION_CLASSES`, for any label a stage 7 model emits.

    Args:
        label: The raw label, any case.
        default: Returned instead of raising for an unrecognised label. Pass it
            only where a wrong class is preferable to a crash - and log when you
            do, because a silent default here is the bug this function exists to
            prevent.

    Raises:
        ValueError: on an unrecognised label, when `default` is None.
    """
    key = str(label).strip().lower()
    if key in EMOTION_CLASSES:
        return key
    if key in EMOTION_ALIASES:
        return EMOTION_ALIASES[key]
    if default is not None:
        return default
    raise ValueError(
        f"unrecognised emotion label {label!r}. Known: "
        f"{sorted(set(EMOTION_CLASSES) | set(EMOTION_ALIASES))}. "
        "A stage 7 model emitting a new label needs an entry in EMOTION_ALIASES; "
        "leaving it out silently relabels that share of every video."
    )


def display_emotion(label: str) -> str:
    """The name for a label in the CSV and on the timeline."""
    return EMOTION_DISPLAY_NAMES[canonicalize_emotion(label)]


# YouTube video IDs are exactly 11 characters of [A-Za-z0-9_-]. These patterns
# live here, not in vea.stages.download, for two reasons: `vea run` validates the
# URL before importing the pipeline, and config.py is the only module guaranteed
# importable with no runtime dependencies at all - download.py needs yt_dlp,
# which the fast CI job deliberately does not install. download.py re-exports
# the function so there is one pattern, not two that can drift.
YOUTUBE_ID_PATTERNS = (
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/shorts/)"
    r"([a-zA-Z0-9_-]{11})",
    r"youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})",
)


def extract_youtube_id(url: str) -> str | None:
    """The 11-character video ID in a YouTube URL, or None.

    Returns None rather than raising, because two callers want different
    reactions: `vea run` refuses to start, and the download stage logs and
    moves on to asking yt-dlp directly.
    """
    import re

    for pattern in YOUTUBE_ID_PATTERNS:
        match = re.search(pattern, url or "")
        if match:
            return match.group(1)
    return None


def resolve_device(preferred: str | None = None) -> str:
    """Resolve a torch device string, honouring ``VEA_DEVICE``.

    Args:
        preferred: Explicit override. ``None`` or ``"auto"`` picks cuda when it
            is actually available, otherwise cpu.

    Returns:
        A device string such as ``"cpu"``, ``"cuda"`` or ``"cuda:1"``.

    Note:
        torch is imported lazily so that importing :mod:`vea.config` stays cheap
        and works in environments without torch installed.
    """
    requested = (preferred or get_settings().device or "auto").lower()

    try:
        import torch
    except ImportError:
        if requested.startswith("cuda"):
            raise RuntimeError(
                f"Device {requested!r} requested but torch is not installed. "
                "Install the training/inference extras: `uv sync --extra gpu`."
            ) from None
        return "cpu"

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Device {requested!r} requested but torch reports no CUDA devices. "
            "Use VEA_DEVICE=cpu to run on CPU, or check CUDA_VISIBLE_DEVICES."
        )
    return requested


def configure_logging(level: str | None = None) -> None:
    """Configure root logging once, from the application entry point."""
    resolved = (level or get_settings().log_level).upper()
    logging.basicConfig(
        level=getattr(logging, resolved, logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        force=True,
    )
