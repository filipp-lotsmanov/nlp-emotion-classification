"""Video emotion analysis pipeline (Russian video -> emotion timeline).

Public surface:
    vea.config    Settings, model registry, device and logging helpers.
    vea.pipeline  Stage orchestration (``main``) and per-stage runners.
    vea.stages    One module per pipeline stage.

Importing this package is cheap: nothing here pulls in torch, transformers or
faster-whisper. Those are imported by the individual stage modules when a stage
actually runs, so the CLI can report configuration and missing checkpoints on a
machine without a GPU stack installed.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
