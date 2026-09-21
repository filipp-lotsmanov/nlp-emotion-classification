"""HTTP API for the pipeline: submit runs, watch them, read results.

Imported only by `vea serve`, because FastAPI and uvicorn are in the `api`
extra rather than in the core dependencies. `vea models` and `vea config` must
keep working on a machine with neither.
"""

from __future__ import annotations

__all__ = ["create_app"]


def __getattr__(name: str):
    # Lazy, so `import vea.api` does not require fastapi to be installed.
    if name == "create_app":
        from vea.api.app import create_app

        return create_app
    raise AttributeError(name)
