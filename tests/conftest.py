"""Shared test fixtures and helpers.

`load_training_module` exists because `training/` is not an installed package
and both trainers are called `train.py`. Adding each directory to `sys.path` and
writing `from train import ...` works in isolation and breaks as soon as two
such test modules run in one session: the first import wins `sys.modules["train"]`
and the second silently gets the wrong file, failing with an ImportError that
names a symbol from the other trainer. Loading each one by path under a distinct
module name removes the ambiguity instead of depending on collection order.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[1]
TRAINING = REPO / "training"


def load_training_module(package: str, module: str = "train") -> ModuleType:
    """Import `training/<package>/<module>.py` under a collision-free name.

    Args:
        package: directory under `training/`, e.g. "va_regressor".
        module: file stem within it, defaulting to "train".

    Returns:
        The imported module, cached in `sys.modules` under
        `"_training_<package>_<module>"`.

    Raises:
        FileNotFoundError: if the file does not exist, rather than the
            ModuleNotFoundError that an import-path approach would give.
    """
    path = TRAINING / package / f"{module}.py"
    if not path.is_file():
        raise FileNotFoundError(path)

    name = f"_training_{package}_{module}"
    if name in sys.modules:
        return sys.modules[name]

    # The trainers import their own siblings (train.py imports data_prep), so
    # the package directory still has to be importable from inside them.
    directory = str(path.parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)

    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded
