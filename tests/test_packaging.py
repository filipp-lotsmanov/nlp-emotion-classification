"""What the project declares, and what CI actually exercises.

These are not tests of behaviour. They are tests of the two places where a
declaration and the thing it describes drift apart silently: a dependency
duplicated between a group and an extra, and a CI matrix that claims to cover
platforms it does not run on.
"""

from __future__ import annotations

import tomllib

from conftest import REPO

PYPROJECT = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
CI = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")


def _requirement(entries: list[str], name: str) -> str | None:
    for entry in entries:
        if entry.split(">=")[0].split("[")[0].strip() == name:
            return entry
    return None


class TestDevGroupMirrorsTheApiExtra:
    """fastapi is declared twice, and the copies must not drift.

    The fast CI job installs the dev group alone: the `api` extra would pull
    the project's core dependencies, torch included, and turn a one-minute job
    into a 3 GB one. So fastapi is duplicated into dev, and this is what stops
    the copy going stale.
    """

    def test_both_declare_fastapi_at_the_same_bound(self):
        extra = _requirement(PYPROJECT["project"]["optional-dependencies"]["api"], "fastapi")
        dev = _requirement(PYPROJECT["dependency-groups"]["dev"], "fastapi")
        assert extra is not None, "the api extra no longer declares fastapi"
        assert dev is not None, (
            "the dev group must declare fastapi, or the fast CI job skips every "
            "API test via pytest.importorskip and the routes go unexercised"
        )
        assert extra == dev, f"bounds have drifted: api has {extra!r}, dev has {dev!r}"

    def test_uvicorn_stays_out_of_the_dev_group(self):
        """uvicorn[standard] pulls uvloop, which has no Windows wheel.

        TestClient does not need a server, so adding it would break the
        Windows runner to no purpose.
        """
        assert _requirement(PYPROJECT["dependency-groups"]["dev"], "uvicorn") is None


class TestCiCoversEveryPlatform:
    """The claim is that this runs on Linux, macOS and Windows."""

    def test_all_three_runners_are_in_the_matrix(self):
        matrix = CI[CI.index("os: [") : CI.index("]", CI.index("os: [")) + 1]
        for runner in ("ubuntu-latest", "windows-latest", "macos-latest"):
            assert runner in matrix, f"{runner} is not in the CI matrix: {matrix}"

    def test_the_fast_job_installs_the_group_the_api_tests_need(self):
        assert "--only-group dev" in CI
