"""API tests: the job model, the run reader, and the routes.

No pipeline is started. `JobStore` runs `python -m vea.cli run <url>` as a
subprocess, so the tests that exercise the worker point it at URLs the CLI
rejects before importing torch - which makes them fast and keeps them honest
about the plumbing rather than about the models.

Skipped entirely when fastapi is absent: it lives in the `api` extra, and CI's
fast job installs only the dev group.
"""

from __future__ import annotations

import ast
import csv
import json
import signal
import time
from pathlib import Path

import pytest

from conftest import REPO

fastapi = pytest.importorskip("fastapi", reason="needs the `api` extra")
from fastapi.testclient import TestClient  # noqa: E402

from vea.api import jobs as jobs_module  # noqa: E402
from vea.api import runs as runs_module  # noqa: E402
from vea.api.app import create_app  # noqa: E402
from vea.api.jobs import STAGE_IDS, STAGE_PATTERN, STAGES, JobStore, tail_log  # noqa: E402

PIPELINE = REPO / "src" / "vea" / "pipeline.py"

# One row per emotion the first real run produced, including the two that
# exposed the vocabulary defect: the Russian model's `enthusiasm`, and the CSV's
# display-cased `Happiness`.
SAMPLE_ROWS = [
    {
        "segment_id": "0",
        "start_time": "0.0",
        "end_time": "5.0",
        "duration": "5.0",
        "text_ru": "Это прекрасно",
        "text_en": "This is wonderful",
        "ru_arousal": "0.7",
        "ru_valence": "0.9",
        "en_arousal": "0.6",
        "en_valence": "0.88",
        "emotion_ru": "enthusiasm",
        "emotion_ru_confidence": "0.81",
        "emotion_en_distil": "Happiness",
        "emotion_en_distil_confidence": "0.77",
        "emotion_en_deberta": "Happiness",
        "emotion_en_deberta_confidence": "0.93",
        "emotion_final": "Happiness",
        "emotion_agreement": "full",
    },
    {
        "segment_id": "1",
        "start_time": "5.0",
        "end_time": "9.5",
        "duration": "4.5",
        "text_ru": "Совещание во вторник",
        "text_en": "The meeting is on Tuesday",
        "ru_arousal": "0.2",
        "ru_valence": "0.5",
        "en_arousal": "0.18",
        "en_valence": "0.51",
        "emotion_ru": "Neutral",
        "emotion_ru_confidence": "0.66",
        "emotion_en_distil": "Neutral",
        "emotion_en_distil_confidence": "0.72",
        "emotion_en_deberta": "Neutral",
        "emotion_en_deberta_confidence": "0.95",
        "emotion_final": "Neutral",
        "emotion_agreement": "full",
    },
    {
        "segment_id": "2",
        "start_time": "9.5",
        "end_time": "14.0",
        "duration": "4.5",
        "text_ru": "Это отвратительно",
        "text_en": "That is revolting",
        "ru_arousal": "0.82",
        "ru_valence": "0.2",
        "en_arousal": "0.85",
        "en_valence": "0.13",
        "emotion_ru": "Anger",
        "emotion_ru_confidence": "0.55",
        "emotion_en_distil": "Disgust",
        "emotion_en_distil_confidence": "0.61",
        "emotion_en_deberta": "Disgust",
        "emotion_en_deberta_confidence": "0.88",
        "emotion_final": "Disgust",
        "emotion_agreement": "majority",
    },
]


@pytest.fixture
def downloads(tmp_path: Path) -> Path:
    """A downloads/ tree with one finished run in it."""
    root = tmp_path / "downloads"
    run = root / "video-laywavOYyYk"
    run.mkdir(parents=True)
    with (run / runs_module.CSV_NAME).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SAMPLE_ROWS[0]))
        writer.writeheader()
        writer.writerows(SAMPLE_ROWS)
    (run / runs_module.TIMELINE_NAME).write_bytes(b"\x89PNG\r\n\x1a\n" + b"stub")
    return root


@pytest.fixture
def client(downloads: Path, tmp_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setenv("VEA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VEA_MODELS_DIR", str(tmp_path / "models"))
    return TestClient(create_app(downloads=downloads, jobs_dir=tmp_path / "jobs"))


# ---------------------------------------------------------------------------
# The stage list is parsed out of a log, so it is pinned against the source
# ---------------------------------------------------------------------------


class TestStageParsing:
    @staticmethod
    def _stages_in_pipeline() -> list[str]:
        """Stage ids from pipeline.py's own `logger.info("STAGE ...")` calls."""
        source = PIPELINE.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(PIPELINE))
        found: list[str] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "info"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                match = STAGE_PATTERN.search(node.args[0].value)
                if match:
                    found.append(match.group(1))
        return found

    def test_the_stage_list_matches_what_the_pipeline_logs(self):
        """A renamed stage header must fail here, not flatline a progress bar.

        Progress is parsed from log text because `vea run` exposes no callback.
        That coupling is the price of not rewriting the orchestrator, and this
        is what keeps it honest.
        """
        assert self._stages_in_pipeline() == list(STAGE_IDS)

    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ("2026-09-19 10:00:00 - INFO - vea.pipeline - STAGE 1: DOWNLOADING VIDEO", "1"),
            ("STAGE 5A: SCENE ALIGNMENT", "5A"),
            ("STAGE 7B: ENGLISH EMOTION CLASSIFICATION", "7B"),
            ("STAGE 9: CSV EXPORT", "9"),
        ],
    )
    def test_it_reads_a_stage_out_of_a_log_line(self, line, expected):
        match = STAGE_PATTERN.search(line)
        assert match and match.group(1) == expected

    def test_it_does_not_match_prose_about_a_stage(self):
        # "Stage 4 failed" and "Stage 2 complete:" are not stage starts.
        for line in ("Pipeline stopped: Stage 4 failed", "Stage 2 complete:"):
            assert STAGE_PATTERN.search(line) is None

    def test_every_stage_has_a_label(self):
        assert all(label for _, label in STAGES)
        assert len(STAGES) == 12


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


class TestJobStore:
    def test_a_bad_url_is_refused_before_anything_starts(self, tmp_path):
        store = JobStore(jobs_dir=tmp_path / "jobs")
        with pytest.raises(ValueError, match="not a YouTube URL"):
            store.submit("not-a-url")
        assert store.list() == []

    def test_a_submitted_job_is_persisted_immediately(self, tmp_path):
        store = JobStore(jobs_dir=tmp_path / "jobs")
        job = store.submit("https://www.youtube.com/watch?v=laywavOYyYk")
        assert job.video_id == "laywavOYyYk"
        record = json.loads((tmp_path / "jobs" / f"{job.id}.json").read_text(encoding="utf-8"))
        assert record["url"] == job.url

    def test_progress_is_stage_count(self, tmp_path):
        store = JobStore(jobs_dir=tmp_path / "jobs")
        job = store.submit("https://youtu.be/laywavOYyYk")
        assert job.progress == 0.0
        job.stages_done = list(STAGE_IDS[:6])
        assert job.progress == 0.5
        job.stages_done = list(STAGE_IDS)
        assert job.progress == 1.0

    def test_a_job_orphaned_by_a_restart_is_failed_not_left_running(self, tmp_path):
        """A "running" job with no process behind it blocks the queue forever."""
        jobs = tmp_path / "jobs"
        jobs.mkdir()
        (jobs / "abc123.json").write_text(
            json.dumps(
                {
                    "id": "abc123",
                    "url": "https://youtu.be/laywavOYyYk",
                    "video_id": "laywavOYyYk",
                    "status": "running",
                    "stage": "4",
                    "stages_done": ["1", "2", "3"],
                    "created_at": "2026-09-19T10:00:00+00:00",
                    "started_at": "2026-09-19T10:00:01+00:00",
                    "finished_at": None,
                    "error": None,
                    "exit_code": None,
                }
            ),
            encoding="utf-8",
        )
        store = JobStore(jobs_dir=jobs)
        recovered = store.get("abc123")
        assert recovered is not None
        assert recovered.status == "failed"
        assert "restart" in (recovered.error or "")

    def test_an_unreadable_record_does_not_stop_startup(self, tmp_path):
        jobs = tmp_path / "jobs"
        jobs.mkdir()
        (jobs / "broken.json").write_text("{not json", encoding="utf-8")
        assert JobStore(jobs_dir=jobs).list() == []

    def test_a_queued_job_can_be_cancelled(self, tmp_path):
        store = JobStore(jobs_dir=tmp_path / "jobs")
        store._stop.set()  # keep the worker from draining the queue
        job = store.submit("https://youtu.be/laywavOYyYk")
        assert store.cancel(job.id) is True
        assert store.get(job.id).status == "cancelled"
        assert store.cancel(job.id) is False

    def test_a_failing_run_is_recorded_as_failed(self, tmp_path, monkeypatch):
        """End to end through the real subprocess, with a URL the CLI rejects.

        Exercises the worker, the log file and the terminal state without
        importing torch: `vea run` refuses an unparseable URL with exit 2
        before it loads anything.
        """
        monkeypatch.setenv("VEA_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("VEA_MODELS_DIR", str(tmp_path / "models"))
        store = JobStore(jobs_dir=tmp_path / "jobs")
        job = store.submit("https://www.youtube.com/watch?v=laywavOYyYk")

        for _ in range(200):
            if store.get(job.id).status in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(0.1)

        done = store.get(job.id)
        assert done.status == "failed", done.to_dict()
        assert done.exit_code not in (None, 0)
        assert store.log_path(job.id).read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Reading finished runs
# ---------------------------------------------------------------------------


class TestRunReader:
    def test_it_finds_the_run(self, downloads):
        assert runs_module.list_runs(downloads) == ["laywavOYyYk"]

    def test_labels_come_back_canonical(self, downloads):
        rows = runs_module.read_segments(downloads, "laywavOYyYk")
        # `enthusiasm` and `Happiness` are one class. Treating them as two is
        # the defect that made stages 8 and 9 disagree.
        assert rows[0]["emotion_ru"] == "joy"
        assert rows[0]["emotion_en_distil"] == "joy"
        assert rows[0]["emotion_final"] == "joy"

    def test_the_raw_label_is_kept_alongside(self, downloads):
        # The difference between them is a finding, not noise.
        rows = runs_module.read_segments(downloads, "laywavOYyYk")
        assert rows[0]["emotion_ru_raw"] == "enthusiasm"
        assert rows[0]["emotion_ru"] == "joy"

    def test_summary_counts(self, downloads):
        summary = runs_module.summarize(downloads, "laywavOYyYk")
        assert summary.segments == 3
        assert summary.neutral_share == pytest.approx(1 / 3)
        assert summary.duration_seconds == pytest.approx(14.0)
        assert summary.agreement == {"full": 2, "majority": 1}
        assert summary.has_timeline is True

    def test_per_model_counts_cover_all_seven_classes(self, downloads):
        counts = runs_module.per_model_counts(downloads, "laywavOYyYk")
        assert set(counts) == set(runs_module.MODEL_COLUMNS.values())
        for row in counts.values():
            assert len(row) == 7

    def test_a_traversing_video_id_is_refused(self, downloads):
        # This id arrives from an HTTP path segment.
        for bad in ("../../etc", "a/b", "..", "x\\y"):
            with pytest.raises(ValueError, match="invalid video id"):
                runs_module.video_dir(downloads, bad)

    def test_a_run_without_a_csv_is_not_listed(self, downloads):
        (downloads / "video-halfdone").mkdir()
        assert "halfdone" not in runs_module.list_runs(downloads)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


class TestRoutes:
    def test_meta_serves_the_shared_vocabulary(self, client):
        body = client.get("/api/meta").json()
        assert len(body["stages"]) == 12
        assert body["emotions"] == list(runs_module.display_names())
        assert body["display_names"]["joy"] == "Happiness"

    def test_health_reports_whether_a_run_is_possible(self, client):
        body = client.get("/api/health").json()
        assert body["ok"] is True
        # The fixture points at an empty models dir, so it must say so rather
        # than letting the UI offer a submit button that cannot work.
        assert body["can_run_pipeline"] is False
        assert "va-xlmroberta-large" in body["missing_checkpoints"]

    def test_listing_and_reading_a_run(self, client):
        listing = client.get("/api/runs").json()
        assert [run["video_id"] for run in listing] == ["laywavOYyYk"]

        detail = client.get("/api/runs/laywavOYyYk").json()
        assert detail["segments"] == 3
        assert "per_model" in detail

        rows = client.get("/api/runs/laywavOYyYk/segments").json()
        assert len(rows) == 3
        assert rows[0]["text_ru"] == "Это прекрасно"

    def test_the_timeline_is_served_as_png(self, client):
        response = client.get("/api/runs/laywavOYyYk/timeline")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_an_unknown_run_is_404_not_500(self, client):
        assert client.get("/api/runs/aaaaaaaaaaa").status_code == 404
        assert client.get("/api/runs/aaaaaaaaaaa/segments").status_code == 404
        assert client.get("/api/runs/aaaaaaaaaaa/timeline").status_code == 404

    def test_a_bad_url_is_422_with_a_reason(self, client):
        response = client.post("/api/jobs", json={"url": "https://vimeo.com/123"})
        assert response.status_code == 422
        assert "YouTube" in response.json()["detail"]

    def test_submitting_returns_a_job_that_can_be_read_back(self, client):
        created = client.post(
            "/api/jobs", json={"url": "https://www.youtube.com/watch?v=laywavOYyYk"}
        )
        assert created.status_code == 201
        job = created.json()
        assert job["video_id"] == "laywavOYyYk"
        assert job["progress"] == 0.0

        assert client.get(f"/api/jobs/{job['id']}").json()["id"] == job["id"]
        assert any(item["id"] == job["id"] for item in client.get("/api/jobs").json())

    def test_cancelling_an_unknown_job_is_404(self, client):
        assert client.delete("/api/jobs/deadbeef").status_code == 404


class TestServeFlagsAreWired:
    """Every `vea serve` flag must actually be read.

    `--reload` was accepted and then ignored: uvicorn was always handed an app
    object, which it cannot reload. It is the same defect as `--structural-only`
    in the verification script - a flag that parses, prints no warning, and
    does nothing. This test is written against the parser rather than against
    the one flag, so the next one cannot slip through either.
    """

    @staticmethod
    def _serve_source() -> tuple[set[str], str]:
        tree = ast.parse((REPO / "src" / "vea" / "cli.py").read_text(encoding="utf-8"))

        flags: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            target = node.func.value
            if not (isinstance(target, ast.Name) and target.id == "p_serve"):
                continue
            if node.func.attr != "add_argument":
                continue
            for argument in node.args:
                if isinstance(argument, ast.Constant) and argument.value.startswith("--"):
                    flags.add(argument.value[2:].replace("-", "_"))

        body = next(
            ast.get_source_segment((REPO / "src" / "vea" / "cli.py").read_text(encoding="utf-8"), n)
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_cmd_serve"
        )
        return flags, body

    def test_every_flag_is_read_by_the_handler(self):
        flags, body = self._serve_source()
        assert flags, "the parser declares no serve flags; this test is looking in the wrong place"
        unused = sorted(flag for flag in flags if f"args.{flag}" not in body)
        assert not unused, f"vea serve accepts {unused} and never reads them"

    def test_reload_uses_an_import_string(self):
        """An app *object* cannot be reloaded; uvicorn requires the string."""
        _, body = self._serve_source()
        assert '"vea.api.app:app"' in body

    def test_the_module_level_app_honours_the_downloads_environment_variable(self, tmp_path):
        """That import string is how --reload's downloads path survives.

        The reloader re-imports the module in a fresh process, so anything set
        on the app built in the parent is gone. The environment variable is the
        only channel left.
        """
        source = (REPO / "src" / "vea" / "api" / "app.py").read_text(encoding="utf-8")
        assert "VEA_DOWNLOADS_DIR" in source

        import os
        import subprocess
        import sys

        marker = tmp_path / "downloads-from-env"
        marker.mkdir()
        env = {**os.environ, "VEA_DOWNLOADS_DIR": str(marker)}
        result = subprocess.run(
            # sys.executable, not "python". On Windows a bare `python` resolves
            # through PATH to a system interpreter - or the Store shim - which
            # has no `vea` installed, and the test failed there while passing
            # on Linux for no reason connected to what it is testing.
            [sys.executable, "-c", "from vea.api.app import app; print(app.state.downloads)"],
            capture_output=True,
            text=True,
            env=env,
            cwd=REPO,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == str(marker)


class TestTheEventStreamReplaysTheWholeLog:
    """Opening a finished job must show its output, not its first line.

    The stream used to break out of its loop the moment the job was in a
    terminal state, and that check ran after the *first* log line. Every
    finished job therefore streamed `queued <timestamp>` and stopped, so the
    reason a run failed never reached the browser - while sitting in the log
    file on disk the whole time.
    """

    @staticmethod
    def _events(client, job_id: str) -> list[tuple[str, str]]:
        with client.stream("GET", f"/api/jobs/{job_id}/events") as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            text = "".join(response.iter_text())
        events = []
        for block in text.split("\n\n"):
            lines = block.splitlines()
            if len(lines) >= 2 and lines[0].startswith("event: "):
                events.append((lines[0][7:], lines[1][6:]))
        return events

    @pytest.fixture
    def finished_job(self, tmp_path):
        store = JobStore(jobs_dir=tmp_path / "jobs", downloads_dir=tmp_path / "downloads")
        store.shutdown()  # no worker: this job's outcome is written by hand
        job = store.submit("https://www.youtube.com/watch?v=laywavOYyYk")
        job.status = "failed"
        job.error = "pipeline exited with code 1"
        store._save(job)
        store.log_path(job.id).write_text(
            "queued 2026-01-01T00:00:00+00:00\n"
            "url https://www.youtube.com/watch?v=laywavOYyYk\n"
            "started 2026-01-01T00:00:01+00:00\n"
            "STAGE 1: DOWNLOAD\n"
            "Refusing to start: checkpoints that a stage needs are missing.\n"
            "  va-xlmroberta-large -> stage_6a_russian_intensity\n",
            encoding="utf-8",
        )
        # A second store over the same directory: the API reads the files,
        # which is the point of keeping the state on disk.
        with TestClient(
            create_app(downloads=tmp_path / "downloads", jobs_dir=tmp_path / "jobs")
        ) as client:
            yield client, job.id

    def test_every_line_is_streamed_not_only_the_first(self, finished_job):
        client, job_id = finished_job
        logs = [json.loads(data) for kind, data in self._events(client, job_id) if kind == "log"]
        assert len(logs) == 6, f"expected the whole log, got {logs}"
        assert logs[0].startswith("queued")
        assert "Refusing to start" in logs[4]
        assert logs[-1].strip().startswith("va-xlmroberta-large")

    def test_the_stream_still_terminates(self, finished_job):
        client, job_id = finished_job
        kinds = [kind for kind, _ in self._events(client, job_id)]
        assert kinds[0] == "job"
        assert kinds[-1] == "end"
        assert kinds.count("end") == 1


class TestTailLog:
    def test_backlog_keeps_only_the_last_lines(self, tmp_path):
        path = tmp_path / "big.log"
        path.write_text("".join(f"line {i}\n" for i in range(5000)), encoding="utf-8")
        lines = list(tail_log(path, follow=False, backlog=2000))
        assert len(lines) == 2000
        assert lines[0] == "line 3000\n"
        assert lines[-1] == "line 4999\n"

    def test_no_backlog_yields_everything(self, tmp_path):
        path = tmp_path / "small.log"
        path.write_text("a\nb\nc\n", encoding="utf-8")
        assert list(tail_log(path, follow=False)) == ["a\n", "b\n", "c\n"]

    def test_following_stops_once_the_producer_is_done(self, tmp_path):
        """And drains what was written after the stop condition flipped."""
        path = tmp_path / "live.log"
        path.write_text("first\n", encoding="utf-8")

        done = {"value": False}

        def stop() -> bool:
            # Write a final line at the same moment the job goes terminal,
            # which is the race the extra drain pass exists for.
            if not done["value"]:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write("last\n")
                done["value"] = True
                return True
            return True

        lines = list(tail_log(path, follow=True, poll=0.01, stop=stop))
        assert lines == ["first\n", "last\n"]

    def test_a_missing_log_is_not_an_error(self, tmp_path):
        assert list(tail_log(tmp_path / "nope.log", follow=False)) == []


class TestCancellingIsPortable:
    """Stopping a run must work on every platform this project claims.

    `cancel()` called `os.killpg(os.getpgid(pid), SIGTERM)` unconditionally.
    Neither function exists on Windows, so it raised AttributeError - which
    the handler did not catch - and the DELETE route answered 500 while the
    pipeline carried on holding a GPU. Docker hid it: the container is Linux
    whatever the host is, and CI never ran the API tests at all because
    fastapi was missing from the group the fast job installs.

    Both branches are exercised on every runner, by faking os.name. Testing
    only the branch the current platform takes is what let this through.
    """

    class FakeProcess:
        def __init__(self, pid: int = 4321) -> None:
            self.pid = pid
            self.terminated = False

        def terminate(self) -> None:
            self.terminated = True

    def test_windows_kills_the_whole_tree(self, monkeypatch):
        process = self.FakeProcess()
        calls: list[list[str]] = []
        monkeypatch.setattr(
            jobs_module.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or None
        )
        # Present on POSIX runners; must not be reached on the Windows path.
        monkeypatch.setattr(
            jobs_module.os,
            "killpg",
            lambda *a: pytest.fail("killpg does not exist on Windows"),
            raising=False,
        )

        jobs_module.terminate_tree(process, windows=True)

        assert calls == [["taskkill", "/F", "/T", "/PID", "4321"]], calls
        assert not process.terminated

    def test_posix_signals_the_process_group(self, monkeypatch):
        process = self.FakeProcess()
        signalled: list[tuple[int, int]] = []
        monkeypatch.setattr(jobs_module.os, "getpgid", lambda pid: pid + 1, raising=False)
        monkeypatch.setattr(
            jobs_module.os,
            "killpg",
            lambda pgid, sig: signalled.append((pgid, sig)),
            raising=False,
        )

        jobs_module.terminate_tree(process, windows=False)

        assert signalled == [(4322, signal.SIGTERM)]
        assert not process.terminated

    def test_a_vanished_group_falls_back_to_the_child(self, monkeypatch):
        process = self.FakeProcess()
        monkeypatch.setattr(jobs_module.os, "getpgid", lambda pid: pid, raising=False)

        def gone(*_args):
            raise ProcessLookupError

        monkeypatch.setattr(jobs_module.os, "killpg", gone, raising=False)

        jobs_module.terminate_tree(process, windows=False)

        assert process.terminated, "a dead group must not leave the child running"

    def test_the_spawn_flags_match_the_platform(self):
        """POSIX gets a session, Windows gets a process group.

        `start_new_session` is silently ignored on Windows, so passing it
        there would look correct and give the run no group to be killed by.
        """
        source = (REPO / "src" / "vea" / "api" / "jobs.py").read_text(encoding="utf-8")
        assert "CREATE_NEW_PROCESS_GROUP" in source
        assert '"start_new_session": True' in source
        # Never unconditionally, which is what it used to be.
        assert "start_new_session=True," not in source
