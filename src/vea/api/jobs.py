"""Run the pipeline as a background job, and track where it got to.

The pipeline is a 2-hour subprocess that streams its progress to a log. Three
things follow from that, and they set the whole design:

1. **A job outlives the request that started it, and the page that watched it.**
   State lives in files under ``jobs_dir``, not in memory, so an API restart or
   a browser refresh loses nothing. A job started before a redeploy is still
   readable after one.
2. **Progress has to be parsed, not reported.** ``vea run`` does not expose a
   callback; it logs ``STAGE 5B: TRANSLATION``. `STAGE_PATTERN` reads those
   lines out of the log as they are written. That is a coupling to log text,
   which is fragile - so `tests/test_api.py` pins it against the stage headers
   parsed out of pipeline.py, and a rename fails a test rather than silently
   flatlining the progress bar.
3. **One job at a time.** Every heavy stage wants the GPU, and two concurrent
   runs would OOM rather than share. Extra submissions queue.

Deliberately no Redis, no Celery, no database. A JSON file per job and one
worker thread is the whole mechanism, which is inspectable with `cat` and has
no service to keep alive.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from vea.config import extract_youtube_id, get_settings

JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]

#: Every stage the orchestrator announces, in execution order. The keys are
#: what appears after "STAGE " in the log; the values are for the UI.
STAGES: tuple[tuple[str, str], ...] = (
    ("1", "Download"),
    ("2", "Scene detection"),
    ("3", "Audio preprocessing"),
    ("4", "Transcription"),
    ("5A", "Scene alignment"),
    ("5B", "Translation"),
    ("6A", "Russian intensity"),
    ("6B", "English intensity"),
    ("7A", "Russian emotion"),
    ("7B", "English emotion"),
    ("8", "Timeline"),
    ("9", "CSV export"),
)
STAGE_IDS = tuple(stage for stage, _ in STAGES)

# Matches `STAGE 5B: TRANSLATION` and `STAGE 9: CSV EXPORT` in the log stream.
STAGE_PATTERN = re.compile(r"STAGE (\d[AB]?):")
# The orchestrator's own terminal markers, so a job is not called succeeded
# merely because the process exited zero.
DONE_PATTERN = re.compile(r"PIPELINE COMPLETE \(ALL STAGES\)")
FAILED_PATTERN = re.compile(r"Pipeline stopped: Stage (\S+) failed")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Job:
    """One pipeline run. Serialised to ``<jobs_dir>/<id>.json`` on every change."""

    id: str
    url: str
    video_id: str
    status: JobStatus = "queued"
    stage: str | None = None
    stages_done: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    exit_code: int | None = None

    @property
    def progress(self) -> float:
        """Fraction of stages finished, 0.0 to 1.0.

        Stage count, not elapsed time: the stages are wildly uneven -
        transcription and translation dominate - so this is honest about
        *where* a run is, not about how long is left. The UI says so.
        """
        return len(self.stages_done) / len(STAGES)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["progress"] = self.progress
        data["stage_label"] = dict(STAGES).get(self.stage or "", None)
        return data


def terminate_tree(process: subprocess.Popen, *, windows: bool | None = None) -> None:
    """Stop a running pipeline and everything it started.

    Killing the direct child is not enough on any platform: the job runs
    `python -m vea.cli run`, which spawns yt-dlp and ffmpeg and holds a GPU.
    Signal only the parent and the work carries on with nothing tracking it.

    The two branches are not stylistic. `os.killpg` and `os.getpgid` do not
    exist on Windows at all - this raised AttributeError, which the handler
    below did not catch, so cancelling a job on a native Windows install threw
    a 500 and left the pipeline running. Docker hid it, because the container
    is Linux whatever the host is.

    Args:
        windows: which branch to take. Defaults to the running platform; the
            tests pass it explicitly so both branches are exercised on every
            runner. Faking `os.name` instead would be a global mutation -
            pathlib reads it too, and the first attempt at this test broke
            pytest's own cache writer.
    """
    if os.name == "nt" if windows is None else windows:
        # taskkill /T walks the child tree, which is the only reliable way to
        # reach a grandchild on Windows. /F because a console application that
        # is mid-decode will not close politely.
        subprocess.run(  # noqa: S603, S607
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            check=False,
        )
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        # The group is already gone, or is not ours to signal. Falling back to
        # the direct child is worse than the group but better than nothing.
        process.terminate()


class JobStore:
    """Jobs on disk, plus one worker thread that runs them in submission order.

    Thread-safety: `_lock` guards the in-memory index and the queue. Log files
    are append-only and read without the lock, which is what lets a reader
    stream a running job's output.
    """

    def __init__(self, jobs_dir: Path | None = None, downloads_dir: Path | None = None):
        settings = get_settings()
        self.jobs_dir = Path(jobs_dir) if jobs_dir else settings.data_dir / "jobs"
        self.downloads_dir = Path(downloads_dir) if downloads_dir else Path("downloads")
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._queue: list[str] = []
        self._jobs: dict[str, Job] = {}
        self._current: subprocess.Popen | None = None
        self._current_id: str | None = None
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()

        self._load_existing()

    # -- persistence -------------------------------------------------------

    def _path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def log_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.log"

    def _save(self, job: Job) -> None:
        # Written whole then renamed: a reader polling this file never sees a
        # half-written record, which a plain open-truncate-write would allow.
        tmp = self._path(job.id).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(job), indent=2), encoding="utf-8")
        tmp.replace(self._path(job.id))

    def _load_existing(self) -> None:
        """Re-read jobs from disk, and fail any that a restart orphaned.

        A job recorded as running with no process behind it is not running. It
        is marked failed rather than left in place, because a permanently
        "running" job blocks the queue and tells the UI a lie.
        """
        for path in sorted(self.jobs_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                job = Job(**data)
            except (json.JSONDecodeError, TypeError) as exc:
                # A corrupt record must not stop the service from starting.
                print(f"skipping unreadable job file {path}: {exc}", file=sys.stderr)
                continue
            if job.status in ("running", "queued"):
                job.status = "failed"
                job.error = "interrupted by an API restart"
                job.finished_at = _now()
                self._save(job)
            self._jobs[job.id] = job

    # -- public API --------------------------------------------------------

    def submit(self, url: str) -> Job:
        """Queue a run.

        Raises:
            ValueError: if the URL is not one yt-dlp can resolve to a video id.
                Checked here rather than in the stage, because the orchestrator
                loads every model before stage 1 and a bad URL would otherwise
                cost a full NLLB-3.3B load first.
        """
        video_id = extract_youtube_id(url)
        if not video_id:
            raise ValueError(f"not a YouTube URL this pipeline can parse: {url!r}")

        job = Job(id=uuid.uuid4().hex[:12], url=url, video_id=video_id)
        with self._lock:
            self._jobs[job.id] = job
            self._queue.append(job.id)
        self._save(job)
        self.log_path(job.id).write_text(
            f"queued {job.created_at}\nurl {url}\nvideo {video_id}\n", encoding="utf-8"
        )
        self._ensure_worker()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def cancel(self, job_id: str) -> bool:
        """Stop a running job, or drop a queued one. True if anything changed."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in ("succeeded", "failed", "cancelled"):
                return False
            if job.status == "queued":
                self._queue.remove(job_id)
                job.status = "cancelled"
                job.finished_at = _now()
                self._save(job)
                return True
            process, running_id = self._current, self._current_id

        if process is not None and running_id == job_id:
            terminate_tree(process)
            return True
        return False

    def shutdown(self) -> None:
        self._stop.set()

    # -- the worker --------------------------------------------------------

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(target=self._run_loop, daemon=True, name="vea-jobs")
            self._worker.start()

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                if not self._queue:
                    self._worker = None
                    return
                job_id = self._queue.pop(0)
                job = self._jobs[job_id]
            self._run_one(job)

    def _run_one(self, job: Job) -> None:
        job.status = "running"
        job.started_at = _now()
        self._save(job)

        command = [sys.executable, "-m", "vea.cli", "run", job.url]
        log = self.log_path(job.id)

        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\nstarted {job.started_at}\n$ {' '.join(command)}\n\n")
            handle.flush()
            try:
                # Give the run its own group so cancel() can reach the whole
                # tree. `start_new_session` is POSIX-only and silently ignored
                # on Windows, where the equivalent is a new process group.
                spawn_kwargs: dict = (
                    {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                    if os.name == "nt"
                    else {"start_new_session": True}
                )
                process = subprocess.Popen(  # noqa: S603
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    # Both ends of this pipe carry Russian transcript text, and
                    # both default to the locale encoding - cp1252 on a
                    # Western-European Windows install. PYTHONUTF8 fixes the
                    # child's end for anything that prints before vea.cli's own
                    # _force_utf8_output runs; `encoding` fixes ours, which
                    # would otherwise decode the child's UTF-8 as cp1252 and
                    # hand the log file mojibake or raise outright.
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"},
                    **spawn_kwargs,
                )
            except OSError as exc:
                job.status = "failed"
                job.error = f"could not start the pipeline: {exc}"
                job.finished_at = _now()
                self._save(job)
                handle.write(f"{job.error}\n")
                return

            with self._lock:
                self._current, self._current_id = process, job.id

            saw_complete = False
            assert process.stdout is not None
            for line in process.stdout:
                handle.write(line)
                handle.flush()

                match = STAGE_PATTERN.search(line)
                if match and match.group(1) in STAGE_IDS:
                    stage = match.group(1)
                    if job.stage and job.stage not in job.stages_done:
                        job.stages_done.append(job.stage)
                    job.stage = stage
                    self._save(job)
                    continue
                if DONE_PATTERN.search(line):
                    saw_complete = True
                    if job.stage and job.stage not in job.stages_done:
                        job.stages_done.append(job.stage)
                elif (failure := FAILED_PATTERN.search(line)) is not None:
                    job.error = f"stage {failure.group(1)} failed"

            code = process.wait()

        with self._lock:
            self._current, self._current_id = None, None

        job.exit_code = code
        job.finished_at = _now()
        if code == 0 and saw_complete:
            job.status = "succeeded"
            job.stage = None
            job.stages_done = list(STAGE_IDS)
        elif code < 0:
            # Negative means killed by a signal, which here means cancel().
            job.status = "cancelled"
        else:
            job.status = "failed"
            # Exit zero without the completion banner means the orchestrator
            # returned early. Saying "succeeded" there would be the same class
            # of quiet wrongness this project keeps finding.
            job.error = job.error or (
                "pipeline exited 0 without completing all stages"
                if code == 0
                else f"pipeline exited with code {code}"
            )
        self._save(job)


def tail_log(
    path: Path,
    follow: bool,
    poll: float = 0.5,
    stop: Callable[[], bool] | None = None,
    backlog: int | None = None,
) -> Iterator[str]:
    """Yield a log's lines, optionally waiting for more to be appended.

    Args:
        follow: keep waiting for new lines instead of returning at EOF.
        stop: consulted only at EOF while following. When it returns True the
            producer has finished, so one further read picks up anything
            written between the last line and the state change, and then the
            generator ends. Without it a caller has to break out of the loop
            itself, and breaking on a per-line condition truncates the replay -
            which is exactly the bug this parameter exists to remove.
        backlog: yield only the last N lines already in the file before
            following. A full run writes tens of thousands of lines and no
            reader wants them all.

    Generators are closed when the client disconnects, which is how a
    disconnected browser stops costing anything.
    """
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        if backlog is not None:
            # readlines() leaves the handle at EOF, which is where following
            # has to resume from.
            existing = handle.readlines()
            yield from existing[-backlog:] if backlog > 0 else []

        draining = not follow
        while True:
            line = handle.readline()
            if line:
                yield line
                continue
            if draining:
                return
            if stop is not None and stop():
                # One more pass, then stop: the process may have written its
                # last line after the status was saved.
                draining = True
                continue
            time.sleep(poll)
