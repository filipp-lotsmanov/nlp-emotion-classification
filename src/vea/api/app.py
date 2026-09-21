"""HTTP API: submit a run, watch it, read the result.

    uv run vea serve                      # http://127.0.0.1:8000
    uv run vea serve --host 0.0.0.0       # inside a container only

**This service has no authentication, and it must not be exposed.** It takes a
URL from the caller and hands it to yt-dlp, and it starts a subprocess that
writes to disk and claims a GPU. Anyone who can reach the port can make this
machine fetch a URL of their choosing and spend two hours of compute on it. It
binds to 127.0.0.1 by default; the container publishes to 127.0.0.1 too. The
`--host 0.0.0.0` in the image's CMD binds inside the container's own network
namespace, which is not the same as publishing it.

Routes:

    GET  /api/meta                    stage list, emotion vocabulary, palette
    GET  /api/health                  liveness, plus whether checkpoints exist
    POST /api/jobs                    {"url": ...} -> a job
    GET  /api/jobs                    every job, newest first
    GET  /api/jobs/{id}               one job
    DELETE /api/jobs/{id}             cancel a queued or running job
    GET  /api/jobs/{id}/events        SSE: job state + log lines, live
    GET  /api/runs                    finished runs
    GET  /api/runs/{video_id}         summary + per-model counts
    GET  /api/runs/{video_id}/segments    every row
    GET  /api/runs/{video_id}/timeline    the PNG stage 8 drew
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from vea.api import runs as runs_module
from vea.api.jobs import STAGES, JobStore, tail_log
from vea.config import EMOTION_CLASSES, get_settings, missing_checkpoints

# The Next.js dev server runs on 3000 and the API on 8000, so a browser calling
# one from the other is cross-origin. In the shipped container they are behind
# one origin and this does nothing. Listed explicitly rather than "*": the
# service has no auth, so a wildcard would let any page a developer happens to
# have open drive it.
DEV_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

#: How much of an existing log an opening stream replays. A full run writes
#: tens of thousands of lines; the browser keeps the last 500 of them, so
#: sending every one would be paid for on both ends and then discarded. Higher
#: than the browser's own cap so a reader can still scroll back a little.
LOG_BACKLOG_LINES = 2000


class SubmitRequest(BaseModel):
    url: str = Field(..., description="YouTube URL", max_length=2048)


def create_app(downloads: Path | None = None, jobs_dir: Path | None = None) -> FastAPI:
    settings = get_settings()
    downloads_dir = Path(downloads) if downloads else Path("downloads")
    store = JobStore(jobs_dir=jobs_dir, downloads_dir=downloads_dir)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        # Stops the worker thread from picking up another job while the
        # process is going down mid-run.
        store.shutdown()

    app = FastAPI(
        title="Video Emotion Analysis",
        description="Nine-stage emotion analysis of Russian video. Loopback only.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEV_ORIGINS,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )
    app.state.store = store
    app.state.downloads = downloads_dir

    # -- metadata ----------------------------------------------------------

    @app.get("/api/meta")
    def meta() -> dict:
        """Everything the frontend needs to render without hardcoding it.

        The palette and the display names come from the same modules the
        pipeline uses, so the browser cannot disagree with the PNG about what
        colour Joy is, or with the CSV about what it is called.
        """
        try:
            colours = runs_module.palette()
        except ImportError:
            # The viewer image has no matplotlib. The UI falls back to its own
            # colours rather than failing to load.
            colours = {}
        return {
            "stages": [{"id": stage, "label": label} for stage, label in STAGES],
            "emotions": list(EMOTION_CLASSES),
            "display_names": runs_module.display_names(),
            "palette": colours,
        }

    @app.get("/api/health")
    def health() -> dict:
        missing = missing_checkpoints(settings, required_only=True)
        return {
            "ok": True,
            "downloads": str(downloads_dir),
            "can_run_pipeline": not missing,
            "missing_checkpoints": sorted(missing),
        }

    # -- jobs --------------------------------------------------------------

    @app.post("/api/jobs", status_code=201)
    def submit(request: SubmitRequest) -> dict:
        try:
            return store.submit(request.url).to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/jobs")
    def list_jobs() -> list[dict]:
        return [job.to_dict() for job in store.list()]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id}")
        return job.to_dict()

    @app.delete("/api/jobs/{job_id}")
    def cancel_job(job_id: str) -> dict:
        if store.get(job_id) is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id}")
        return {"cancelled": store.cancel(job_id)}

    @app.get("/api/jobs/{job_id}/events")
    def job_events(job_id: str) -> StreamingResponse:
        """Server-sent events: the job record, then log lines as they arrive.

        SSE rather than websockets because the traffic is one-directional and
        it reconnects on its own. The generator ends when the job reaches a
        terminal state, so the browser is not left holding an idle stream.
        """
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id}")

        def stream() -> Iterator[str]:
            current = store.get(job_id)
            assert current is not None
            yield f"event: job\ndata: {json.dumps(current.to_dict())}\n\n"

            terminal = {"succeeded", "failed", "cancelled"}
            follow = current.status not in terminal
            last_status = current.status

            def finished() -> bool:
                latest = store.get(job_id)
                return latest is None or latest.status in terminal

            # The loop must not decide when to stop. It used to break as soon
            # as the job was terminal, which meant opening a *finished* job
            # streamed exactly one line of its log and stopped - the run's
            # actual output never reached the browser. Ending is tail_log's
            # job now, once the producer is done and the file is drained.
            lines = tail_log(
                store.log_path(job_id),
                follow=follow,
                stop=finished,
                backlog=LOG_BACKLOG_LINES,
            )
            for line in lines:
                yield f"event: log\ndata: {json.dumps(line.rstrip())}\n\n"
                latest = store.get(job_id)
                if latest is None:
                    break
                # Re-send the record whenever anything a caller renders moved.
                if latest.status != last_status or latest.stage != current.stage:
                    current, last_status = latest, latest.status
                    yield f"event: job\ndata: {json.dumps(latest.to_dict())}\n\n"

            final = store.get(job_id)
            if final is not None:
                yield f"event: job\ndata: {json.dumps(final.to_dict())}\n\n"
            yield "event: end\ndata: {}\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                # Without this an intermediate proxy buffers the stream and the
                # progress bar only moves when the run finishes.
                "X-Accel-Buffering": "no",
            },
        )

    # -- runs --------------------------------------------------------------

    @app.get("/api/runs")
    def list_runs() -> list[dict]:
        out = []
        for video_id in runs_module.list_runs(downloads_dir):
            try:
                out.append(runs_module.summarize(downloads_dir, video_id).to_dict())
            except (FileNotFoundError, ValueError):
                continue
        return out

    @app.get("/api/runs/{video_id}")
    def get_run(video_id: str) -> dict:
        try:
            summary = runs_module.summarize(downloads_dir, video_id)
            return {
                **summary.to_dict(),
                "per_model": runs_module.per_model_counts(downloads_dir, video_id),
            }
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/runs/{video_id}/segments")
    def get_segments(video_id: str) -> list[dict]:
        try:
            return runs_module.read_segments(downloads_dir, video_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/runs/{video_id}/timeline")
    def get_timeline(video_id: str) -> FileResponse:
        try:
            path = runs_module.video_dir(downloads_dir, video_id) / runs_module.TIMELINE_NAME
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"no timeline for {video_id}")
        return FileResponse(path, media_type="image/png")

    return app


#: The module-level app, for `uvicorn vea.api.app:app` and for `vea serve
#: --reload`. Reload needs an import string rather than an object - uvicorn
#: re-imports the module in a fresh process on every edit, so an app built here
#: in the parent would never be the one serving. That is why the downloads
#: directory arrives by environment variable: it is the only thing that
#: survives the re-import.
app = create_app(downloads=os.environ.get("VEA_DOWNLOADS_DIR") or None)
