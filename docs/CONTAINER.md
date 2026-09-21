# Running it in containers

The pipeline was built to run on one server with a GPU. That is fine for
producing results and useless for showing them: a marker cannot install CUDA to
read a timeline. So the project ships as two images rather than one.

| | `vea:viewer` | `vea:app` |
|---|---|---|
| base | `python:3.11-slim` | same, plus ffmpeg |
| extras installed | `api` | `api`, `train` |
| torch / CUDA | no | yes, plus a CUDA 12 cuBLAS and cuDNN for CTranslate2 |
| can run stages 1–9 | no | yes |
| can read finished runs | yes | yes |
| approximate size | a few hundred MB | several GB |
| needs a GPU | no | no — but attach one only via `compose.gpu.yaml`, and a CPU run is roughly ten times slower |

Both expose the same API on port 8000, so the frontend does not know or care
which one is behind it.

## Quick start

```bash
./scripts/setup_docker.sh              # viewer + frontend
./scripts/setup_docker.sh --full       # full pipeline + frontend
./scripts/setup_docker.sh --full --cpu # the same, card never attached
./scripts/setup_docker.sh --down       # stop, keep the data
```

The script checks the host before it builds anything — docker, the compose
plugin, BuildKit, and (only for the full pipeline) whether containers can
actually reach the card, attaching it only if they can. It then builds, starts, and waits for `/api/health` to answer
rather than printing a URL that is not live yet.

Without the script:

```bash
docker compose up -d                                # viewer + frontend
docker compose --profile full up -d app frontend    # pipeline on the CPU

# pipeline with the card attached
docker compose -f compose.yaml -f compose.gpu.yaml --profile full up -d app frontend
```

Name the services in the second and third forms. `--profile full up` on its own
starts the viewer as well, and the two cannot both hold port 8000. (`--profile
gpu` still selects the same service; it is kept as an alias.)

### The GPU is opt-in, in a second file

`compose.gpu.yaml` holds nothing but the device reservation, and that split is
deliberate. A GPU reservation is not a preference the daemon can decline: with
no adapter visible it refuses to create the container at all.

```
nvidia-container-cli: initialization error: WSL environment detected but no
adapters were found
```

With the reservation in the base file, a laptop without a card could not start
the pipeline — not slowly, at all — even though every stage runs perfectly well
on a CPU. Opting in is now a flag; opting out used to be an edit.

Confirm a card is actually reachable before using the overlay:

```bash
docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L
```

`scripts/setup_docker.sh` runs exactly that probe and adds the overlay only if
it passes.

### Building without CUDA

`CUDA_RUNTIME=0` skips the CUDA 12 cuBLAS and cuDNN wheels and the `ldconfig`
step that follows them — about 1.4 GB that only stage 4 on a card ever loads:

```bash
CUDA_RUNTIME=0 docker compose --profile full build app
```

The setup script sets it automatically when it finds no usable GPU.

- frontend: <http://127.0.0.1:3000>
- API docs: <http://127.0.0.1:8000/docs>

## On Windows

Docker Desktop with the WSL2 backend. Two ways to drive it:

**PowerShell**, no bash involved — this is the shortest path, and the setup
script's checks are the only thing you give up:

```powershell
docker compose build viewer frontend
docker compose up -d viewer frontend
docker compose cp .\downloads\video-XXXXXXXXXXX viewer:/data/downloads/
start http://127.0.0.1:3000
```

`docker compose cp` avoids the seeding step entirely, and with it the one
Windows trap worth knowing about. The copied files end up owned by root rather
than by the container's user; the viewer only reads them, so that is fine. It
would matter for the `app` image, which writes into a run's directory.

**Git Bash** (ships with Git for Windows) runs `scripts/setup_docker.sh`
unchanged. The script sets `MSYS_NO_PATHCONV` around the one `docker run` that
needs it: Git Bash rewrites the container-side half of `-v host:/container`
into a Windows path, so `/seed` arrives as `C:/Program Files/Git/seed` and the
mount fails naming a directory you never wrote.

Do not use PowerShell to run the script, and do not use `curl` in PowerShell to
check the API — `curl` there is an alias for `Invoke-WebRequest`, which takes
different arguments. Use `curl.exe`, or just open the page.

`.gitattributes` pins `*.sh`, `Dockerfile` and the compose files to LF. Without
it, Git's Windows default rewrites them to CRLF on checkout and a Linux shell
reads `#!/usr/bin/env bash\r`, then reports "bad interpreter" against a file
that is plainly there. If you unzipped rather than cloned, the line endings
were already LF and nothing converted them.

GPU passthrough works on Windows through WSL2 — recent NVIDIA driver, no
toolkit install inside WSL. Without an NVIDIA card, do not use
`compose.gpu.yaml`: the daemon refuses to create the container rather than
falling back. The `app` image itself runs fine on the CPU.

## Showing real results without a GPU

The viewer reads runs out of the `/data` volume, so a run produced on the
server can be copied into a laptop's volume and browsed there:

```bash
./scripts/setup_docker.sh --seed downloads/
```

That copies every `downloads/video-*/` directory into the volume and hands
ownership to the container's user. A run needs only its
`emotion_analysis_data.csv`; `emotion_timeline.png` is shown when present.

## What is bound where, and why

Everything publishes to `127.0.0.1` and nothing to `0.0.0.0`.

This is not caution for its own sake. **The API has no authentication**, and
`POST /api/jobs` takes a URL from the caller and hands it to yt-dlp, then
spends up to hours of GPU on the result. Anyone who can reach port 8000 can
make the machine fetch a URL of their choosing. Before putting it on a network,
put something in front of it that asks who is calling.

The `--host 0.0.0.0` in each image's `CMD` is not a contradiction: it binds
inside that container's own network namespace. What reaches the outside is the
`ports:` mapping in `compose.yaml`, and that says `127.0.0.1`.

Both images run as a non-root user (uid 10001 for the API, 10002 for the
frontend), and `/data` is the only path either writes to.

## State

One volume, `vea_vea-data`, mounted at `/data`:

```
/data/downloads/video-<id>/   per-run artefacts, including the stage 9 CSV
/data/models/                 HuggingFace cache; roughly 4 GB after a first run
/data/jobs/<job-id>.json      one file per job
/data/jobs/<job-id>.log       its full stdout
/data/cache/                  matplotlib and friends
```

Inside the image, `/app/downloads` and `/app/models` are symlinks into `/data`,
because those are the paths the CLI's own defaults use. Rebuilding an image
therefore never costs you the model cache.

Jobs are files, which is the reason a run survives a restart of the API, a
closed browser, and a `docker compose down`. On start the store re-reads the
directory and marks as failed anything that was running when the process died,
rather than leaving a job that claims to be running with no process behind it.

To throw the state away:

```bash
docker compose --profile full down
docker volume rm vea_vea-data
```

## The CUDA 12 problem, inside the image

Stage 4 (transcription) runs on faster-whisper, which runs on CTranslate2,
whose wheels are built against CUDA 12: at runtime they load `libcublas.so.12`
and `libcudnn.so.9` **by name**. torch now resolves to a CUDA 13 build and
brings `nvidia-cublas-cu13`, which provides `libcublas.so.13`. Same library,
different soname, so the loader finds nothing usable and stage 4 dies with:

```
RuntimeError: Library libcublas.so.12 is not found or cannot be loaded
```

Only cuBLAS is actually missing. cuDNN's soname is `libcudnn.so.9` under both
CUDA majors and torch already ships it — it is simply on torch's own loader
path and nowhere the system linker looks.

The `app` image installs **`nvidia-cublas-cu12` only, into `/opt/cuda12`**, and
lists both that prefix and torch's own `nvidia/*/lib` directories in
`/etc/ld.so.conf.d/` before running `ldconfig`. torch keeps using `.so.13`;
CTranslate2 finds `.so.12` and torch's `libcudnn.so.9`.

### The mistake that made this worth writing down

An earlier version of the Dockerfile ran, inside torch's own virtualenv:

```
uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

`nvidia-cudnn-cu12` installs `nvidia/cudnn/lib/libcudnn.so.9`. That is the
exact path `nvidia-cudnn-cu13` already owns, and torch links it at import. The
install replaced the file, and the image then died before stage 1 on every
machine, including ones with no GPU at all:

```
ImportError: libcudnn.so.9: cannot open shared object file
```

The build had checked that both sonames were in the loader cache. It had not
checked that torch could still be imported — so the patch broke the thing it
was patching and the build passed. It now ends with:

```
python -c "import torch, ctranslate2; print(...)"
```

and `tests/test_source_hygiene.py` fails if that line, or the `--target`
prefix, goes away. cuBLAS was never the problem: `nvidia-cublas-cu12` installs
to `nvidia/cublas/lib/`, torch's to `nvidia/cu13/lib/`, and they do not
collide.

`ldconfig` rather than `LD_LIBRARY_PATH` so that it holds for any process in
the container, not only the ones started through a wrapper that remembered to
set it. The build then asserts both sonames are in the cache, so the image
fails to build rather than failing at stage 4 an hour into a run.

This costs about 1.4 GB. The alternative, pinning torch to a cu12 index, would
change `uv.lock` for every environment including the server where the current
pin is already proven. See `docs/PROVENANCE.md` section 12.

## Build layout

Both targets branch from a shared `base` rather than `app` inheriting from
`viewer`: a layer that is overwritten is still a layer that gets pulled.

Each target runs `uv sync` twice — once with `--no-install-project` to resolve
from `uv.lock` alone, then once more for the project itself. The first is most
of the image and stays cached until the lock changes; the second is small and
changes constantly. `pyproject.toml` and `uv.lock` are bind-mounted into those
`RUN` steps rather than copied, so editing the README does not invalidate the
dependency layer. `UV_FROZEN=1` means a build fails if the lock is stale
instead of quietly resolving something else.

`.dockerignore` denies everything and re-admits what the build needs. A
denylist would start shipping whatever gets added to the repository next.

## Frontend

Next.js 15, App Router, built with `output: "standalone"`, so the runtime image
carries the server and the traced dependencies rather than the whole of
`node_modules`. No CSS framework: five pages did not justify one.

Two themes, one set of token names: a charcoal dark and a warm off-white
light. Nothing below the token blocks in `globals.css` names a literal colour,
so the second theme was a block of tokens rather than an audit of the file.
The system preference decides; the control in the header overrides it and the
override is stored per browser. `layout.tsx` applies a stored choice in a
blocking inline script so a return visit never flashes the wrong theme.

The timeline is drawn in the browser from the same `/api/runs/{id}/segments`
rows the table renders, in stage 8's own colours: hovering a band gives the
sentence that produced it, clicking one scrolls to and highlights its row, and
a legend chip filters the table and dims every other band at once — which is
how you see where one class actually falls across a 23-minute video. Stage 8's
PNG is still one click away under "Report figure"; it is the same bytes that
go in the report, and two renderings are two things that can disagree.

`motion` (framer-motion) is the only UI dependency: the tooltip's enter and
exit, and the count-up on the headline figures. Everything else is CSS, which
is also why a 311-row table still scrolls — animating each row as a component
would not.

The raw pipeline log is folded away behind a disclosure, open by default only
when the run failed. It is the first thing wanted when something breaks and
the last thing wanted otherwise. Above it, `lib/logs.ts` pulls the line that
actually explains the failure out of the log and shows that instead of
`pipeline exited with code 1` — "Sign in to confirm your age", "Refusing to
start: checkpoints ... are missing", "libcudnn.so.9: cannot open shared object
file".

Motion marks change and nothing else — a stage becoming active, a bar reaching
its value, rows arriving. Durations are 140–420 ms on one easing curve, the
in-progress bar carries a slow sheen so a stage that holds for an hour does not
look like a stalled one, and everything is disabled under
`prefers-reduced-motion`, delays included.

`NEXT_PUBLIC_API_URL` is read **by the browser**, not by the Next server. It
has to be an address you can reach — `http://127.0.0.1:8000` — not the compose
service name, because the fetching happens on your machine.

It is a **build argument**, not a runtime environment variable. Next inlines
every `NEXT_PUBLIC_*` value into the client bundle during `next build`, so
setting it on the running container changes nothing and the browser keeps
calling whatever was compiled in — a failure that looks like the API being
down. To change it, edit `compose.yaml`'s `build.args` and rebuild:

```bash
docker compose build frontend
```

The build greps the compiled chunks for the value and fails if it is not
there, so a bundle that would silently call the wrong address never ships.

Live progress uses server-sent events (`GET /api/jobs/{id}/events`). The
response sets `X-Accel-Buffering: no`; without it an intermediate proxy buffers
the stream and the progress bar only moves once the run has finished. The log
view keeps the last 500 lines: a full run emits tens of thousands, and holding
them all grows the DOM until the tab stalls.

## Troubleshooting

**`docker: 'compose' is not a docker command`** — you have the old
`docker-compose` script. The Dockerfile uses BuildKit bind mounts and compose
profiles, both of which need the plugin.

**The GPU is invisible inside the container.** Install the NVIDIA Container
Toolkit, then `sudo nvidia-ctk runtime configure --runtime=docker && sudo
systemctl restart docker`. Confirm with
`docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L`.

**The submit form is disabled.** `/api/health` reports
`can_run_pipeline: false` when a checkpoint a stage actually loads is missing.
Run `scripts/fetch_va_checkpoint.sh`. Only the four checkpoints in
`STAGE_MODELS` block a run; the rest are listed as unused.

**The frontend loads but every panel says the API is unreachable.** The browser
is using `NEXT_PUBLIC_API_URL`, which is baked in at container start. Check the
value in `compose.yaml` matches the port you published.

**A run shows as failed straight after a restart.** That is the orphan sweep:
the process died with the container, and the job record is corrected rather
than left claiming to be running.
