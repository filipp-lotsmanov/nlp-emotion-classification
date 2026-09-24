# Setup

Everything here runs the pipeline **locally** — your own machine, your own
video, no remote server. Three platforms, side by side.

Pick one of two routes:

| | Docker | Native |
|---|---|---|
| Install burden | Docker Desktop, nothing else | Python, uv, ffmpeg, Node |
| Reproducible | yes, same image everywhere | depends on your machine |
| GPU | Linux and Windows (WSL2) with an NVIDIA card | wherever torch finds one |
| Best for | running it | changing it |

**Docker is the recommended route.** The native route is faster to iterate on
if you are editing the code.

---

## What you need before either route

### The two checkpoints

Stages 6A, 6B and half of 7B will not run without them. Everything else
downloads itself from Hugging Face on first use.

| Checkpoint | Directory | Where it comes from |
|---|---|---|
| `va-xlmroberta-large` | `models/xlmroberta-base-va/` | `scripts/fetch_va_checkpoint.sh` |
| `emotion-en-deberta` | `models/emotion-en-deberta/` | `scripts/fetch_emotion_en_checkpoint.sh` |

Both are published as release assets on this repository, so neither needs a GPU
and neither depends on anyone else's hosting staying up:

```bash
./scripts/fetch_va_checkpoint.sh
./scripts/fetch_emotion_en_checkpoint.sh
```

Each downloads the weights, verifies their SHA-256, and runs a contract check
that loads the checkpoint rather than trusting the file to be what it claims. A
mismatched download is deleted rather than left to be picked up as a cache hit
on the next run. The VA script additionally patches the missing `id2label` into
`config.json`. On Windows, run both from **Git Bash**, not PowerShell.

About 1.7 GB between them. Re-running either is safe, but they treat an
existing checkpoint differently. `fetch_emotion_en_checkpoint.sh` refuses
(exit 3) if the directory already exists — pass `--force` if replacing it is
what you meant. `fetch_va_checkpoint.sh` has no `--force`: it keeps any file
already present and non-empty, downloads only the missing ones, then checks the
weights' SHA-256 and deletes `model.safetensors` on a mismatch, so a second run
fetches it fresh.

On the Docker route these land in `./models` on the host, which the containers
do not see — see [Getting the checkpoints into the container](#getting-the-checkpoints-into-the-container).

`./scripts/fetch_emotion_en_checkpoint.sh --balanced` fetches a second,
class-weighted run of the same model. No stage loads it; it is published so
`training/emotion_en_deberta/compare_runs.py` can reproduce the class-weighting
comparison without a GPU.

To rebuild `emotion-en-deberta` from scratch instead, `scripts/train_emotion_en.sh`
needs a GPU and several hours.

Check what you have at any point:

```bash
uv run vea models          # native
docker compose exec app vea models   # docker
```

Missing checkpoints that no stage loads are listed separately and do not block
a run. Only the two above do.

### Disk

Roughly 30 GB free for the full setup: the `app` image is several GB, and the
first run pulls Hub models (Whisper large-v3, NLLB, two classifiers, a sentence
embedder) — over 20 GB with the default NLLB-3.3B, which is ~17 GB on its own,
or about 6 GB with `nllb-600m` (see
[If your machine is short of memory](#if-your-machine-is-short-of-memory)).
They are cached in a Docker volume or under `~/.cache/huggingface`, so it is a
one-time cost.

---

## Docker

### Windows

**Install Docker Desktop** with the WSL2 backend (the installer default). Start
it and wait for the whale icon to settle before running anything.

```powershell
cd C:\path\to\nlp-emotion-classification
docker compose up -d
```

That builds and starts the **viewer** (read finished runs) and the frontend.
Open <http://127.0.0.1:3000>.

For the full pipeline:

```powershell
docker compose --profile full up -d app frontend
```

With an NVIDIA card, add the GPU overlay:

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml --profile full up -d app frontend
```

Only use the overlay if `docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L`
works. Without a visible adapter the daemon **refuses to create the
container** rather than falling back to the CPU.

No NVIDIA card? Build smaller and skip ~600 MB of CUDA 12 cuBLAS nothing will
load:

```powershell
$env:CUDA_RUNTIME="0"
docker compose --profile full build app
```

Windows notes:

- Use `curl.exe`, not `curl` — in PowerShell `curl` is an alias for
  `Invoke-WebRequest` and takes different arguments.
- `scripts/*.sh` need **Git Bash**. `.gitattributes` pins them to LF so Git's
  Windows default does not rewrite them to CRLF and produce
  `bad interpreter: /usr/bin/env bash^M`.
- The first `app` build is slow — one measured case took 28 minutes, most of it
  Docker Desktop unpacking layers through the WSL2 filesystem rather than the
  build itself. Rebuilds after a source edit are seconds.

### macOS

**Install Docker Desktop for Mac**, then:

```bash
cd ~/path/to/nlp-emotion-classification
docker compose up -d                              # viewer + frontend
docker compose --profile full up -d app frontend  # the whole pipeline
```

On **Apple Silicon**, build the pipeline image without the CUDA runtime —
there is no NVIDIA GPU to use it on and the cuBLAS wheel is x86-64:

```bash
CUDA_RUNTIME=0 docker compose --profile full build app
```

There is no GPU passthrough to Linux containers on macOS at all, so the
pipeline runs on the CPU. Metal/MPS is only reachable from a **native** install
(below), and even then not every stage uses it.

Give Docker Desktop more memory than the default if a stage is killed
mid-run: Settings → Resources → Memory, 8 GB or more.

### Linux

**Docker Engine plus the compose plugin** — `docker compose version` must work;
the old `docker-compose` script is not enough, because the Dockerfile uses
BuildKit bind mounts and the compose file uses profiles.

```bash
cd ~/path/to/nlp-emotion-classification
./scripts/setup_docker.sh            # viewer + frontend, checks the host first
./scripts/setup_docker.sh --full     # the whole pipeline
./scripts/setup_docker.sh --down     # stop, keep the data
```

The script checks docker, the compose plugin and BuildKit, probes for a usable
GPU and attaches one only if it finds it, builds, starts, and waits for
`/api/health` to answer rather than printing a URL that is not live yet.

For the GPU you need the **NVIDIA Container Toolkit**:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L
```

Without the script:

```bash
docker compose --profile full up -d app frontend                                   # CPU
docker compose -f compose.yaml -f compose.gpu.yaml --profile full up -d app frontend  # GPU
```

### Getting the checkpoints into the container

The fetch scripts write to `./models` on the host. The `app` container reads
checkpoints from `/data/models` inside the `vea_vea-data` volume, and `models/`
is excluded from the image build, so a host fetch alone leaves
`can_run_pipeline: false`. Copy both directories in once `app` is running, then
hand them to the container's user (uid 10001), as `setup_docker.sh --seed` does
for runs:

```bash
docker compose --profile full cp models/xlmroberta-base-va app:/data/models/
docker compose --profile full cp models/emotion-en-deberta app:/data/models/
docker compose --profile full exec -u 0 app chown -R 10001:10001 /data/models/xlmroberta-base-va /data/models/emotion-en-deberta
docker compose --profile full exec app vea models
```

They live in the volume from then on, so this survives rebuilds. `/api/health`
re-checks on every request; no restart is needed.

### Putting a finished run in front of the viewer

The viewer reads runs out of the `vea_vea-data` volume. To look at one produced
elsewhere:

```bash
docker compose cp ./downloads/video-XXXXXXXXXXX viewer:/data/downloads/
```

or, with the script, `./scripts/setup_docker.sh --seed downloads/`.

A run needs only `emotion_analysis_data.csv`; `emotion_timeline.png` is shown
when present.

---

## Native

### Windows

1. **Python 3.11–3.13** from python.org.
2. **uv**:
   ```powershell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
3. **ffmpeg** — `winget install Gyan.FFmpeg`, then reopen the terminal so it is
   on `PATH`. Stage 1 fails without it (yt-dlp uses it to extract the audio).
4. **Node 20.9 or newer** from nodejs.org, for the frontend.

```powershell
uv sync --extra api --extra train
uv run vea config
```

`vea config` prints the resolved paths and whether torch sees a GPU. On Windows
the PyPI wheel is **CPU-only**, so expect `CUDA available: False`. That is
correct, not a misconfiguration — see the README for why CUDA on Windows is not
wired up here.

### macOS

```bash
brew install uv ffmpeg node
cd ~/path/to/nlp-emotion-classification
uv sync --extra api --extra train
uv run vea config
```

torch installs the CPU/MPS build. **Apple Silicon only**: the torch version in
`uv.lock` publishes no Intel (`x86_64`) macOS wheel, and its arm64 wheel wants
macOS 14 or later. On an Intel Mac, use the Docker route. Expect CPU-class
speed.

### Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo apt install ffmpeg          # Debian/Ubuntu
cd ~/path/to/nlp-emotion-classification
uv sync --extra api --extra train
uv run vea config
```

On Linux the PyPI torch wheel is a **CUDA** build, so a machine with an NVIDIA
driver gets GPU acceleration with no extra configuration. `vea config` lists the
cards it can see.

If stage 4 dies with `Library libcublas.so.12 is not found`, run
`./scripts/setup_ctranslate2_cuda.sh` — CTranslate2 is built against CUDA 12
while torch now ships CUDA 13, and the script installs the missing runtime
beside it without disturbing torch.

### Running it, natively, on any of the three

Two terminals.

```bash
uv run vea serve            # API on http://127.0.0.1:8000
```

```bash
cd frontend
npm install
npm run dev                 # UI on http://localhost:3000
```

Open the UI as `localhost`, not `127.0.0.1`. The Next.js dev server only trusts
`localhost` by default and refuses its dev WebSocket to any other host name, a
bare IP included. The page still renders, but React never hydrates it, so the
**Run pipeline** button stays disabled whatever you type.

On a remote machine, forward both ports over SSH and use the same address from
your own browser:

```bash
ssh -L 3000:localhost:3000 -L 8000:localhost:8000 <user>@<server>
```

Port 8000 is needed as well: the page calls the API from your browser, at
`http://127.0.0.1:8000`, which the tunnel carries to the server.

Or skip the web interface entirely:

```bash
uv run vea run "https://www.youtube.com/watch?v=VIDEO_ID"
```

---

## First run

Open <http://localhost:3000>, paste a Russian-language YouTube URL, press
**Run pipeline**, and watch the stage strip.

**Start with a three-to-five minute video.** You are proving that nine stages
wire together; find that out in minutes rather than discovering at stage 7 that
something was wrong.

What to expect:

- Long quiet periods on the very first run while models download into the
  cache — over 20 GB with the default NLLB-3.3B, about 6 GB with `nllb-600m`.
- `Using int8 compute_type for CPU` in the log at stage 4 on a machine with no
  GPU. Expected: the transcriber adapting, not an error.
- On CPU, transcription and translation dominate the wall clock. The progress
  bar counts **stages, not time**, and says so, because the stages are wildly
  uneven.

### If your machine is short of memory

Stage 5B loads NLLB-3.3B by default: roughly 17 GB of weights, wanting about
8 GB of VRAM. On a laptop, pick a smaller one:

```bash
# Linux / macOS
VEA_TRANSLATION_MODEL=nllb-600m docker compose --profile full up -d app frontend
```

```powershell
# Windows
$env:VEA_TRANSLATION_MODEL="nllb-600m"
docker compose --profile full up -d app frontend
```

Three sizes are selectable:

| `VEA_TRANSLATION_MODEL` | Weights | Notes |
|---|---|---|
| `nllb-3.3b` | ~17 GB | the default; wants ~8 GB of VRAM |
| `nllb-1.3b` | ~5 GB | the middle option |
| `nllb-600m` | ~2.5 GB | runs on a CPU |

Translation quality drops with size, so use `nllb-3.3b` for anything that goes
in a report. An unrecognised name is rejected outright rather than quietly
falling back to the 17 GB default.

---

## Verifying

```bash
curl http://127.0.0.1:8000/api/health     # curl.exe on Windows PowerShell
```

```json
{"ok": true, "downloads": "...", "can_run_pipeline": true, "missing_checkpoints": []}
```

`can_run_pipeline: false` means a checkpoint a stage actually loads is missing —
the response names which. The submit form in the UI disables itself in that
case rather than letting you start a run that cannot finish.

---

## Security

**The API has no authentication.** It takes a URL from the caller, hands it to
yt-dlp, and spends up to hours of compute on the result. Anyone who can reach
port 8000 can make your machine fetch a URL of their choosing.

Everything binds and publishes to `127.0.0.1` for that reason. The
`--host 0.0.0.0` in the container's own command binds inside the container's
network namespace; what reaches the outside is the `ports:` mapping in
`compose.yaml`, and that says loopback. Do not widen it without putting
something in front that asks who is calling.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `docker: 'compose' is not a docker command` | You have the old `docker-compose` script. Install the plugin. |
| `nvidia-container-cli: ... no adapters were found` | You used `compose.gpu.yaml` without a usable NVIDIA GPU. Drop the overlay; the image runs fine on the CPU. |
| `bad interpreter: /usr/bin/env bash^M` | A script got CRLF line endings. `.gitattributes` prevents it on a fresh clone; re-checkout the file. |
| `Library libcublas.so.12 is not found` | Stage 4 on a GPU without the CUDA 12 runtime. `./scripts/setup_ctranslate2_cuda.sh`, or rebuild the image with `CUDA_RUNTIME=1`. |
| `Sign in to confirm your age` | The video is age-restricted. yt-dlp has no session. Use a different video. |
| Submit form disabled | A required checkpoint is missing; the notice names it. |
| Every panel says the API is unreachable | `NEXT_PUBLIC_API_URL` is baked into the frontend at **build** time. Change it in `compose.yaml`'s `build.args` and `docker compose build frontend`. |
| A run shows as failed right after a restart | The orphan sweep. The process died with the container, and the record is corrected rather than left claiming to be running. |
| `address already in use` on 3000 or 8000 | Something else holds the port — usually an older container. `docker compose --profile full down`. |

Deeper container detail, including the CUDA soname split and the build layout,
is in [docs/CONTAINER.md](docs/CONTAINER.md).
