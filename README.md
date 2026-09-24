# Video Emotion Analysis (Russian)

A nine-stage pipeline that takes a Russian-language YouTube URL and produces an
emotion timeline: download, scene detection, audio normalisation, Whisper
transcription, semantic segmentation, NLLB translation, valence-arousal
regression in both languages, 7-class emotion classification in both languages,
a timeline plot, and a consolidated CSV.

**Setting it up on your own machine: [SETUP.md](SETUP.md)** — Windows, macOS
and Linux side by side, Docker or native.

Full stage-by-stage design notes: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Current state

This is a revived repository. The pipeline code is the group's original
implementation; the packaging, configuration, tests and provenance
documentation around it are new. Read this section before running anything.

| Stages | Status | Notes |
| --- | --- | --- |
| 1-5B (download → translation) | Runnable | All models download from the Hub on first use |
| 6A, 6B (valence-arousal) | Runnable after fetch | `scripts/fetch_va_checkpoint.sh`. The checkpoint is a published upstream model, pinned by digest. Its arousal head is weak — AUC 0.5734 on this project's own classes, see [PROVENANCE](docs/PROVENANCE.md) section 11 |
| 7A (Russian emotion) | Runnable | Uses a third-party Hub model, not the group's own |
| 7B (English emotion) | Runnable after fetch | `scripts/fetch_emotion_en_checkpoint.sh`. Both ensemble slots are real: DistilRoBERTa and the retrained DeBERTa-v3-base (macro F1 0.8162) |
| 8, 9 (timeline, CSV) | Runnable | Consume stage 6/7 output; the significance gate inherits 6A's weak arousal |

Two checkpoints the pipeline expects were absent from the archive, along with
the code that produced them. Both gaps are now closed:

- **`va-xlmroberta-large`** (stages 6A and 6B) — an XLM-RoBERTa-large
  valence-arousal regressor. Stages 7A and 7B gate on its output, so nothing
  from stage 6 onward ran without it. Found published upstream and mirrored
  here; retraining it from scratch is still blocked on a valence-arousal
  corpus, see [training/README.md](training/README.md).
- **`emotion-en-deberta`** (stage 7B) — the DeBERTa classifier documented in
  [docs/model_cards/emotion_en_deberta.md](docs/model_cards/emotion_en_deberta.md).
  Retrained here on 2026-09-16 from the committed corpus.

Both are release assets on this repository:

```bash
./scripts/fetch_va_checkpoint.sh
./scripts/fetch_emotion_en_checkpoint.sh
```

Check what your machine actually has:

```bash
vea models
```

**Provenance warning.** The original orchestrator filled the DeBERTa ensemble
slot with `tae898/emoberta-large` — a third-party RoBERTa-large trained on MELD
— while still writing its output to files named `emotion_deberta-finetuned_*`.
Any result previously reported as coming from "our fine-tuned DeBERTa" was
produced by EmoBERTa instead, and any output generated before the switch must
be regenerated before it is reported. The slot now runs the real checkpoint,
retrained here on 2026-09-16: macro F1 0.8162 against the card's 0.8127,
accuracy 0.9202 against 0.8995. Details and the full audit:
[docs/PROVENANCE.md](docs/PROVENANCE.md).

`docs/evaluation/error_analysis.md` is regenerated against both released
checkpoints and agrees with the figures above. `interpretability_xai.md` is regenerated
against the same checkpoint; its integrated-gradients runs do not satisfy
completeness, which that document states and explains.

## Install

Per-platform instructions with prerequisites, the container route and
troubleshooting are in **[SETUP.md](SETUP.md)**. The short version:

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11-3.13.

```bash
uv sync --extra api                  # runtime plus `vea serve` (the web API)
uv sync --extra api --extra train    # plus retraining dependencies
```

Other extras: `baselines` (classical baselines: imbalanced-learn, TextBlob) and
`xai` (interpretability: captum). Name every extra you want in one command —
`uv sync` is exact, so it removes any extra you leave off, and running
`uv sync --extra train` after `uv sync --extra api` leaves you without the API.
A bare `uv sync` gives the pipeline and CLI with no `vea serve`.

Stage 1 needs `ffmpeg` on the PATH (`apt install ffmpeg`).

### Which torch you get

Nothing needs configuring, because the PyPI wheel already differs per platform:

| Platform | What `uv sync` installs |
| --- | --- |
| Linux | a **CUDA** build — the wheel depends on `nvidia-cudnn-cu13`, `nvidia-nccl-cu13` and `triton` under `sys_platform == 'linux'` |
| Windows | **CPU-only**; `vea config` reports `+cpu` |
| macOS | CPU/MPS, **Apple Silicon only** (see [Platforms](#platforms)) |

So a Linux GPU box needs no special handling and a Windows laptop is CPU by
construction. `uv run vea config` reports which build is actually loaded, the
visible GPUs and their VRAM.

If you ever need CUDA on *Windows*, that does require pulling torch from
PyTorch's own index rather than PyPI — add a `cu130` extra with a
`[tool.uv.sources]` entry and an explicit `[[tool.uv.index]]`. It is left out
here deliberately: it would make every `uv lock` depend on reaching
`download.pytorch.org`, for a configuration no machine in this project uses.

## Running on the GPU server

The BUAS server is an ephemeral container: `/` is an overlay filesystem and is
wiped on restart, while `/workspace` is a real volume. It also has 8 GPUs
shared with other tenants.

```bash
./scripts/setup_server.sh --train
```

That one command sets persistent cache locations, installs `ffmpeg`/`tmux`/`uv`,
picks the emptiest GPU, syncs, and verifies. It is idempotent, and you re-run it
after every container restart. In each new shell afterwards:

```bash
source /workspace/vea-env.sh
export CUDA_VISIBLE_DEVICES="$(scripts/pick_free_gpu.sh)"
```

Three things it handles that are easy to get wrong there:

- **Caches on persistent storage.** `HF_HOME` defaults to `~/.cache`, which is
  on the overlay; the script moves it to `/workspace`. Whisper large-v3 (~3 GB)
  goes there. NLLB-3.3B (~17 GB) does not: stage 5B caches it in
  `models_cache/` under the directory you run from, which survives only because
  the repo itself lives on `/workspace`. Run from the overlay and it is
  downloaded again after every restart.
- **GPU choice at runtime, never baked in.** `scripts/pick_free_gpu.sh` reads
  free VRAM and fails fast if nothing has room, rather than OOMing mid-run. The
  original code hardcoded GPU 5; when this project was revived, GPU 5 had
  47 GB of its 49 GB already allocated by someone else.
- **`nvidia-smi` under-reports.** Inside a container it prints "No running
  processes found" because other tenants' PIDs are invisible, while their
  memory is very much allocated. Free VRAM is the only honest signal.

Run long jobs under `tmux` so a dropped SSH session does not kill a 3-hour
fine-tune.

Stage 1 needs `ffmpeg` on the PATH:

```bash
sudo apt install ffmpeg      # Debian/Ubuntu
```

## Quick start

```bash
uv run vea config                    # show resolved settings and visible GPUs
uv run vea models                    # preflight: which weights are missing
uv run vea run "https://www.youtube.com/watch?v=VIDEO_ID"
```

`vea run` refuses to start while required checkpoints are missing, rather than
failing 20 minutes in at stage 6. To run the stages that do work:

```bash
uv run vea run "https://www.youtube.com/watch?v=VIDEO_ID" --allow-missing-models
```

Every stage writes into `downloads/video-{VIDEO_ID}/`, relative to the
directory you run the command from, and skips work whose output is already
present, so an interrupted run resumes instead of restarting.

## Web interface

Paste a URL, watch the nine stages live, read the timeline and the per-segment
table. Same pipeline, same vocabulary, same colours as the PNG.

```bash
uv run vea serve                     # API on http://127.0.0.1:8000 (needs --extra api)
cd frontend && npm install && npm run dev   # UI on http://localhost:3000
```

Open it as `localhost`: the dev server refuses other host names, `127.0.0.1`
included, and the page then never becomes interactive. On a remote server,
tunnel ports 3000 and 8000 over SSH. See [SETUP.md](SETUP.md#running-it-natively-on-any-of-the-three).

Or in containers, which is the easier path and the one that runs on a machine
with no GPU:

```bash
./scripts/setup_docker.sh            # viewer + frontend
./scripts/setup_docker.sh --gpu      # the full pipeline instead
```

Two images: `vea:viewer` reads finished runs; `vea:app` runs all nine stages.
The viewer is not a light image — it installs the core dependencies, torch
among them — but it has no ffmpeg, so a run submitted to it fails at stage 1.
Both serve the same API, so the frontend does not change between them. See
[docs/CONTAINER.md](docs/CONTAINER.md).

**The API has no authentication.** It takes a URL from the caller and hands it
to yt-dlp, then spends hours of compute on it. Everything binds and publishes
to `127.0.0.1` for that reason; do not widen it without putting an
authenticating proxy in front.

## Configuration

No GPU index is hardcoded, and checkpoint paths and the translation model
resolve through `src/vea/config.py`, driven by five environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `VEA_DATA_DIR` | `./data` | API job records (`jobs/`); runs go to `./downloads` |
| `VEA_MODELS_DIR` | `./models` | Locally trained checkpoints |
| `VEA_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, or `cuda:N` |
| `VEA_LOG_LEVEL` | `INFO` | Standard logging level name |
| `VEA_TRANSLATION_MODEL` | `nllb-3.3b` | Stage 5B size: `nllb-3.3b`, `nllb-1.3b`, `nllb-600m` |

`VEA_TRANSLATION_MODEL` is what makes the pipeline runnable off the server.
NLLB-3.3B is roughly 17 GB of weights and wants 8 GB of VRAM; `nllb-600m` is
about 2.5 GB and runs on a CPU. Translation quality drops accordingly — use
the default for anything that goes in the report. An unrecognised name raises
rather than quietly loading the 17 GB default.

Models are declared once, in `MODEL_REGISTRY` in the same file. Add or swap a
model there, not in a stage module.

Three things are still set in `src/vea/pipeline.py` rather than through the
environment: the runs directory (`downloads`, in `DEFAULT_CONFIGS`), the
Whisper size (`large-v3`, same place) and the sentence-embedding model name
(`paraphrase-multilingual-MiniLM-L12-v2`, in `main`). Change them there.

## Retraining the checkpoints

Both local checkpoints are published and fetched by the scripts in `scripts/`;
retraining is optional. See [training/README.md](training/README.md) for the reproduction plan, dataset
sources, and what is specified versus what has to be reconstructed by
experiment. Short version:

```
training/
├── va_regressor/         # rebuilds va-xlmroberta-large   (spec: external paper)
├── emotion_en_deberta/   # rebuilds emotion-en-deberta    (spec: the model card)
├── emotion_ru/           # the group's Russian classifier (complete, reproducible)
├── baselines/            # classical + transformer baselines on MELD
├── translation/          # the from-scratch translation experiments
└── interpretability/     # IG / LRP attribution for the XAI report
```

## Layout

```
src/vea/
├── config.py        Settings, model registry, device and logging resolution
├── cli.py           `vea config` / `vea models` / `vea run` / `vea serve`
├── pipeline.py      Stage orchestration and per-stage runners
├── api/             The HTTP API behind `vea serve` and the frontend
└── stages/          One module per stage
docs/
├── ARCHITECTURE.md      Stage-by-stage design notes
├── PROVENANCE.md        What was kept, dropped, and what the outputs really came from
├── LICENSING.md         Why the repository is CC BY-SA 4.0
├── model_cards/         Model documentation
└── evaluation/          Error analysis, interpretability, prompt engineering reports
tests/                   Runs without torch or any model download
```

## Platforms

CI runs the fast suite on **Linux, macOS and Windows**. That matrix is not
decoration: each runner is there for a class of bug the others cannot see.

- **Windows** is the only runner where `os.name == "nt"`. Cancelling a job
  called `os.killpg`, which does not exist there, so the route raised
  AttributeError and left the pipeline running. Docker hid it — the container
  is Linux whatever the host is. It is also where a file opened without an
  explicit encoding meets cp1252 and this project's Cyrillic output.
- **macOS** is POSIX but not Linux: case-insensitive filesystem, different
  temp layout, arm64 wheels.
- **Linux** is where the GPU server runs.

`uv.lock` carries wheels for `macosx_*_arm64`, `manylinux_*_{x86_64,aarch64}`
and `win_amd64` for both torch and CTranslate2. The exception is **Intel
macOS**: the locked torch 2.14 publishes no `macosx_*_x86_64` wheel, so a
native install on macOS needs Apple Silicon (the torch wheel is tagged for
macOS 14 or later). An Intel Mac can still use the containers.

Two things to know when running the container off x86-64 Linux: build with
`CUDA_RUNTIME=0`, since the CUDA 12 cuBLAS wheel is x86-64 and there is no
NVIDIA GPU to use it on, and expect CPU speeds.

## Development

```bash
uv run pytest                  # fast tests, no GPU, no downloads
uv run ruff check .
uv run ruff format .
```

The test suite includes static guards (`tests/test_source_hygiene.py`) against
the specific defects this revival fixed: import-time `CUDA_VISIBLE_DEVICES`
pinning, `logging.basicConfig` in library modules, `sys.exit()` from module
bodies, machine-specific absolute paths, and stage configs naming models that
are not in the registry. They fail loudly if any of it comes back.

## Licence

CC BY-SA 4.0, inherited from the super-emotion dataset. The NLLB models used in
stage 5B are non-commercial. See [LICENSE](LICENSE) and
[docs/LICENSING.md](docs/LICENSING.md) before redistributing anything.
