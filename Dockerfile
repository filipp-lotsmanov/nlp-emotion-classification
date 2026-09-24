# syntax=docker/dockerfile:1.9
#
# Two images from one file, because the project splits that way already:
#
#   --target viewer   API + the run reader. Reads finished runs out of /data,
#                     serves the timeline and the segment table. No ffmpeg,
#                     so it cannot run stage 1; but `--extra api` still pulls
#                     the core dependencies, torch among them, so it is not
#                     small. Runs anywhere, including a laptop with no GPU,
#                     which is what makes it demoable.
#
#   --target app      everything above plus ffmpeg, Whisper, NLLB and the
#                     classifiers. Runs the whole nine-stage pipeline on a
#                     video of your own. Several GB, because CUDA kernels are.
#
#     docker build --target viewer -t vea:viewer .
#     docker build --target app    -t vea:app .
#
# `app` still starts without a GPU: torch reports no CUDA and every stage falls
# back to the CPU. A two-hour recording then takes most of a day rather than
# most of an hour, so it is a way to prove the plumbing, not to get results.
#
# The project is installed into /app rather than as a bare wheel because the
# corpora and the docs are looked up relative to the package, and a wheel on
# its own would leave them behind.

ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.9

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# --- base ---------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_FROZEN=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app

# /data is the only path either image writes to. `downloads/` and `models/` are
# symlinks into it because that is where the CLI's own defaults point, so a
# bind-mounted volume survives every rebuild with the model cache intact.
# `models_cache/` is where stage 5B caches NLLB (translate.py), relative to the
# working directory. Without the link, ~17 GB lands in the container layer and is
# downloaded again every time the container is recreated. It targets
# /data/models rather than a new directory because a volume created before this
# line has no new directory, and a dangling link would fail stage 5B.
RUN useradd --create-home --uid 10001 vea \
 && install -d -o vea -g vea \
      /data /data/downloads /data/models /data/jobs /data/cache /data/cache/matplotlib \
 && ln -s /data/downloads /app/downloads \
 && ln -s /data/models /app/models \
 && ln -s /data/models /app/models_cache

ENV VEA_DATA_DIR=/data \
    VEA_MODELS_DIR=/data/models \
    MPLCONFIGDIR=/data/cache/matplotlib

# uv is mounted for the length of each RUN and never copied, so it costs the
# shipped image nothing. pyproject and the lock are mounted rather than COPYed
# for the same reason and one better: editing the README no longer invalidates
# the dependency layer, which is almost the whole of both images.
#
# Two `uv sync` calls per target, always. The first resolves from the lock
# alone and stays cached until the lock changes; the second installs the
# project itself, which is small and changes constantly.

# --- viewer -------------------------------------------------------------------
FROM base AS viewer-deps
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --no-install-project --extra api

FROM viewer-deps AS viewer
COPY --link README.md LICENSE ./
COPY --link src/ ./src/
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --extra api

# After the install, so editing a document rebuilds only its own layer.
COPY --link docs/ ./docs/

LABEL org.opencontainers.image.title="Video Emotion Analysis (viewer)" \
      org.opencontainers.image.description="Reads finished runs. No models, no GPU."

USER vea
EXPOSE 8000
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/health')"

# 0.0.0.0 binds inside this container's own network namespace and nowhere else.
# Publish it to loopback - `-p 127.0.0.1:8000:8000` - because the API hands a
# URL to yt-dlp and has no authentication. See src/vea/api/app.py.
ENTRYPOINT ["vea"]
CMD ["serve", "--host", "0.0.0.0", "--downloads", "/data/downloads"]

# --- app ----------------------------------------------------------------------
# From `base`, not from `viewer`: inheriting it would leave the viewer's whole
# virtual environment in the image underneath the one that replaces it, and a
# layer that is overwritten is still a layer that is pulled.
FROM base AS app-deps

# ffmpeg is what stage 1 (yt-dlp) uses to turn a download into the 16 kHz mono
# that stage 3 expects.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
 && apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --no-install-project --extra api --extra train

# Stage 4 needs one CUDA 12 library that torch does not provide.
#
# faster-whisper runs on CTranslate2, whose wheels are built against CUDA 12:
# at runtime they load libcublas.so.12 and libcudnn.so.9 by name. torch now
# resolves to a CUDA 13 build, which provides libcublas.so.**13**. Different
# soname, so the loader finds nothing usable and stage 4 dies with:
#
#   RuntimeError: Library libcublas.so.12 is not found or cannot be loaded
#
# Only cuBLAS is actually missing. cuDNN's soname is libcudnn.so.9 for both
# CUDA majors and torch already ships it - it is simply on torch's own loader
# path and nowhere the system linker looks.
#
# WHAT NOT TO DO, because an earlier version of this file did it and shipped a
# broken image: `uv pip install nvidia-cudnn-cu12` into torch's environment.
# That wheel installs to nvidia/cudnn/lib/libcudnn.so.9 - the exact path
# torch's own cuDNN occupies - so the install replaces the distribution owning
# that file and torch loses a library it links at import:
#
#   ImportError: libcudnn.so.9: cannot open shared object file
#
# The whole image then fails before stage 1, on a CPU-only machine that was
# never going to touch CUDA. So: install nothing into the venv, take only
# cuBLAS, and put it in its own prefix where it cannot collide with anything.
# scripts/setup_ctranslate2_cuda.sh has installed outside the venv from the
# start, for this reason; this file had not caught up.
#
# CUDA_RUNTIME=0 skips it entirely. Nothing here is loaded unless stage 4 runs
# on a card, so on a CPU-only machine it is ~600 MB bought for nothing.
ARG CUDA_RUNTIME=1

RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    if [ "$CUDA_RUNTIME" = "1" ]; then \
        uv pip install --target /opt/cuda12 nvidia-cublas-cu12; \
    else \
        echo "CUDA_RUNTIME=0: skipping the CUDA 12 runtime for CTranslate2"; \
    fi

# torch keeps its bundled NVIDIA libraries on its own loader path and nowhere
# else, so a library loaded by name from outside torch cannot see them. Listing
# both torch's directories and the cuBLAS 12 prefix in ld.so.conf fixes that
# for CTranslate2 without an LD_LIBRARY_PATH every caller has to remember.
#
# The checks are the point of the step. The soname greps catch a missing
# library at build time instead of an hour into a run - and the torch import
# catches the far worse case this block used to cause, where the patch itself
# breaks the thing it was patching. A build that cannot import torch is not a
# build worth pushing.
RUN if [ "$CUDA_RUNTIME" = "1" ]; then \
        printf '%s\n' /opt/venv/lib/python3*/site-packages/nvidia/*/lib \
                      /opt/cuda12/nvidia/*/lib \
          > /etc/ld.so.conf.d/nvidia-from-pip.conf \
     && ldconfig \
     && ldconfig -p | grep -q 'libcublas\.so\.12' \
     && ldconfig -p | grep -q 'libcudnn\.so\.9'; \
    fi \
 && python -c "import torch, ctranslate2; print('torch', torch.__version__, '| ctranslate2', ctranslate2.__version__)"

FROM app-deps AS app
COPY --link README.md LICENSE ./
COPY --link src/ ./src/
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --extra api --extra train

COPY --link docs/ ./docs/
COPY --link corpora/ ./corpora/
COPY --link training/ ./training/
COPY --link scripts/ ./scripts/

# Both caches go to the volume, so the first run's model downloads (over 20 GB
# with the default nllb-3.3b) survive into the second.
ENV HF_HOME=/data/models \
    XDG_CACHE_HOME=/data/cache

LABEL org.opencontainers.image.title="Video Emotion Analysis (full pipeline)" \
      org.opencontainers.image.description="All nine stages, from a YouTube link."

USER vea
EXPOSE 8000
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/health')"

ENTRYPOINT ["vea"]
CMD ["serve", "--host", "0.0.0.0", "--downloads", "/data/downloads"]
