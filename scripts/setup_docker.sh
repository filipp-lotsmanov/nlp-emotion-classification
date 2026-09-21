#!/usr/bin/env bash
# Bring the containerised stack up, from nothing.
#
#   ./scripts/setup_docker.sh                 viewer + frontend (no GPU needed)
#   ./scripts/setup_docker.sh --full          the full pipeline instead
#   ./scripts/setup_docker.sh --full --cpu    the same, card never attached
#   ./scripts/setup_docker.sh --seed downloads/   copy finished runs into the volume
#   ./scripts/setup_docker.sh --down          stop everything, keep the data
#
#   --gpu is an accepted alias for --full. Whether a card is attached is not
#   decided by the flag: the script probes for one, because the reservation is
#   not a preference the daemon can decline - with no adapter it refuses to
#   create the container at all.
#
# WHAT IT DOES
#
#   1. checks docker, the compose plugin and BuildKit
#   2. probes for a usable GPU, but only for the full pipeline
#   3. builds the images the chosen profile needs
#   4. seeds the /data volume with runs you already have, if asked
#   5. starts the stack and waits for /api/health to answer
#
# WHICH PROFILE
#
# The default is `viewer`: the read-only API plus the frontend. It has no
# torch, no CUDA and no ffmpeg, it builds in a couple of minutes, and it can
# show every run already in the volume. That is the demo.
#
# `--full` builds the `app` image instead - several GB, and its first run pulls
# roughly 4 GB of model weights. Use it to analyse a video of your own. With no
# card it runs on the CPU at roughly a tenth of the speed, and the build then
# also skips the 1.4 GB CUDA 12 runtime that only a card would load.
#
# Both publish to 127.0.0.1 only. The API has no authentication and it hands a
# caller-supplied URL to yt-dlp, so anyone who can reach port 8000 can spend
# your GPU on a video of their choosing. Do not publish it wider without
# putting something in front of it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROFILE="viewer"
FORCE_CPU=0
BUILD=1
SEED=""
DOWN=0
WAIT_SECONDS=120

# The compose project is named `vea`, so the volume is `vea_vea-data`.
VOLUME="vea_vea-data"
API="http://127.0.0.1:8000"
WEB="http://127.0.0.1:3000"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full|--app) PROFILE="app"; shift ;;
        # --gpu keeps working; it now means "the full pipeline image", and
        # whether a card is attached is decided by probing for one.
        --gpu) PROFILE="app"; shift ;;
        --cpu) PROFILE="app"; FORCE_CPU=1; shift ;;
        --viewer) PROFILE="viewer"; shift ;;
        --no-build) BUILD=0; shift ;;
        --seed) SEED="${2:?--seed needs a directory}"; shift 2 ;;
        --down) DOWN=1; shift ;;
        --wait) WAIT_SECONDS="${2:?--wait needs seconds}"; shift 2 ;;
        -h|--help) sed -n '2,37p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  ok    %s\n' "$*"; }
warn() { printf '  warn  %s\n' "$*"; }
die()  { printf '\n  FAIL  %s\n\n' "$*" >&2; exit 1; }

# --- 1. prerequisites ---------------------------------------------------------
say "Checking prerequisites"

command -v docker >/dev/null 2>&1 \
    || die "docker is not installed. https://docs.docker.com/engine/install/"

docker info >/dev/null 2>&1 \
    || die "the docker daemon is not reachable. Start Docker Desktop, or add yourself to the docker group and log in again."
ok "docker daemon reachable"

# `docker-compose` (the old Python script) is not enough: this file uses
# profiles and `--mount=type=bind` in the Dockerfile, which need the plugin.
docker compose version >/dev/null 2>&1 \
    || die "the compose plugin is missing. 'docker compose version' must work; 'docker-compose' is not the same thing."
ok "compose plugin: $(docker compose version --short 2>/dev/null || echo present)"

# BuildKit does the cache mounts and the bind mounts in the Dockerfile. It is
# the default from Docker 23 on, so this is a check rather than a step.
if [[ "${DOCKER_BUILDKIT:-1}" == "0" ]]; then
    die "DOCKER_BUILDKIT=0 is set; the Dockerfile needs BuildKit. Unset it and try again."
fi
ok "BuildKit enabled"

if [[ $DOWN -eq 1 ]]; then
    say "Stopping"
    # --profile full so the app container is included even when the last start
    # was a viewer one; compose ignores a profile with nothing running in it.
    docker compose --profile full down
    printf '\n  The %s volume is untouched, so your runs and model cache survive.\n' "$VOLUME"
    printf '  Remove them with: docker volume rm %s\n\n' "$VOLUME"
    exit 0
fi

# --- 2. the GPU, only if it was asked for -------------------------------------
# The reservation is only added once a card is proven reachable. It is not a
# preference the daemon can decline: with no adapter it refuses to create the
# container ("no adapters were found"), so guessing wrong costs a start,
# not a slowdown.
GPU_ARGS=()
if [[ "$PROFILE" == "app" ]]; then
    GPU_ARGS=(--profile full)
    say "Checking the GPU wiring"

    if [[ $FORCE_CPU -eq 1 ]]; then
        ok "--cpu: not attaching a card"
        export CUDA_RUNTIME=0
    elif ! command -v nvidia-smi >/dev/null 2>&1; then
        warn "no nvidia-smi on the host: building and running on the CPU."
        warn "a two-hour recording then takes most of a day rather than most of an hour."
        # Nothing will load libcublas.so.12, so do not ship 1.4 GB of it.
        export CUDA_RUNTIME=0
    elif docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L >/dev/null 2>&1; then
        ok "containers can see the GPU: $(nvidia-smi -L | head -1)"
        GPU_ARGS=(-f compose.yaml -f compose.gpu.yaml --profile full)
    else
        # The card is there and the daemon cannot reach it, which is one
        # specific missing piece almost every time.
        warn "the host has a GPU but containers cannot reach it."
        warn "Linux: install the NVIDIA Container Toolkit, then"
        warn "  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
        warn "Windows: update the NVIDIA driver; Docker Desktop needs the WSL2 backend."
        warn "continuing on the CPU - the card is not attached, so the container still starts."
        export CUDA_RUNTIME=0
    fi
    if [[ "${CUDA_RUNTIME:-1}" == "0" ]]; then
        ok "CPU build: skipping the 1.4 GB CUDA 12 runtime"
    fi
fi

SERVICES=("$PROFILE" frontend)

# --- 3. build -----------------------------------------------------------------
if [[ $BUILD -eq 1 ]]; then
    say "Building ${SERVICES[*]}"
    printf '  First build: a few minutes for viewer, considerably longer for app.\n'
    docker compose "${GPU_ARGS[@]}" build "${SERVICES[@]}"
else
    ok "skipping the build (--no-build)"
fi

# --- 4. seed the volume -------------------------------------------------------
# A run the pipeline produced outside Docker lives in ./downloads/video-<id>/.
# Copying it in is what lets the viewer show real results on a laptop with no
# GPU, which is the whole point of having a viewer image.
if [[ -n "$SEED" ]]; then
    say "Seeding runs from $SEED"
    [[ -d "$SEED" ]] || die "$SEED is not a directory"

    runs=("$SEED"/video-*/)
    if [[ ! -d "${runs[0]:-}" ]]; then
        warn "no video-*/ directories in $SEED; nothing to seed"
    else
        # The volume has to exist before anything can be copied into it, and
        # `up --no-start` creates it without running a container.
        docker compose "${GPU_ARGS[@]}" up --no-start "$PROFILE" >/dev/null

        # root, because the volume's /data is owned by uid 10001 inside the
        # image and the files are then chowned back to it. --entrypoint sh
        # because the image's entrypoint is the `vea` CLI.
        #
        # MSYS_NO_PATHCONV stops Git Bash on Windows from "helpfully" rewriting
        # the container-side half of a -v argument into a Windows path: without
        # it, `-v ...:/seed:ro` arrives as `C:/Program Files/Git/seed` and the
        # mount fails with an error that names a directory nobody wrote.
        MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' \
        docker run --rm \
            -v "${VOLUME}:/data" \
            -v "$(cd "$SEED" && pwd):/seed:ro" \
            --user root --entrypoint sh \
            "vea:${PROFILE}" -c '
                set -e
                mkdir -p /data/downloads
                cp -r /seed/video-* /data/downloads/
                chown -R 10001:10001 /data/downloads
                ls -d /data/downloads/video-* | wc -l
            ' | while read -r count; do ok "$count run(s) in the volume"; done
    fi
fi

# --- 5. start -----------------------------------------------------------------
say "Starting ${SERVICES[*]}"
docker compose "${GPU_ARGS[@]}" up -d "${SERVICES[@]}"

printf '  Waiting for the API'
deadline=$(( $(date +%s) + WAIT_SECONDS ))
healthy=0
while [[ $(date +%s) -lt $deadline ]]; do
    if curl -fsS "${API}/api/health" >/dev/null 2>&1; then healthy=1; break; fi
    printf '.'
    sleep 2
done
printf '\n'

if [[ $healthy -eq 0 ]]; then
    printf '\n'
    warn "the API did not answer within ${WAIT_SECONDS}s. Recent output:"
    docker compose "${GPU_ARGS[@]}" logs --tail 40 "$PROFILE"
    die "see the log above; 'docker compose logs -f $PROFILE' follows it."
fi
ok "API healthy"

# can_run_pipeline is false when a checkpoint is missing, and the frontend
# disables the submit form when it is. Saying so here is cheaper than letting
# someone discover it by submitting.
if command -v python3 >/dev/null 2>&1; then
    missing="$(curl -fsS "${API}/api/health" \
        | python3 -c 'import json,sys; print(",".join(json.load(sys.stdin)["missing_checkpoints"]))' 2>/dev/null || true)"
    if [[ -n "$missing" ]]; then
        warn "missing checkpoints: $missing"
        warn "the viewer still reads finished runs; fetch them with scripts/fetch_va_checkpoint.sh to run new ones."
    else
        ok "all required checkpoints present"
    fi
fi

cat <<EOF

  frontend   ${WEB}
  API docs   ${API}/docs

  logs       docker compose ${GPU_ARGS[*]} logs -f ${PROFILE}
  stop       ./scripts/setup_docker.sh --down

  Bound to 127.0.0.1 only: the API has no authentication and it runs yt-dlp on
  whatever URL it is given. See docs/CONTAINER.md.

EOF
