#!/usr/bin/env bash
# Print the index of the GPU with the most free memory.
#
# Composable, so it can set the variable in *your* shell:
#
#     export CUDA_VISIBLE_DEVICES="$(scripts/pick_free_gpu.sh)"
#
# This is the same selection rule as the BUAS server's /home/set_gpu.py, but
# usable: that script assigns os.environ['CUDA_VISIBLE_DEVICES'] inside its
# __main__ block, which only mutates its own process before it exits, so
# `python /home/set_gpu.py` has no effect on the caller.
#
# Why this matters on a shared box: `nvidia-smi` shows "No running processes
# found" inside a container because other tenants' PIDs are not visible, but
# their allocations are. Free memory is the only honest signal available.
#
# Options:
#   --min-free MIB   Refuse to pick a GPU with less than this free (default 20000).
#                    Exits 1 with a message on stderr if nothing qualifies, so
#                    a training run fails fast instead of OOMing ten minutes in.
#   --list           Print all GPUs with their free memory, then exit.

set -euo pipefail

MIN_FREE=20000
LIST_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --min-free) MIN_FREE="${2:?--min-free needs a value in MiB}"; shift 2 ;;
        --list) LIST_ONLY=1; shift ;;
        -h|--help) sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "error: nvidia-smi not found; no GPU on this machine" >&2
    exit 1
fi

# index, free MiB, total MiB, name - one row per GPU, no header, no units.
QUERY="index,memory.free,memory.total,name"
if ! rows="$(nvidia-smi --query-gpu="$QUERY" --format=csv,noheader,nounits 2>/dev/null)"; then
    echo "error: nvidia-smi failed" >&2
    exit 1
fi

if (( LIST_ONLY )); then
    printf '%-5s %12s %12s  %s\n' idx free_MiB total_MiB name
    while IFS=, read -r idx free total name; do
        printf '%-5s %12s %12s  %s\n' "${idx// /}" "${free// /}" "${total// /}" "${name# }"
    done <<< "$rows"
    exit 0
fi

best_idx=""
best_free=-1
while IFS=, read -r idx free _total _name; do
    idx="${idx// /}"
    free="${free// /}"
    [[ -z "$idx" ]] && continue
    if (( free > best_free )); then
        best_free="$free"
        best_idx="$idx"
    fi
done <<< "$rows"

if [[ -z "$best_idx" ]]; then
    echo "error: could not parse any GPU from nvidia-smi" >&2
    exit 1
fi

if (( best_free < MIN_FREE )); then
    echo "error: the freest GPU (index $best_idx) has only ${best_free} MiB free, below the ${MIN_FREE} MiB floor." >&2
    echo "       Wait for capacity, or lower --min-free if you know the job fits." >&2
    exit 1
fi

echo "selected GPU $best_idx with ${best_free} MiB free" >&2
printf '%s' "$best_idx"
