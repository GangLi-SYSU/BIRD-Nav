#!/usr/bin/env bash
# Keep new circular-control artifacts on the project disk; no Isaac imports or network.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
code_dir="$project_dir/CodeInsectInspired"
phase="${1:-}"
case "$phase" in
    prepare|preview|select|evaluate) gpu_args=(); mode_args=() ;;
    smoke) gpu_args=(--gpus all); mode_args=() ;;
    train) gpu_args=(--gpus all); mode_args=(-d) ;;
    *) printf 'Usage: bash run_circular_on_disk.sh {prepare|preview|smoke|train|select|evaluate} [arguments]\n' >&2; exit 1 ;;
esac
storage_root=$(bash "$project_dir/disk_docker.sh" docker info --format '{{.DockerRootDir}}')
[[ "$storage_root" == /mnt/bee-nav-docker/docker ]] || {
    printf 'Unexpected Docker storage root.\n' >&2; exit 1
}
if [[ ${#gpu_args[@]} -gt 0 ]]; then
    running=$(bash "$project_dir/disk_docker.sh" docker ps --format '{{.Names}}')
    if [[ "$running" =~ bee-nav-stage(09-view|11-coverage|12-navigation) ]]; then
        printf 'Close the existing Isaac/training container before GPU training.\n' >&2
        exit 1
    fi
fi
exec bash "$project_dir/disk_docker.sh" docker run "${mode_args[@]}" --rm \
    --name bee-nav-stage13-circular --network none "${gpu_args[@]}" --shm-size=2g \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=2 \
    -e MPLCONFIGDIR=/tmp/stage13-mpl \
    -v "$code_dir:/isaac-sim/CodeInsectInspired:ro" \
    -v "$code_dir/runs:/isaac-sim/CodeInsectInspired/runs:rw" \
    -w /isaac-sim/CodeInsectInspired --entrypoint /isaac-sim/python.sh \
    insect-sim:submission run_circular_experiment.py "$@"
