#!/usr/bin/env bash
# Same147 sampling positions compressed into20m; reuse the established training/test pipeline.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
code_dir="$project_dir/CodeInsectInspired"
storage_root=$(bash "$project_dir/disk_docker.sh" docker info --format '{{.DockerRootDir}}')
[[ "$storage_root" == /mnt/bee-nav-docker/docker ]] || { printf 'Unexpected Docker storage.\n' >&2; exit 1; }
mode_args=(-d)
gpu_args=(--gpus all)
for argument in "$@"; do
    if [[ "$argument" == --plan-only ]]; then mode_args=(); gpu_args=(); fi
done
if [[ ${#gpu_args[@]} -gt 0 ]]; then
    running=$(bash "$project_dir/disk_docker.sh" docker ps --format '{{.Names}}')
    [[ -z "$running" ]] || { printf 'Another project container is active: %s\n' "$running" >&2; exit 1; }
fi
exec bash "$project_dir/disk_docker.sh" docker run "${mode_args[@]}" --rm \
    --name bee-nav-stage23-compact20 --network none "${gpu_args[@]}" --shm-size=4g \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=2 \
    -e MPLCONFIGDIR=/tmp/stage23-mpl \
    -v "$code_dir:/isaac-sim/CodeInsectInspired:ro" \
    -v "$code_dir/runs:/isaac-sim/CodeInsectInspired/runs:rw" \
    -v /mnt/bee-nav-docker/cache/kit:/isaac-sim/kit/cache \
    -v /mnt/bee-nav-docker/cache/ov:/root/.cache/ov \
    -w /isaac-sim/CodeInsectInspired --entrypoint /isaac-sim/python.sh \
    insect-sim:submission run_compact_spiral_experiment.py "$@"
