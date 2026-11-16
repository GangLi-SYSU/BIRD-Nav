#!/usr/bin/env bash
# Run the frozen Stage11 experiment with all persistent output on the project disk.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
code_dir="$project_dir/CodeInsectInspired"
storage_root=$(bash "$project_dir/disk_docker.sh" docker info --format '{{.DockerRootDir}}')
[[ "$storage_root" == /mnt/bee-nav-docker/docker ]] || {
    printf 'Unexpected Docker storage root: %s\n' "$storage_root" >&2
    exit 1
}
exec bash "$project_dir/disk_docker.sh" docker run -d --rm \
    --name bee-nav-stage11-coverage --network none --gpus all --shm-size=8g \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=2 \
    -e MPLCONFIGDIR=/tmp/stage11-mpl \
    -v "$code_dir:/isaac-sim/CodeInsectInspired:ro" \
    -v "$code_dir/runs:/isaac-sim/CodeInsectInspired/runs:rw" \
    -v /mnt/bee-nav-docker/cache/kit:/isaac-sim/kit/cache \
    -v /mnt/bee-nav-docker/cache/ov:/root/.cache/ov \
    -w /isaac-sim/CodeInsectInspired --entrypoint /isaac-sim/python.sh \
    insect-sim:submission run_coverage_experiment.py "$@"
