#!/usr/bin/env bash
# Render the prepared PI trajectory in the disk-backed Isaac environment.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
actual_root=$(bash "$project_dir/disk_docker.sh" docker info --format '{{.DockerRootDir}}')
[[ "$actual_root" == /mnt/bee-nav-docker/docker ]] || { printf 'Unexpected Docker storage.\n' >&2; exit 1; }
exec bash "$project_dir/disk_docker.sh" docker run --rm \
    --name bee-nav-pi-bird-record --network none --gpus all --shm-size=1g \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=2 \
    -v /mnt/bee-nav-docker/cache/kit:/isaac-sim/kit/cache \
    -v /mnt/bee-nav-docker/cache/ov:/root/.cache/ov \
    -v "$project_dir/CodeInsectInspired:/isaac-sim/CodeInsectInspired:ro" \
    -v "$project_dir/CodeInsectInspired/runs:/isaac-sim/CodeInsectInspired/runs" \
    -w /isaac-sim/CodeInsectInspired --entrypoint /isaac-sim/python.sh \
    insect-sim:submission tools/record_pi_bird.py --output runs/pi_bird_video_01 "$@"
