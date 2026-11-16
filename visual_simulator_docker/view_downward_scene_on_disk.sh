#!/usr/bin/env bash
# Browse stage C terrain using the existing disk-backed Docker service.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ -n ${DISPLAY:-} ]] || { printf 'Run from a desktop terminal with DISPLAY set.\n' >&2; exit 1; }
xauthority_file="${XAUTHORITY:-$HOME/.Xauthority}"
[[ -f "$xauthority_file" ]] || { printf 'X11 authority file is missing.\n' >&2; exit 1; }
actual_root=$(bash "$project_dir/disk_docker.sh" docker info --format '{{.DockerRootDir}}')
[[ "$actual_root" == /mnt/bee-nav-docker/docker ]] || { printf 'Unexpected Docker storage.\n' >&2; exit 1; }

exec bash "$project_dir/disk_docker.sh" docker run --rm \
    --name bee-nav-stage03-view --network none --gpus all --shm-size=1g \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e "DISPLAY=$DISPLAY" \
    -e XAUTHORITY=/root/.Xauthority -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$xauthority_file:/root/.Xauthority:ro" \
    -v /mnt/bee-nav-docker/cache/kit:/isaac-sim/kit/cache \
    -v /mnt/bee-nav-docker/cache/ov:/root/.cache/ov \
    -v "$project_dir/CodeInsectInspired:/isaac-sim/CodeInsectInspired:ro" \
    -v "$project_dir/CodeInsectInspired/runs:/isaac-sim/CodeInsectInspired/runs" \
    -w /isaac-sim/CodeInsectInspired --entrypoint /isaac-sim/python.sh \
    insect-sim:submission tools/view_stage03_scene.py "$@"
