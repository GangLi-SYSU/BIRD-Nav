#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ -n ${DISPLAY:-} ]] || { printf 'Run from a desktop terminal with DISPLAY set.\n' >&2; exit 1; }
actual_root=$(bash "$project_dir/disk_docker.sh" docker info --format '{{.DockerRootDir}}')
[[ "$actual_root" == /mnt/bee-nav-docker/docker ]] || { printf 'Unexpected Docker storage.\n' >&2; exit 1; }
xauthority_file="${XAUTHORITY:-$HOME/.Xauthority}"
[[ -f "$xauthority_file" ]] || { printf 'X11 authority file is missing: %s\n' "$xauthority_file" >&2; exit 1; }
xhost +si:localuser:root
trap 'xhost -si:localuser:root >/dev/null 2>&1 || true' EXIT
bash "$project_dir/disk_docker.sh" docker run \
    --name isaac-sim-experiment --entrypoint bash -it --gpus all --rm --network host \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e "DISPLAY=$DISPLAY" --shm-size=8g \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$xauthority_file:/root/.Xauthority:ro" \
    -v /mnt/bee-nav-docker/cache/kit:/isaac-sim/kit/cache \
    -v /mnt/bee-nav-docker/cache/ov:/root/.cache/ov \
    -v /mnt/bee-nav-docker/cache/pip:/root/.cache/pip \
    -v "$project_dir/CodeInsectInspired:/isaac-sim/CodeInsectInspired" \
    insect-sim:submission
