#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
actual_root=$(bash "$project_dir/disk_docker.sh" docker info --format '{{.DockerRootDir}}')
[[ "$actual_root" == /mnt/bee-nav-docker/docker ]] || { printf 'Unexpected Docker storage.\n' >&2; exit 1; }
run_id="$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$project_dir/CodeInsectInspired/runs/interactive_demo"
options=()
if [[ " ${*} " != *" --headless-smoke "* ]]; then
    [[ -n ${DISPLAY:-} ]] || { printf 'Run from your desktop terminal.\n' >&2; exit 1; }
    demo_xauthority="${XAUTHORITY:-$HOME/.Xauthority}"
    [[ -f "$demo_xauthority" ]] || { printf 'X11 authority file missing.\n' >&2; exit 1; }
    options+=(-d -e "DISPLAY=$DISPLAY" -e XAUTHORITY=/root/.Xauthority
              -v /tmp/.X11-unix:/tmp/.X11-unix -v "$demo_xauthority:/root/.Xauthority:ro")
fi
printf 'Session output: %s/CodeInsectInspired/runs/interactive_demo/%s\n' "$project_dir" "$run_id"
exec bash "$project_dir/disk_docker.sh" docker run --rm "${options[@]}" \
    --name bee-nav-interactive-demo --network none --gpus all --shm-size=1g \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=2 \
    -v /mnt/bee-nav-docker/cache/kit:/isaac-sim/kit/cache \
    -v /mnt/bee-nav-docker/cache/ov:/root/.cache/ov \
    -v "$project_dir/CodeInsectInspired:/isaac-sim/CodeInsectInspired:ro" \
    -v "$project_dir/CodeInsectInspired/runs:/isaac-sim/CodeInsectInspired/runs" \
    -w /isaac-sim/CodeInsectInspired --entrypoint /isaac-sim/python.sh \
    insect-sim:submission tools/interactive_homing_multiview.py \
    --output "runs/interactive_demo/$run_id" "$@"
