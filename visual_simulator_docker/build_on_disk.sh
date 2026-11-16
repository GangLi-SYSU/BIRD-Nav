#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
disk_docker=(bash "$project_dir/disk_docker.sh" docker)
expected_root=/mnt/bee-nav-docker/docker
actual_root=$("${disk_docker[@]}" info --format '{{.DockerRootDir}}')
[[ "$actual_root" == "$expected_root" ]] || { printf 'Unexpected Docker data root: %s\n' "$actual_root" >&2; exit 1; }
mkdir -p downloads/docker-disk/client
if ! "${disk_docker[@]}" image inspect nvcr.io/nvidia/isaac-sim:4.2.0 >/dev/null 2>&1; then
    "${disk_docker[@]}" load -i downloads/isaac-sim-4.2.0.docker.tar
fi
DOCKER_BUILDKIT=1 "${disk_docker[@]}" build --network host --progress plain \
    --build-arg "PIP_INDEX_URL=${BEE_NAV_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple/}" \
    --build-arg HTTP_PROXY= --build-arg HTTPS_PROXY= --build-arg ALL_PROXY= \
    --build-arg http_proxy= --build-arg https_proxy= --build-arg all_proxy= \
    -t insect-sim:submission . 2>&1 | tee downloads/docker-disk/build.log
"${disk_docker[@]}" image inspect insect-sim:submission \
    --format 'Built {{.RepoTags}} {{.Id}} ({{.Size}} bytes)'
