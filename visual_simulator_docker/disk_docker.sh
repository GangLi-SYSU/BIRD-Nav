#!/usr/bin/env bash
# Manage a separate Docker daemon backed by an ext4 image on the project disk.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
storage_dir="$project_dir/downloads/docker-disk"
disk_image="$storage_dir/docker.ext4"
mount_dir=/mnt/bee-nav-docker
runtime_dir=/run/bee-nav-docker
socket="unix://$runtime_dir/docker.sock"

die() { printf '%s\n' "$*" >&2; exit 1; }
require_root() { [[ $EUID -eq 0 ]] || die "Run this operation with sudo."; }
check_mount() {
    mountpoint -q "$mount_dir" || die "Docker disk is not mounted; run sudo bash disk_docker.sh start."
    [[ $(findmnt -n -o FSTYPE --target "$mount_dir") == ext4 ]] || die "Unexpected filesystem at $mount_dir."
    local device backing
    device=$(findmnt -n -o SOURCE --target "$mount_dir")
    backing=$(losetup -n -O BACK-FILE "$device")
    [[ $(readlink -f -- "$backing") == "$disk_image" ]] || die "Mount belongs to a different disk image."
}
docker_client() {
    env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
        docker --config "$storage_dir/client" --host "$socket" "$@"
}

case "${1:-status}" in
    start)
        require_root
        mkdir -p "$storage_dir/client" "$mount_dir" "$runtime_dir"
        if [[ ! -e "$disk_image" ]]; then
            available=$(df -B1 --output=avail "$storage_dir" | tail -1 | tr -d ' ')
            (( available >= 42 * 1024 * 1024 * 1024 )) || die "At least 42 GiB free is required for initial setup."
            # Only format a newly created file, never an existing image or disk.
            (set -o noclobber; : > "$disk_image")
            truncate -s 40G "$disk_image"
            /usr/sbin/mkfs.ext4 -F -m 0 -L bee-nav-docker "$disk_image"
        fi
        if ! mountpoint -q "$mount_dir"; then
            [[ $(blkid -p -s TYPE -o value "$disk_image") == ext4 ]] || die "Existing image is not ext4; refusing to format it."
            mount -o loop,noatime "$disk_image" "$mount_dir"
        fi
        check_mount
        if docker_client info >/dev/null 2>&1; then
            [[ $(docker_client info --format '{{.DockerRootDir}}') == "$mount_dir/docker" ]] || die "Unexpected daemon on project socket."
            printf 'Project Docker daemon is already running.\n'
            exit 0
        fi
        if [[ -f "$runtime_dir/docker.pid" ]] && kill -0 "$(cat "$runtime_dir/docker.pid")" 2>/dev/null; then
            die "A daemon is starting or unhealthy; inspect $storage_dir/dockerd.log."
        fi
        mkdir -p "$mount_dir/docker" "$mount_dir/tmp" \
            "$mount_dir/cache/kit" "$mount_dir/cache/ov" "$mount_dir/cache/pip"
        cat > "$storage_dir/daemon.json" <<EOF
{
  "data-root": "$mount_dir/docker",
  "exec-root": "$runtime_dir/exec",
  "pidfile": "$runtime_dir/docker.pid",
  "hosts": ["$socket"],
  "storage-driver": "overlay2",
  "bridge": "none",
  "iptables": false,
  "ip6tables": false,
  "ip-forward": false,
  "ip-masq": false,
  "containerd-namespace": "bee-nav",
  "containerd-plugins-namespace": "bee-nav-plugins",
  "runtimes": {"nvidia": {"path": "/usr/bin/nvidia-container-runtime", "args": []}},
  "log-opts": {"max-size": "10m", "max-file": "3"}
}
EOF
        dockerd --validate --config-file "$storage_dir/daemon.json"
        nohup env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
            TMPDIR="$mount_dir/tmp" DOCKER_TMPDIR="$mount_dir/tmp" \
            dockerd --config-file "$storage_dir/daemon.json" \
            >> "$storage_dir/dockerd.log" 2>&1 < /dev/null &
        for attempt in {1..60}; do
            if docker_client info >/dev/null 2>&1; then
                printf 'Docker ready: %s\nData: %s\n' "$socket" "$disk_image"
                exit 0
            fi
            sleep 1
        done
        tail -40 "$storage_dir/dockerd.log" >&2
        die "Daemon did not become ready."
        ;;
    stop)
        require_root
        check_mount
        if [[ -f "$runtime_dir/docker.pid" ]]; then
            [[ $(docker_client info --format '{{.DockerRootDir}}') == "$mount_dir/docker" ]] || die "Cannot verify daemon identity."
            [[ -z $(docker_client ps -q) ]] || die "Stop project containers before stopping the daemon."
            daemon_pid=$(cat "$runtime_dir/docker.pid")
            kill -TERM "$daemon_pid"
            for attempt in {1..60}; do
                kill -0 "$daemon_pid" 2>/dev/null || break
                sleep 1
            done
            kill -0 "$daemon_pid" 2>/dev/null && die "Daemon is still shutting down; disk remains mounted."
        fi
        sync
        umount "$mount_dir"
        ;;
    docker)
        shift
        docker_client "$@"
        ;;
    status)
        docker_client info --format 'Storage={{.DockerRootDir}} Driver={{.Driver}}'
        df -h "$mount_dir" "$storage_dir"
        ;;
    *) die "Usage: bash disk_docker.sh {start|stop|status|docker <arguments>}" ;;
esac
