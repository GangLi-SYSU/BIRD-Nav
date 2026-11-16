# 在 /media/lg/lg-512 上构建容器

本硬盘为 NTFS，不能直接作为 Docker `overlay2` 的存储目录。`disk_docker.sh` 在 `downloads/docker-disk/docker.ext4` 中创建 40 GiB 的稀疏 ext4 文件，挂载到 `/mnt/bee-nav-docker`，并启动项目专用 Docker 服务。文件按实际写入量占用硬盘空间。镜像、构建缓存、容器数据及导入临时文件均位于该文件中。

系统 Docker 服务继续使用原配置；项目服务使用独立套接字 `/run/bee-nav-docker/docker.sock`，不创建网桥或修改防火墙。`/run` 仅保存少量运行状态。创建文件系统、挂载及启动服务需要管理员权限。

## 启动与构建

在本目录执行：

```bash
sudo bash disk_docker.sh start
bash build_on_disk.sh
```

第一条命令首次创建存储，之后用于重启后的重新挂载和启动。不会格式化物理硬盘，也不会重新格式化已有镜像文件。构建脚本导入已下载的 Isaac Sim 4.2.0 归档，再按项目 Dockerfile 构建 `insect-sim:submission`。日志保存在 `downloads/docker-disk/build.log`。

为保持 Isaac Sim 内置 `osqp` 的兼容性，Dockerfile 固定 `scipy==1.10.1` 和 `scikit-image==0.24.0`。硬盘构建脚本默认使用清华 PyPI 源，可通过 `BEE_NAV_PIP_INDEX_URL=https://pypi.org/simple bash build_on_disk.sh` 切换回官方源；Dockerfile 单独构建时默认使用官方源。

## 检查与管理

```bash
bash disk_docker.sh status
bash disk_docker.sh docker images
bash disk_docker.sh docker ps -a
sudo bash disk_docker.sh stop
```

停止命令要求先停止项目容器，然后关闭专用服务并卸载存储文件。卸载后才可拔出硬盘。脚本不配置开机自启。普通 `docker` 命令连接系统服务；操作项目镜像时请使用上述包装命令。

## 启动仿真

构建完成后，在桌面终端执行：

```bash
bash run_sim_on_disk.sh
# 在容器中执行：
cd CodeInsectInspired
../python.sh run_ANN_homing_simulations.py
```

该脚本沿用 README 的 GPU、共享内存及 Isaac Sim 许可环境变量，使用专用 Docker 服务，并将缓存映射到 ext4 文件中的 `/mnt/bee-nav-docker/cache/`。源码与实验输出目录映射到硬盘上的 `CodeInsectInspired/`。原 `run_sim.sh` 使用系统 Docker 和 `$HOME` 缓存，请改用本脚本。脚本沿用 NVIDIA 的 `ACCEPT_EULA=Y` 和 `PRIVACY_CONSENT=Y` 设置，启动前请确认接受相应许可和隐私条款。

## 已完成的验证

2026-10-14 构建完成的镜像为 `insect-sim:submission`（ID 前缀 `110deb27e963`）。依赖导入、NumPy/PyTorch 互转、RTX 4060 上的 CUDA 运算以及场景文件检查通过；完整导航实验尚未执行。`pip check` 仍有基础镜像原有的 12 条告警，新增的 SciPy/osqp 冲突已消除。记录保存在 `downloads/docker-disk/build-complete.json`。

可重新运行轻量验证：

```bash
bash disk_docker.sh docker run --rm --network none --gpus all \
  --entrypoint /isaac-sim/python.sh \
  -v "$PWD/downloads/verify_container.py:/tmp/verify_container.py:ro" \
  insect-sim:submission /tmp/verify_container.py
```
