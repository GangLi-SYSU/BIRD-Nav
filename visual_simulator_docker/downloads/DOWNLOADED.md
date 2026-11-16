# Isaac Sim 4.2.0 下载记录

已根据上级目录的 `README.md` 下载并校验以下内容，全部保存在 `/media/lg/lg-512` 硬盘中。

## 下载文件

- Isaac Sim 基础镜像：`isaac-sim-4.2.0.docker.tar`，8.54 GiB。
- 镜像标签：`nvcr.io/nvidia/isaac-sim:4.2.0`；平台：`linux/amd64`。
- 场景资源：`../isaac-sim_assets/`，2.34 GiB，共 966 个文件，包含 44 个树模型 USD 文件。
- 下载与校验记录：`download-complete.json`、`image-download.json`、`assets-download.json`、`assets-sha256.json`。

## 校验结果

镜像内容的 SHA256 与 NVIDIA 镜像清单匹配。Docker 归档结构、镜像标签和平台检查通过，包含 37 个镜像层。全部资源文件尺寸与公开资源目录一致；44 个 USD 模型和 HDR 环境贴图的格式检查通过。

镜像清单摘要：`sha256:b606646df3aab3f38ba655ce96cdf8f45a13bb167f01c4b2992e40d6f2900089`。

归档 SHA256：`cefd095f1e73a680d04228c3c16c0758d473a79ecfcc8e312fcedfddc5ae7d28`。

## 已完成构建与验证

已于 2026-10-14 完成镜像导入、项目构建及 GPU 验证。镜像标签为 `insect-sim:submission`，ID 为 `sha256:110deb27e96333b194ee0f680b9d3253e35a06e80075e16905c1983359202980`，大小约 19.30 GiB。

项目 Docker 数据目录为 `/mnt/bee-nav-docker/docker`，实际存储文件为本目录下的 `docker-disk/docker.ext4`。40 GiB ext4 存储目前使用约 22 GiB。系统 Docker 配置仍使用 `/var/lib/docker`。启动和管理方式见 [DISK_BUILD.md](../DISK_BUILD.md)。

在 `visual_simulator_docker/` 中执行：

```bash
sudo bash disk_docker.sh start
bash build_on_disk.sh
```

构建后使用 `bash run_sim_on_disk.sh`，将运行缓存和实验输出保存在本硬盘。

已添加 `.dockerignore`，将下载归档和辅助脚本排除在 Docker 构建上下文之外，场景资源仍会进入构建。

## 容器验证记录

- Python 3.10.14；NumPy 1.26.0；SciPy 1.10.1；scikit-image 0.24.0。
- PyTorch 2.4.0+cu118 和 torchvision 0.19.0+cu118 导入成功。
- NumPy/PyTorch 数组互转及 RTX 4060 上的实际 CUDA 运算通过。
- 实验入口、基础场景、HDR 贴图及 44 个树模型存在检查通过。
- 已修复原 Dockerfile 安装新版 scikit-image 引入的 SciPy/osqp 版本冲突。
- `pip check` 仍报告 12 条基础镜像已有的缺失依赖或版本警告；当前项目依赖和 GPU 验证通过，未执行完整导航实验。

最终记录见 `docker-disk/build-complete.json`，验证输出见 `docker-disk/verification.log`，依赖检查见 `docker-disk/pip-check.txt`。完整构建日志与兼容性修复日志分别为 `docker-disk/build.log` 和 `docker-disk/repair-build.log`。
