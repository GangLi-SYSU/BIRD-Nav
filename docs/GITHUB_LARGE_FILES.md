# GitHub 发布前大文件盘点

目标：`GangLi-SYSU/BIRD-Nav`，公有仓库；用途为重新训练与复现实验。仓库尚未创建。
扫描日期：2026-11-16。本次仅盘点，没有上传、迁移或删除任何文件。

扫描到 113,491 个文件；单文件 ≥50 MiB 共 18 个，其中 5 个超过 100 MiB。
扫描排除了 `.git/`、`.agents/`、`.codex/`、`.aws/`、Python 缓存和符号链接；本盘点不是凭据审计。
目录体积采用 `du -h` 的近似磁盘占用；下面单文件清单采用精确逻辑长度。稀疏文件的两者可能不同。

## 大型目录与文件用途

以下路径均相对 workspace；目录之间存在包含关系，不能把所有行直接求和。

| 路径 | 约占用 | 内容与复现用途 |
|---|---:|---|
| `Bee-Nav/visual_simulator_docker/downloads/docker-disk/docker.ext4` | 40 GiB 逻辑大小 | 已安装 Docker 环境的稀疏磁盘，实际占用见下文；不作为源码发布 |
| `Bee-Nav/visual_simulator_docker/downloads/isaac-sim-4.2.0.docker.tar` | 8.54 GiB | Isaac 基础镜像归档，可通过环境安装流程重新获取 |
| `Bee-Nav/visual_simulator_docker/CodeInsectInspired/runs/` | 65 GiB | 历史采集图像、训练、导航取图与结果，绝大部分单文件很小 |
| `Bee-Nav/visual_simulator_docker/isaac-sim_assets/` | 2.4 GiB | 原项目仿真素材，其中 Trees 约 2.2 GiB |
| `Bee-Nav/visual_simulator_docker/CodeInsectInspired/maps/` | 604 MiB | 所有已生成地图及其依赖 |
| `Bee-Nav/visual_simulator_docker/CodeInsectInspired/maps/downward_real_h40_v1/` | 375 MiB | 当前 demo 使用的房屋、树木、水池场景与纹理 |
| `Bee-Nav/path_integration_noise_simulation/magnetic_guidance/runs/` | 376 MiB | 路径积分批量实验的轨迹与统计输出，可用代码重新生成 |
| `Bee-Nav/visual_simulator_docker/CodeInsectInspired/runs/interactive_demo/` | 662 MiB | 所有交互演示的日志、相机图像及结果 |

## 单文件 ≥50 MiB 的完整清单

GitHub 普通 Git 会拒绝超过 100 MiB 的单文件。参考：[GitHub 文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。
即使某个单文件没有超过限制，也需要考虑同目录中大量小文件的累计体积。

| 文件（相对 workspace） | 文件长度 | 类型 | 超过 100 MiB |
|---|---:|---|---|
| `Bee-Nav/visual_simulator_docker/downloads/docker-disk/docker.ext4` | 40.00 GiB | Docker 工作磁盘 | 是 |
| `Bee-Nav/visual_simulator_docker/downloads/isaac-sim-4.2.0.docker.tar` | 8.54 GiB | Isaac 镜像归档 | 是 |
| `Bee-Nav/visual_simulator_docker/CodeInsectInspired/maps/downward_landmarks_v1/ground.png` | 101.19 MiB | 历史几何路标场景地面纹理 | 是 |
| `Bee-Nav/visual_simulator_docker/CodeInsectInspired/maps/downward_landmarks_v2/ground.png` | 101.19 MiB | 历史几何路标场景地面纹理 | 是 |
| `Bee-Nav/visual_simulator_docker/isaac-sim_assets/Trees/materials/textures/bark1_normal.png` | 100.72 MiB | 树木模型、纹理或环境光素材 | 是 |
| `Bee-Nav/visual_simulator_docker/isaac-sim_assets/qwantani.hdr` | 94.99 MiB | 树木模型、纹理或环境光素材 | 否 |
| `Bee-Nav/visual_simulator_docker/isaac-sim_assets/Trees/materials/textures/bark1_basecolor.png` | 85.85 MiB | 树木模型、纹理或环境光素材 | 否 |
| `Bee-Nav/visual_simulator_docker/isaac-sim_assets/Trees/materials/textures/pinebark1_basecolor.png` | 85.71 MiB | 树木模型、纹理或环境光素材 | 否 |
| `Bee-Nav/visual_simulator_docker/isaac-sim_assets/Trees/materials/textures/sycamorebark2_basecolor.png` | 84.00 MiB | 树木模型、纹理或环境光素材 | 否 |
| `Bee-Nav/visual_simulator_docker/CodeInsectInspired/maps/downward_real_h40_v1/assets/nvidia/Trees/Japanese_Maple.usd` | 77.99 MiB | 树木模型、纹理或环境光素材 | 否 |
| `Bee-Nav/visual_simulator_docker/CodeInsectInspired/runs/stage09_real_preview/initial_bundle_missing_mdl_textures/assets/nvidia/Trees/Japanese_Maple.usd` | 77.99 MiB | 历史实验中的素材副本 | 否 |
| `Bee-Nav/visual_simulator_docker/isaac-sim_assets/Trees/Japanese_Maple.usd` | 77.99 MiB | 树木模型、纹理或环境光素材 | 否 |
| `Bee-Nav/visual_simulator_docker/isaac-sim_assets/Trees/Hawthorn.usd` | 77.82 MiB | 树木模型、纹理或环境光素材 | 否 |
| `Bee-Nav/visual_simulator_docker/isaac-sim_assets/Trees/Orange_Tree.usd` | 67.77 MiB | 树木模型、纹理或环境光素材 | 否 |
| `Bee-Nav/visual_simulator_docker/isaac-sim_assets/Trees/Japanese_Maple_Fall.usd` | 59.69 MiB | 树木模型、纹理或环境光素材 | 否 |
| `Bee-Nav/visual_simulator_docker/CodeInsectInspired/runs/interactive_demo/20261110_095601_903728/events.jsonl` | 56.63 MiB | 交互演示日志 | 否 |
| `Bee-Nav/visual_simulator_docker/isaac-sim_assets/Trees/Lombardy_Poplar.usd` | 52.91 MiB | 树木模型、纹理或环境光素材 | 否 |
| `Bee-Nav/visual_simulator_docker/isaac-sim_assets/Trees/Lombardy_Poplar_Fall.usd` | 52.91 MiB | 树木模型、纹理或环境光素材 | 否 |

`docker.ext4` 的逻辑长度为 40.00 GiB，当前实际分配约 32.16 GiB。

## 占用最大的历史实验目录

以下都位于 `Bee-Nav/visual_simulator_docker/CodeInsectInspired/runs/`：

| 目录 | 约占用 |
|---|---:|
| `stage12_navigation_01/` | 19 GiB |
| `stage14_rotation_r20_01/` | 7.5 GiB |
| `stage20_near_finetune_01/` | 6.0 GiB |
| `stage11_coverage_02/` | 5.7 GiB |
| `stage17_radius_5_10_01/` | 5.4 GiB |
| `stage23_compact_spiral_h40_fov30_01/` | 3.4 GiB |
| `stage15_rotation_300_01/` | 3.2 GiB |
| `stage16_rotation_600_01/` | 3.0 GiB |
| `stage25_fixed50_15m_01/` | 2.5 GiB |
| `stage19_near_data_01/` | 2.1 GiB |

## 当前训练与 demo 的相关资源

当前使用 `stage23_compact_spiral_h40_fov30_01` 中按验证集选择的 epoch 484 模型。其约 3.4 GiB 总占用包括：

- `capture/`：约 1018 MiB，当前螺旋采集数据。
- `training/`：约 128 MiB，包括训练状态、日志与导出文件。
- `navigation/`：约 2.3 GiB，主要是导航测试相机图像、日志和结果，不是训练输入。

关键模型文件本身很小，适合与代码一起提供：

| 相对 Stage23 目录的文件 | 大小 |
|---|---:|
| `training/best_validation.pth` | 51.4 KiB |
| `training/final.pth` | 51.4 KiB |
| `training/deployment/home_vector_m.torchscript.pt` | 667.4 KiB |
| `training/deployment/home_vector_m.onnx` | 630.8 KiB |

## 本次发布采用的资源处理方式

按用户选择，必要的大资源在 [复现资源指南](REPRODUCTION_RESOURCES.md) 中记录原始下载来源、
目录、校验或重新生成方法；不把本机 Docker 磁盘、镜像归档、缓存、历史导航截图与训练输出上传到 Git。
这些忽略规则不会删除任何本地数据。

参考 epoch 484 的权重、TorchScript 和 ONNX 共约 1.32 MiB，已独立复制至
`CodeInsectInspired/assets/models/stage23_epoch484/`，并记录 SHA256。
demo 改读 `config/experiments/demo_release.json`，消除了对本机历史 `runs/` 的启动依赖。
相机、网络与控制器设置保持一致。

当前自采图像与预生成场景没有公开资源包下载地址；指南提供从原始素材重新建场景、
重新采集和训练的方法。若要重用完全相同的历史图像，后续应另行发布带清单和哈希的数据包。
第三方场景资产保留来源与许可；小型鸟模型的运行依赖随代码提供。

同目录的 `GITHUB_LARGE_FILES.csv` 提供所有大文件的精确字节数、磁盘分配量和路径，便于后续筛选。
