# Isaac Sim 两阶段归巢仿真

本目录提供 BIRD-Nav 的容器构建、真实建筑场景、下视视觉训练、导航测试和交互演示。
仿真固定使用 Isaac Sim 4.2.0；最终方案与参考结果见[仓库首页](../README.md)。

## 环境与场景准备

需要 Linux、NVIDIA RTX GPU、兼容驱动、Docker 和 NVIDIA Container Toolkit。
首次使用请依次完成[复现指南](../docs/REPRODUCTION_RESOURCES.md)中的素材下载、
容器构建、四种房屋校验、场景生成与依赖检查。

素材位于 `isaac-sim_assets/` 和 `downloads/real_preview_v1/`；生成场景位于
`CodeInsectInspired/maps/downward_real_h40_v1/real_scene.usda`。
所有纹理、材质及许可文件应保持完整的相对路径。

下载原项目素材后，在本目录运行：

```bash
sudo bash disk_docker.sh start
bash disk_docker.sh docker pull nvcr.io/nvidia/isaac-sim:4.2.0
BEE_NAV_PIP_INDEX_URL=https://pypi.org/simple bash build_on_disk.sh
```

镜像名为 `insect-sim:submission`。专用 Docker 存储写入本目录的
`downloads/docker-disk/docker.ext4`，管理方式见[磁盘环境说明](DISK_BUILD.md)。

## 采集、训练与测试

完成真实场景生成后，在本目录运行：

```bash
bash run_compact_spiral_on_disk.sh --run-dir runs/reproduce_compact20_01 --plan-only
bash run_compact_spiral_on_disk.sh --run-dir runs/reproduce_compact20_01
bash disk_docker.sh docker logs -f bee-nav-stage23-compact20
```

流程采集 147 个螺旋位置、八个航向的 1,176 张图像，训练 600 轮，按验证损失选模型，
再执行 10 m、20 m 各 64 次归巢。相机高度 40 m、视场角 30°，网络输入为 384×384；
标准到达半径为 2 m，最多执行 50 步。

输出位于 `CodeInsectInspired/runs/reproduce_compact20_01/`。
每次独立实验使用新的目录；运行训练前关闭占用 GPU 的交互演示。

## 启动交互演示

在桌面终端、本目录运行：

```bash
bash run_interactive_homing_demo.sh
```

`T` 起飞，`W/A/S/D` 手动外出，`Q/E` 转向，`H` 自动两阶段归巢，空格暂停，`R` 重置。
主视图为第三人称，同时显示俯视总览和下视 RGB；归巢完成后降落并显示轨迹和起降距离。
使用随代码提供的第 484 轮参考网络，无需先训练。
详细操作见[演示说明](../docs/INTERACTIVE_TWO_STAGE_DEMO.md)。

## 查看状态与输出

```bash
bash disk_docker.sh status
bash disk_docker.sh docker ps
bash disk_docker.sh docker logs -f bee-nav-interactive-demo
```

训练曲线、导航轨迹、图像和日志均写入 `CodeInsectInspired/runs/`。
容器缓存、下载包、生成地图与输出不纳入 Git；获取和生成方法以复现指南为准。
