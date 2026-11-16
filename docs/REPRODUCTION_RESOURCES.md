# 复现资源与操作指南

本指南对应 BIRD-Nav 当前的下视视觉归巢与两阶段 demo。所有相对路径以仓库根目录为准，
不是父级 workspace。原 Bee-Nav 的全景视觉、机载机器人实验仍按各子目录 README 操作。

## 1. 哪些文件随代码提供

源码、协议、测试、场景生成器、资源来源/哈希清单，以及运行所需的小型鸟模型随 Git 提供。
参考网络位于：

```text
visual_simulator_docker/CodeInsectInspired/assets/models/stage23_epoch484/
  best_validation.pth
  home_vector_m.torchscript.pt
  home_vector_m.onnx
  SHA256SUMS
```

三个模型文件合计约 1.32 MiB，复制自 Stage23 的验证集最佳 epoch 484。
`config/experiments/demo_release.json` 保存对应相机/控制器设置及模型哈希；
demo 不再读取历史训练目录。可在模型目录运行 `sha256sum -c SHA256SUMS` 验证。
鸟模型的运行 USD、绑定/动画依赖和许可保留在 `assets/birds/quaternius_eagle/`；
原始 FBX、Blender 工程、下载压缩包和转换诊断不需要重复上传。

## 2. 必需的大资源及来源

| 资源 | 本机约体积 | 用途与获取方法 |
| --- | ---: | --- |
| Isaac Sim 4.2.0 基础镜像 | 归档约 8.54 GiB | NVIDIA NGC：`nvcr.io/nvidia/isaac-sim:4.2.0`，按下节下载构建 |
| 原项目素材 `visual_simulator_docker/isaac-sim_assets/` | 2.4 GiB | [上游 SURFdrive](https://surfdrive.surf.nl/s/qardanQkDLwe7m2)，下载并保留完整目录层级 |
| 四种 Grab3D 房屋及纹理 | 数 MiB | 下节列出 USDZ、许可与清单的下载方法 |
| 生成的 `maps/downward_real_h40_v1/` | 375 MiB | 从上述素材运行场景生成器；目前没有预生成场景包的公开下载地址 |
| Stage23 `capture/` | 约 1018 MiB | 147 个螺旋位置 × 8 航向，共 1,176 张图像；目前没有历史图像包的公开下载地址，按第 5 节重新采集 |

SURFdrive 链接沿用上游 README，本次整理未重新下载验证。素材解压后应直接存在
`isaac-sim_assets/Trees/` 和 `isaac-sim_assets/Grass_Countryside/`，避免多套一层目录。
保留 NVIDIA 素材原有条款。原项目的机器人训练数据只用于相应机器人实验，
下载入口见 [robot_network_training/README.md](../robot_network_training/README.md)。

## 3. 在仓库所在硬盘准备 Docker

需要 Linux、可运行 Isaac Sim 4.2.0 的 NVIDIA RTX GPU/驱动、Docker 和 NVIDIA Container Toolkit。
参见 [NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim) 与
[官方容器安装说明](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html)。
官方页面默认展示新版本；本项目固定使用 **4.2.0**。

将仓库放到容量足够的硬盘，先下载上一节的原项目素材，再运行：

```bash
cd visual_simulator_docker
sudo bash disk_docker.sh start
bash disk_docker.sh docker pull nvcr.io/nvidia/isaac-sim:4.2.0
BEE_NAV_PIP_INDEX_URL=https://pypi.org/simple bash build_on_disk.sh
```

专用 Docker 将存储写入本目录 `downloads/docker-disk/docker.ext4`，不占用系统 Docker 数据目录。
首次需要至少 42 GiB 可用空间；后续采集/输出还需硬盘空间。该 ext4 文件逻辑容量为 40 GiB，
不是需要下载或提交的资源。更多管理命令见 [DISK_BUILD.md](../visual_simulator_docker/DISK_BUILD.md)。
脚本中的 EULA/隐私变量沿用 NVIDIA 容器要求，使用前阅读相应条款。

`build_on_disk.sh` 在基础镜像已存在时直接构建，因此先 `pull` 后不需要本机历史 `.tar`。
原始镜像的本地记录摘要为
`sha256:b606646df3aab3f38ba655ce96cdf8f45a13bb167f01c4b2992e40d6f2900089`；
需要核对时用 `bash disk_docker.sh docker image inspect nvcr.io/nvidia/isaac-sim:4.2.0` 查看 RepoDigests。
Git 中的历史下载器不作为首次安装入口；部分下载器依赖未随代码分发的下载状态文件。

## 4. 下载房屋并重建真实场景

房屋来源为 [Modern House](https://grab3d.com/models/buildings/modern-house/)、
[Japanese House](https://grab3d.com/models/buildings/japanese-house/)、
[Alpine Chalet](https://grab3d.com/models/buildings/alpine-chalet/) 和
[Suburban Family House](https://grab3d.com/models/buildings/suburban-family-house/)。
保留各来源的 CC0 许可与清单。以下命令在 `visual_simulator_docker/` 中执行；需主机安装 `curl` 和 `unzip`：

```bash
mkdir -p downloads/real_preview_v1
cp CodeInsectInspired/config/resources/building_assets.json downloads/real_preview_v1/
for slug in modern-house japanese-house alpine-chalet suburban-family-house; do
  folder="downloads/real_preview_v1/$slug"
  mkdir -p "$folder"
  for name in "$slug.usdz" LICENSE.txt manifest.json; do
    curl -fL --retry 3 "https://grab3d.com/dl/$slug/$name" -o "$folder/$name"
  done
done
```

先核对下载内容与当前资源清单一致，再解包；如来源更新导致哈希不一致，停止并检查版本：

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
base = Path('downloads/real_preview_v1')
for entry in json.loads((base / 'building_assets.json').read_text()):
    slug = entry['slug']
    archive = base / slug / (slug + '.usdz')
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != entry['package_sha256']:
        raise SystemExit('SHA256 mismatch: ' + str(archive))
    print('SHA256 OK:', slug)
PY
for slug in modern-house japanese-house alpine-chalet suburban-family-house; do
  unzip -n "downloads/real_preview_v1/$slug/$slug.usdz" \
    -d "downloads/real_preview_v1/$slug/unpacked"
done
```

目录示例：`downloads/real_preview_v1/modern-house/unpacked/modern-house.usdc`，
纹理保持解包后的相对路径；`LICENSE.txt` 和 `manifest.json` 放在其上一层房屋目录。
然后通过 Isaac 自带 Python 生成场景并检查依赖，不启动 GUI：

```bash
bash disk_docker.sh docker run --rm --network none \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PWD:/workspace:rw" -w /workspace/CodeInsectInspired \
  --entrypoint /isaac-sim/python.sh insect-sim:submission \
  generate_real_world_environment.py

bash disk_docker.sh docker run --rm --network none \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PWD:/workspace:rw" -w /workspace/CodeInsectInspired \
  --entrypoint /isaac-sim/python.sh insect-sim:submission \
  tools/check_stage09_scene.py
```

输出为 `CodeInsectInspired/maps/downward_real_h40_v1/real_scene.usda` 及完整资产依赖。
生成器保留房屋/树木实际尺寸，生成地面、水池与道路，拒绝覆盖已有场景。
`config/resources/real_scene_reference.manifest.json` 保存历史参考场景的资产哈希、布局和来源；
重建后可与输出 `real_scene.manifest.json` 对照。

## 5. 重新采集、训练和测试

在仓库根目录运行以下命令；每次实验使用新的输出目录：

```bash
# 只生成并检查采样计划，不采图、不训练。
bash visual_simulator_docker/run_compact_spiral_on_disk.sh \
  --run-dir runs/reproduce_compact20_01 --plan-only

# 继续同一计划：采图、数据检查、训练与 10 m / 20 m 导航测试。
bash visual_simulator_docker/run_compact_spiral_on_disk.sh \
  --run-dir runs/reproduce_compact20_01
```

脚本使用私有磁盘上的镜像并在后台运行；检查状态/日志：

```bash
bash visual_simulator_docker/disk_docker.sh docker ps
bash visual_simulator_docker/disk_docker.sh docker logs -f bee-nav-stage23-compact20
```

固定协议为 `CodeInsectInspired/config/experiments/stage23_compact_spiral_protocol.json`：

- 40 m AGL、30° 方形下视相机、1024×1024 RGB，模型内缩放为 384×384。
- 20 m 半径内约五圈螺旋，147 个位置、各八个航向；无额外近家网格。
- 按位置分组：133/6/8 个训练/验证/预留测试位置，分别为 1064/48/64 张图。
- 训练时随机旋转图像及标签、固定圆形掩膜；验证与部署不随机旋转。
- 随机种子 10，AdamW、学习率 0.0005、batch 8、600 轮，米制标签归一化尺度 44 m。
- 完成训练后按验证损失选模型，10 m/20 m 各 64 次导航，成功半径 2 m、最多 50 步。

输出全部位于 `CodeInsectInspired/runs/reproduce_compact20_01/`，包含冻结输入、采集清单、
模型、日志、导航结果和报告。预留图像只用于离线评价；闭环导航实时渲染相机。
从源码重新渲染训练可以复现实验流程，但不同 GPU/库/素材版本可能改变像素、最佳轮次和指标。
如果需要逐像素重用本机历史数据，应另行发布原始 `capture/`、清单与校验和；
在该资源包实际发布前，本仓库不声称提供历史图像下载或逐位一致的训练复现。

## 6. 运行 demo 与纯 CPU 路径积分实验

场景准备完成后，在桌面终端、仓库根目录运行：

```bash
bash visual_simulator_docker/run_interactive_homing_demo.sh
```

demo 默认使用随 Git 提供的 epoch 484 参考模型，与新训练输出相互独立。
如需部署新模型，单独准备配置，设置模型路径和正确 SHA256，并通过 `--demo-config` 传入；
同时核对相机和控制器协议。切换模型不会自动发生。
`T` 起飞，`W/A/S/D` 平移，`Q/E` 转向，`H` 归巢，空格暂停，`R` 重置，`V` 恢复面板。
降落后显示两阶段轨迹与起降距离；日志在 `runs/interactive_demo/`。

只复现地磁路径积分统计时，不需要上述 GPU、大素材或模型；在仓库根目录创建本盘 Python 环境即可：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy matplotlib
cd path_integration_noise_simulation/magnetic_guidance
../../.venv/bin/python run_experiment.py --config config.json --output runs/reproduce_01
```

配置定义距离、噪声、试验次数等；详见该目录 README。大规模轨迹输出由脚本生成。

## 7. 不上传的内容

`.gitignore` 排除 Docker 磁盘/归档、下载缓存、生成地图、所有历史视觉实验 `runs/`、
路径积分实验输出、Python 缓存及本机私有配置。历史导航截图、视频、失败/中间模型和旧场景
不属于当前复现的必需输入。文件继续留在原硬盘，此规则不会删除本地数据。
不要用 `git add -f` 强行纳入这些目录。需要归档的选定数据以后可放实验室共享盘或独立数据发布，
并在本指南补入真实下载地址、版本、解压位置和 SHA256。
