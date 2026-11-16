# 第九阶段：正常尺寸房屋、树木和水池预览

日期：2026-10-28。场景预览和 7 张实际 RGB 照片已完成，Isaac 查看窗口保持打开。六个下视图的实际高度、位姿、视场和尺寸核对通过，帧间稳定阈值尚未通过。本阶段没有生成新训练数据，也没有运行训练或闭环归巢。上一轮 10 m / 30° 实验的地图、照片和模型保留。

## 1. 独立场景与正常物理尺寸

新增 `visual_simulator_docker/CodeInsectInspired/maps/downward_real_h40_v1/real_scene.usda`，不替换旧地图。

- 地面 400×400 m；目前布置地物的区域约为 X∈[-90,90]、Y∈[-70,75] m。外围地面用于避免预览视场越界，不代表已经建好整个 400 m 区域的导航基准。
- 12 栋房屋，包含现代平顶住宅、日式住宅、木屋和郊区住宅四种资产。宽约 11.5–14.9 m、进深约 10–11.1 m、高约 5.7–10.1 m，按原始尺寸放置。
- 42 棵树、7 种资产，最高约 19.74 m；保留原始树冠大小，没有为适应视场而缩小模型。
- 一个 12×7 m 水池，包含池沿、铺装及静态水面，另外设置道路、人行道和公园小径。水池是程序生成的外观模型，没有水体动力学。

房屋来源为 [Grab3D Cozy Homes](https://grab3d.com/kits/cozy-homes-world/)，下载的四个包附带 CC0 许可、源清单和 SHA-256。它们是有纹理的轻量风格化模型，**并非实景扫描**。树木和草地纹理来自本项目已有 Isaac 素材，继续适用原有条款。下载文件保存在 `visual_simulator_docker/downloads/real_preview_v1/`。

房屋 USD 使用 Y-up、米；树木 USD 使用 Z-up、厘米。生成器分别做坐标轴变换和单位换算，并将模型底部放到地面，避免房屋躺倒或树木出现百倍尺寸错误。布局使用固定世界坐标，不读取家位置、学习路径或训练标签。

## 2. 相机设置和作用

独立配置为 `config/experiments/stage09_preview.json`；它是预览协议，不接入训练控制器。

| 设置 | 数值与作用 |
| --- | --- |
| 相机 | 机体固定、水平姿态、垂直下视、理想针孔 |
| 高度 | 光心距平地 40 m，最高树冠之上约 20.26 m |
| 分辨率 | 实际 RGB 为 1024×1024，未裁切 |
| 默认视场 | 水平、垂直均为 70°，平地覆盖约 56.02×56.02 m |
| 对照视场 | 60° 覆盖约 46.19 m；80° 覆盖约 67.13 m |
| 图像方向 | 零航向时，世界 +X 为图像上方，世界 -Y 为右方 |

作用是让一张图同时包含屋顶、树冠、水池或道路组合。覆盖公式为 `2 × 高度 × tan(视场角/2)`，只适用于相应水平高度平面；树冠和屋顶更靠近相机，不能将平地覆盖面积直接用于它们。若以后仍采用 384×384 模型输入，默认视场的平地分辨率约为 0.146 m/像素；本阶段没有修改或运行模型。

## 3. 打开预览与切换视图

从 `Bee-Nav/` 运行，使用已启动的硬盘 Docker 服务：

```bash
bash visual_simulator_docker/view_real_scene_on_disk.sh
```

在 `Real scene preview: 40 m AGL` 面板中：

- `Scene overview`：斜视整体布局，属于查看场景的相机。
- `Downward: junction / houses / pool / park`：切换四个位置的下视画面。
- `Downward HFOV 60 / 70 / 80 deg`：保持当前下视位置和 40 m 高度，比较视场。

这些查看位置不是训练采样点。浏览相机在导出期间与实际拍照相机分开，避免 Isaac 自动取景或用户拖动改变拍照高度。导出后释放额外 RGB 渲染产品，界面更新限制为 15 FPS，减少持续占用。

## 4. 验证与照片

实际照片保存在 `visual_simulator_docker/CodeInsectInspired/runs/stage09_real_preview/preview_05/`：

- [整体布局](../visual_simulator_docker/CodeInsectInspired/runs/stage09_real_preview/preview_05/overview.png)、[住宅区 40 m / 70°](../visual_simulator_docker/CodeInsectInspired/runs/stage09_real_preview/preview_05/houses_h40_fov70.png)、[水池区](../visual_simulator_docker/CodeInsectInspired/runs/stage09_real_preview/preview_05/pool_h40_fov70.png)、[公园区](../visual_simulator_docker/CodeInsectInspired/runs/stage09_real_preview/preview_05/park_h40_fov70.png)。
- 同一路口：[60°](../visual_simulator_docker/CodeInsectInspired/runs/stage09_real_preview/preview_05/junction_h40_fov60.png)、[70°](../visual_simulator_docker/CodeInsectInspired/runs/stage09_real_preview/preview_05/junction_h40_fov70.png)、[80°](../visual_simulator_docker/CodeInsectInspired/runs/stage09_real_preview/preview_05/junction_h40_fov80.png)。
- [相机与稳定性凭证](../visual_simulator_docker/CodeInsectInspired/runs/stage09_real_preview/preview_05/preview_report.json)和[照片核对结果](../visual_simulator_docker/CodeInsectInspired/runs/stage09_real_preview/preview_05/artifact_check.json)。

7 张 PNG 均为 1024×1024 RGB。六张下视照片的光心实测 Z=40 m，实际投影视场与 60°/70°/80° 的目标差小于 0.00001°；最后一次帧间 MAE 为 0.55–0.67/255，均明确标记 `settled: false`。斜视图只用于查看布局，不代表无人机的下视输入。

几何检查报告：`CodeInsectInspired/runs/stage09_real_preview/geometry_report.json`。实际尺寸、着地、相机边界、93 个依赖校验通过，其中额外检查了 38 个写在 MDL 内的纹理引用。普通 USD 依赖扫描会漏掉其中部分文件，生成器已补齐，并把它们加入哈希清单。

新增 5 项单元测试覆盖 Y-up 房屋、厘米树木、空模型拒绝、MDL 贴图完整性及 40 m / 70° 图像轴与投影；主机测试和新增源码的 Flake8 检查通过。真实渲染结果记录在预览目录的 `preview_report.json`，下视图核对实际位姿、递增帧和连续两次 RGB MAE≤0.5/255。

静止相机在原渲染设置下，首个位置的帧间差约为 0.64/255，未满足原稳定阈值。增加至 8 子帧以及开启 TAA 的探测均未解决（约 0.8/255），原因尚未确认，不能断言只是随机噪声。最终预览恢复为原来的**抗锯齿关闭、两子帧**。

预览每个位置最多等待 24 次更新；若实际位姿、递增帧、相机投影和图像尺寸正确，但帧间稳定条件未满足，仍可保存供查看，并在逐帧凭证中写明 `settled: false`。总报告使用 `preview_ready` 状态，`dataset_capture_accepted` 始终为 `false`，不把它冒充为采集验收通过。正式采集源码及其最多 90 次更新、连续两次 MAE≤0.5/255 的门槛均未改动。需要在后续正式采集之前定位差异原因并重新验收。

```bash
# 以下从 CodeInsectInspired/ 运行
python3 -m unittest discover -s tests -p 'test_stage09_scene.py' -v
```

下一步应先评价这些照片中的地物外观、组合密度和遮挡，再确定新视场对应的学习范围、采样间距与划分；不能直接沿用旧窄视场的 147 个位置并据此宣称大场景导航成立。本次预览不证明过拟合已经改善。

后续进展：[第十阶段](STAGE_10_REAL_SCENE_TRAINING.md)已定位并修复抗锯齿被 DLSS 覆盖的问题，64 张真实相机工程照片通过原稳定门槛；本阶段的预览图片仍不作为训练数据。
