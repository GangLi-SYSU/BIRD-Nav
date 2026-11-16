# 400 m 场地与简单几何路标预览

日期：2026-10-23。本次按用户要求扩大场地、加入简单几何路标并打开 Isaac 查看效果；未启动正式采集、训练或归巢实验。

## 场地范围为什么改为 400×400 m

原来的 100 m 地面足以容纳 147 个学习位置和最远 25 m 的测试起点，但无法保证连续错误动作始终留在场内。按当前协议，中心的最坏离家距离上界为：

```text
25 m 起点 + 50 步 × 2.5 m/步 = 150 m
地面半宽至少需要：150 + 3.789 m 足迹外接圆 + 5 m 余量 ≈ 158.789 m
对应边长至少约 317.58 m，当前采用 400 m。
```

新地面 XY 均为 `[-200,200] m`，米制、Z-up，地面 `z=0`。暂定 10 m / 30°方形下视相机的保守中心范围约为各轴 `±191.21 m`，覆盖当前起点与动作预算。这是场地的几何预算，视觉可归巢距离仍需后续测量；扩大起点、步数或步长后必须重新计算。

## 路标外观、布局与高度

| 项目 | 当前设置 |
| --- | --- |
| 数量 | 192 个：立方体/长方体、圆柱、圆锥、球体各 48 个 |
| 中心区域 | 96 个，中心坐标随机分布在 XY 各 `[-36,36] m` |
| 外围 | 96 个，分散在场地外围，物体中心距场边至少 15 m |
| 颜色与尺寸 | 12 种颜色；各物体尺寸随机变化，立方体还有随机偏航 |
| 高度 | 实际约 `0.936–2.960 m`，与 `z=10 m` 相机平面的最小竖直距离约 7.040 m |
| 防重叠 | 以物体水平包围圆保守检查，边缘之间至少留 1 m |
| 重建 | 随机种子 10，保存每个物体的位置、类型、尺寸、颜色和高度 |

中心放得较密，目的是先观察家附近的视觉线索；这是一项明确的场景设计。布局没有读取学习轨迹，也没有给“家”放专属标记。物体均为静态碰撞几何，没有刚体或动力学控制。

地面是完整 400 m 范围的新纹理，`8192×8192`、每纹素约 4.883 cm，未把旧物体整体放大。相比旧 100 m / 4096 纹理，每米地表像素密度降低了一半；10 m / 30°地面足迹约对应 110 个纹理像素。三维几何轮廓由渲染器生成，不受这一地表纹素数直接限制。

看到路标顶部时，相机与可见表面的距离小于 10 m，局部图像足迹也会变小。后续相机与数据检查会记录这一点，不把所有像素都当成地面 `z=0` 的投影。

## 文件与重建方式

场景保存在本硬盘中：

```text
visual_simulator_docker/CodeInsectInspired/maps/downward_landmarks_v1/
├── landmarks.usda
├── ground.png
├── landmarks.manifest.json
└── texture_overview.png
```

目录约 105 MB。原有 `downward_flat_v1/` 三种诊断场景保留。

新增 [生成器](../visual_simulator_docker/CodeInsectInspired/generate_landmark_environment.py)与[独立实验配置](../visual_simulator_docker/CodeInsectInspired/config/experiments/downward_h10_hfov30_landmarks.json)。原 attention 网络及训练、控制超参数沿用已有配置；默认平地实验配置未被替换。

在 `CodeInsectInspired/` 目录生成另一个版本（目标目录必须不存在）：

```bash
/media/lg/lg-512/conda_envs/bird-nav/bin/python generate_landmark_environment.py --output-dir maps/downward_landmarks_v2
```

## 在 Isaac 中查看

从宿主机桌面终端执行，已有同名预览应先关闭：

```bash
cd /media/lg/lg-512/new_workspace/Bird_Nav_ws/Bee-Nav/visual_simulator_docker
bash view_downward_scene_on_disk.sh --scene maps/downward_landmarks_v1/landmarks.usda
```

默认先显示中心区域，便于看清物体。`Stage C: terrain preview` 面板提供：

- **Whole field overview**：整个 400 m 场地。
- **Central learning area**：中心路标区域的斜视图。
- **Next landmark at 10 m**：把检查相机移到下一个路标上方，始终保持相机 `z=10 m`。
- **Downward inspection at 10 m**：显示当前下视检查相机。

若面板关闭，也可在视口相机下拉框选择 `StageCOverview`、`StageCLearningArea` 或 `StageCAt10m`。该预览不是机体坐标、视场与采集/推理一致性的正式标定；阶段 D 仍待实施。

## 验证与预览证据

OpenUSD 核对了实际地面边界、193 个几何体（地面加 192 个路标）、物体贴地与高度、完整文件依赖及飞行预算。147 个学习位置、80 个测试起点通过；宿主 Python 3.8 与容器 Python 3.10 中累计 **49 项测试通过**。

- [实际几何与资源报告](../visual_simulator_docker/CodeInsectInspired/runs/landmark_scene_v1_check/scene_report.json)
- [容器检查日志](../visual_simulator_docker/CodeInsectInspired/runs/landmark_scene_v1_verification_final.log)
- [中心区域实际渲染图](../visual_simulator_docker/CodeInsectInspired/runs/landmark_scene_v1_preview_final/central_oblique.png)
- [全场与四种路标的六图预览](../visual_simulator_docker/CodeInsectInspired/runs/landmark_scene_v1_preview_final/contact_sheet.png)
- [六帧预览报告](../visual_simulator_docker/CodeInsectInspired/runs/landmark_scene_v1_preview_final/preview_report.json)和[最终窗口日志](../visual_simulator_docker/CodeInsectInspired/runs/landmark_scene_v1_viewer_final.log)

最终导出在同一个 GUI 实例中完成：四种路标的局部画面均有有效纹理与几何内容，黑像素区域比例为零；导出后回到中心区域，窗口保持可交互。预览图仅用于本次场景检查，正式训练数据数量仍为零。

首次构建的 USD 材质语法已修正；原失败文件和日志保存在本次运行目录，当前清单对应已通过检查的资源。预览导出还修正了安装版本 Replicator 的 `detach` 参数兼容问题，使用明确的 render-product 路径进行清理。
