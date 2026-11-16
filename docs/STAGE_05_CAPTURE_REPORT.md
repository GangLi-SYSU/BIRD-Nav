# 第五阶段：学习路径采样与可靠标签

日期：2026-10-26。**阶段 E 已完成 GPU 采集验收：单相机与四相机各 1176 张有效图像，均覆盖 147 个位置、每位置 8 个真实航向；四相机 5 张尾数检查也通过。** 主机和容器各 95 项测试通过。尚未训练网络或运行闭环归巢，后续数据划分与预处理留在 F。

## 已确认的实验设置

用户确认：“算了，现在就用八个航向吧。进行下一步”。该选择已写入四个下视配置及独立的[第五阶段采样协议](../visual_simulator_docker/CodeInsectInspired/config/experiments/stage05_sampling_protocol.json)。

| 设置 | 确认值 |
| --- | --- |
| 学习位置 | 原 bee `[5,150,0.35]`，147 个位置，seed 10 |
| 航向 | 每位置随机初始航向，加 `0/45/90/135/180/225/270/315°` |
| 图像数 | **147×8=1176 张**，每个航向实际渲染 |
| 相机 | 1024×1024、HFOV 30°、固定下视、光心距平坦地面 10 m |
| 验收 | 单相机完整采集，再以四相机复核相同位姿计划 |
| 学习数据 | 两次完整采集分别保存；后续使用四相机集，不合并为 2352 张 |
| 网络与训练参数 | 原 attention、batch 8、150 epochs 等设置保留；尚未训练 |

`capture.headings_per_position=8` 是显式配置项；缺省时仍在启动 Kit 前停止。同一位置的八个航向共享 `split_group`，供 F 阶段统一划分，防止相近图像落入不同集合。图像数比单航向增加八倍；实际优化步数待 F 的分组划分后计算并记录。

已抽取同一位置的[八航向实际照片](../visual_simulator_docker/CodeInsectInspired/runs/stage05_capture/eight_heading_validation_01/eight_heading_contact_sheet.png)，另有[圆锥路标的八航向示例](../visual_simulator_docker/CodeInsectInspired/runs/stage05_capture/eight_heading_validation_01/eight_heading_cone_contact_sheet.png)。它们来自八次实际相机姿态拍摄，未经全景移位或训练增强。采集按学习路径逐个设置相机位姿，不运行归巢控制。

[图像方向诊断](../visual_simulator_docker/CodeInsectInspired/runs/stage05_capture/eight_heading_validation_01/heading_rotation_diagnostic.json)比较了实际偏航相差 90°的照片：顺时针对齐后的平均 RGB 差约 0.508–0.545/255，反向对齐约 31.6–31.9/255，符合“图像上方随机头转动”的定义。这是方向诊断，不是新的验收阈值；原始学习 PNG 没有被旋转处理。

## 修改一：精确调度每个样本

**作用：** 修复原四相机采集丢失 3 张的尾数问题。

[capture_geometry.py](../visual_simulator_docker/CodeInsectInspired/insect_utils/capture_geometry.py)将位置、航向和文件编号先组成明确的样本表，再分配给相机。147 张图由四相机处理时，共 36 个完整批次加一个 3 张批次；空闲相机不产生额外文件。相机数量改变时，位置和随机航向保持相同。

八航向的 1176 张恰好能被 4 整除，因此另外执行真实 GPU 尾数检查：四相机拍摄 5 张，调度为 `[4,1]`，5 张全部通过图像、标签与稳定记录校验。该检查使用 `engineering_smoke` 标记和独立目录，不混入学习集。

[render_dataset.py](../visual_simulator_docker/CodeInsectInspired/render_dataset.py)的下视分支现连接独立的[downward_capture.py](../visual_simulator_docker/CodeInsectInspired/insect_utils/downward_capture.py)，沿用第四阶段的相机工厂和稳定取图条件。每次返回图像后，按样本编号保存 PNG，避免依赖旧 BasicWriter 的相机目录排序关联标签。全景分支保留。

## 修改二：固定高度并使用实测位姿计算标签

**作用：** 防止复制路标坐标时误改飞行高度，或将旧图像关联到新标签。

本场景所有路标低于飞行平面，采样无需为了绕开地面路标改变 XY；保留原学习位置。每张图检查实际相机矩阵与计划位姿一致，并记录真实光心 Z、参考地面 Z 和 AGL。光心距平坦地面维持 10 m，不将路标顶部当作地面。

标签从渲染器返回的矩阵推导实际位置与机体偏航，再计算：

```text
target_B = Rz(-yaw) × (home_xy - actual_position_xy)
actual_position_xy + Rz(yaw) × target_B = home_xy
```

非零家位置、非零地面高度、四基本航向、任意偏航，以及实际位姿的小偏差均有测试。错误 pose ID、错误高度、未稳定图像或旧帧会拒绝进入数据集。

## 修改三：保存可复核的样本清单

**作用：** 将图像、位姿、帧号和标签关联到同一条观测记录。

每次采集产生以下文件；完成清单仅在本次全部图像采完并通过校验后写入：

- `capture_plan.json`：全部样本、路径、航向、足迹和相机调度。
- `Replicator/rgb/rgb_000000.png` 及 `.png.json`：完整 RGB 与图像哈希、实际位姿、相机路径和稳定历史。
- `samples.jsonl`、`samples.csv`：包含 `sample_id, pose_id, frame_id, image, x,y,z,yaw, target_x,target_y, camera_id, split_group` 等字段。
- `dataset_navigation.csv`：保留原四列格式；通过原代码的解析器逐条回读，核对文件名、位置、偏航和标签。
- `dataset_manifest.json`：仅在全部文件校验后生成，保存相机内外参约定、场景指纹、样本数量和文件哈希。

校验会拒绝少图、多图、错序、错误相机、PNG 变化、清单或 CSV 不一致。原有运行控制器负责失败重试归档与完成记录；已完成清单不会被重写。训练阶段保持 F 的数据划分与预处理限制，闭环保持 H 的控制接入限制。

## 修改四：分析步距、足迹与评测隔离

八航向的计划分析保存在[确认方案报告](../visual_simulator_docker/CodeInsectInspired/runs/stage05_capture/confirmed_eight_heading_plan_01/plan_report.json)。历史 `proposal_147_01` 是此前单航向提案，予以保留，不作为本次实验配置或数据集。

| 指标 | 八航向方案的离线计算值 |
| --- | ---: |
| 位置数 / 图像数 | 147 / 1176 |
| 路径总长 | 169.542 m |
| 离家最大半径 | 10.996 m |
| 相邻位置最大间距 | 2.307 m |
| 相邻视图最小地面投影面积重叠率 | 48.73% |
| 地面投影并集面积估算 | 517.25 m² |
| 最大半径圆盘内覆盖率估算 | 100%（0.25 m 网格中心估算） |
| 现有评测起点落入学习足迹数 | 32 / 80 |

查看[八航向学习足迹与评测起点图](../visual_simulator_docker/CodeInsectInspired/runs/stage05_capture/confirmed_eight_heading_plan_01/learning_coverage.png)和[相邻步距与面积重叠图](../visual_simulator_docker/CodeInsectInspired/runs/stage05_capture/confirmed_eight_heading_plan_01/adjacent_overlap.png)。蓝色为学习足迹，黑线为学习路径，绿/红叉区分落在/落在外的评测起点；图中展示家附近区域，并非整个 400 m 场地。相邻面积重叠率按每位置的首个航向计算，并集使用全部八个航向。

并集和圆盘覆盖率使用 0.25 m 网格中心估算，**未计算路标遮挡或高出地面的可见表面**。面积重叠率也不同于沿轴的线性重叠率。覆盖图不代表导航成功率，更不意味着模型能从学习区域外归巢。

80 个评测起点只用于分析绘图，未加入学习样本。当前学习半径约 11 m；400 m 场地提供空间余量，尚未将学习飞行扩大到全场。

## 实际验收结果与数据位置

本次实际数据来自 400×400 m 路标场景。四个配置均已同步八航向参数；distinct/repeated/low_texture 场景尚未执行本阶段采集。

| 验收项 | 图像数 | 完整位置数 / 每位置航向 | 等待更新次数 | 最大末次 RGB 变化 |
| --- | ---: | --- | --- | ---: |
| 四相机尾数检查 | 5 | 5 / 1（工程检查） | 4–8 | 0.445/255 |
| 单相机完整采集 | 1176 | 147 / 8 | 4–10 | 0.441/255 |
| 四相机完整采集 | 1176 | 147 / 8 | 4–10 | 0.433/255 |

全部图像均满足每次独立取图的连续两次变化 ≤0.5/255，最多 90 次更新、2 个渲染子帧的策略保持不变。实际 Z 和 AGL 在浮点误差范围内恒为 10 m，八个航向间隔的最大数值误差约 `1.14e-13°`。两次完整采集的位姿签名相同，全部 PNG、CSV、侧录与完成清单校验通过，合计验收 2357 张图像。

[单/四相机同位姿抽查](../visual_simulator_docker/CodeInsectInspired/runs/stage05_capture/eight_heading_validation_01/single_vs_four_early_rgb.json)比较了 24 对照片，最大平均 RGB 差约 0.680/255；该数值作为诊断保存，没有新增或放宽验收阈值。

纯取图耗时总和约为单相机 637.24 s、四相机 1649.17 s，平均分别为 0.542 s/张与 1.402 s/张。单张范围分别约 0.519–1.602 s、1.339–3.909 s；这些数值不含 Kit 启动、PNG 压缩保存与文件校验。

后续 F 使用[四相机数据清单](../visual_simulator_docker/CodeInsectInspired/runs/stage05_capture/eight_heading_validation_01/runs/downward_h10_hfov30_landmarks/four_cameras/capture/dataset_manifest.json)，原始照片在[对应 RGB 目录](../visual_simulator_docker/CodeInsectInspired/runs/stage05_capture/eight_heading_validation_01/runs/downward_h10_hfov30_landmarks/four_cameras/capture/Replicator/rgb)。两个完整数据集分别保存，不合并；尾数检查只作工程验收。训练配置保留 batch 8、150 epochs，本阶段执行的优化步数为 **0**，实际训练更新预算由 F/G 根据最终划分记录。

## 复查命令与下一阶段

测试覆盖精确尾数、相机数量与航向预算分离、实际位姿标签、旋转足迹相交、真实 PNG/CSV 解析及文件篡改拒绝。PNG/CSV 测试使用临时人工图像，不能当作 Isaac 真实采集验证；缺少 NumPy/Pillow 的环境会跳过这些文件读取测试。

主机 Python 3.8 与容器 Python 3.10 的 **95 项测试均全部通过，无跳过**。旧基线协议、Black、Flake8 及 `git diff --check` 通过；GPU 三项采集验收均通过。

在 `visual_simulator_docker/CodeInsectInspired/` 中运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_*.py'

# 分析已经确认的八航向配置，不启动 Isaac；输出目录必须尚不存在。
python tools/check_stage05_capture.py --confirmed-plan \
  --output runs/stage05_capture/my_eight_heading_plan
```

在现有 Isaac 容器的 `/isaac-sim/CodeInsectInspired` 中执行完整验收；输出目录必须尚不存在，失败后显式 `--resume` 才复用对应运行，代码或配置变化则使用新目录：

```bash
../python.sh tools/check_stage05_acceptance.py \
  --output runs/stage05_capture/my_eight_heading_validation
```

[最终真实采集验收记录](../visual_simulator_docker/CodeInsectInspired/runs/stage05_capture/eight_heading_validation_01/acceptance.json)按尾数检查、单相机、四相机逐项保存通过结果；[启动与完成日志](../visual_simulator_docker/CodeInsectInspired/runs/stage05_capture/validation_01.launch.log)记录三项通过及程序正常退出。各运行的 `logs/capture_01.log` 保存逐张进度。

[阶段完成记录](../visual_simulator_docker/CodeInsectInspired/runs/stage05_capture/eight_heading_validation_01/completion.json)汇总数据位置、验收指标、源码与协议哈希、主机/容器测试结果，以及既有文件和权重的保留核对结果。

只读复核已经完成的四相机数据，不启动 Kit：

```bash
../python.sh tools/check_stage05_capture.py --verify-run \
  runs/stage05_capture/eight_heading_validation_01/runs/downward_h10_hfov30_landmarks/four_cameras
```

阶段 E 的完整采样已完成。F 的数据划分与预处理、H 的闭环控制仍留待后续阶段；采集验收不能作为模型导航精度或归巢成功率结果。
