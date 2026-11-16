# 第八阶段：可辨识路标与 384×384 输入联合改进

日期：2026-10-27。**1176 张新图采集、数据验收及 150 轮正式训练全部完成，共 19,950 次更新。** 最终权重重载、ONNX 和实际 384×384 输入检查通过。测试平均向量误差由 9.62 m 降至 7.17 m，平均方向误差由 91.68° 降至 64.31°；改善明显，但仍有过拟合和较大预测误差，未运行闭环归巢。

## 用户确认的目标与实验范围

用户要求：增大路标体积、增加颜色与形状变化、降低没有路标候选的图像比例；模型改用 384×384 输入并适配，再重新训练。随后确认训练图零路标候选比例不超过 5%、路标高度不超过 3 m，并要求先打开 Isaac 查看场景。

本轮同时改变场景和网络输入，比较的是联合方案的效果，不能分别量化场景和输入的独立贡献。保持 400 m 地面、原地面纹理、10 m AGL、30°×30°视场、1024×1024 原始 RGB、原 147 位置、每位置八个真实航向、位置分组和原优化设置。

## 步骤一：扩大并丰富路标

新增 [generate_identifiable_landmarks.py](../visual_simulator_docker/CodeInsectInspired/generate_identifiable_landmarks.py)，独立生成 `maps/downward_landmarks_v2/`；上一版地图保持原样。

| 项目 | 上一轮 | 本轮 |
| --- | --- | --- |
| 路标总数 | 192 | 308 |
| 中央 / 外围 | 96 / 96 | 212 / 96 |
| 形状 | 4 类 | 8 类 |
| 形状—主色组合 | 12 种 | 159 种 |
| 表面 | 单色 | 主色与黑/白非对称面片 |
| 尺寸 | 较小立方体、半径 0.55–1.15 m 等 | 方块边长 2.9 m，多数圆形路标半径 1.4–1.75 m；球半径 1.3–1.49 m |
| 最高表面 | 不超过 3 m | 不超过 3 m |

保留方块、圆柱、圆锥、球，新增三角柱、六角柱、梯形柱和箭形柱。箭头朝向及色块方向随机，不指向家。布局使用固定世界坐标三角点阵加小扰动，外围随机布置；不读取家位置、学习路径、航向或标签。物体包围圆之间至少留 0.25 m 间隔。

**作用：** 让窄视场中更容易遇到路标，并用轮廓、配色及非对称图案提供方向线索。增大横向尺寸，保持飞行净空。实际是否改善泛化仍需训练结果验证。

## 步骤二：覆盖与 Isaac 预览验收

沿用上一轮的真实位姿和分组，按“物体 XY 包围圆与地面视场相交”筛查：

| 集合 | 图像数 | 上一轮零候选图 | 本轮零候选图 |
| --- | ---: | ---: | ---: |
| 训练 | 1064 | 316（29.7%） | 0（0%） |
| 验证 | 48 | 16（33.3%） | 0（0%） |
| 测试 | 64 | 36（56.3%） | 0（0%） |
| 全部 | 1176 | 368（31.3%） | 0（0%） |

训练帧大部分有 2–4 个几何候选，达到用户的 ≤5% 目标。候选不是实际可见语义分割：包围范围与地面最大投影会高估可见性，仍须看实际图片。

[场景几何检查](../visual_simulator_docker/CodeInsectInspired/runs/stage08_optimization/scene_checks_01/scene_report.json)验证 308 个网格的点与面索引、着地、顶高、非对称材质子集和依赖；地面 PNG 与上一版 SHA-256 一致。原相机与图像稳定条件保留。

Isaac 已打开新版场景，六张实际检查图通过，保存在 [scene_preview_01](../visual_simulator_docker/CodeInsectInspired/runs/stage08_optimization/scene_preview_01)：[中央区域](../visual_simulator_docker/CodeInsectInspired/runs/stage08_optimization/scene_preview_01/central_oblique.png)、[10 m 方块视图](../visual_simulator_docker/CodeInsectInspired/runs/stage08_optimization/scene_preview_01/cube_at10m.png)。这些是预览，不混入训练集；逐个路标查看模式会把相机居中，正式采样仍走原学习路径。

## 步骤三：方形模型和统一工厂

新增 [attention_model.py](../visual_simulator_docker/CodeInsectInspired/insect_utils/attention_model.py)，提供原模型与 `nadir384x384` 两种显式变体。原 `simple_network.py` 不修改。

下视变体保留原 Inception、空间 attention、卷积层、tanh 和 Xavier 初始化；仅将内部 resize 改为 384×384，末端为 `[N,4,7,7]`，用 `flatten(1)` 展平为 196，并适配 `Linear(196,16)`。总参数 **10,628**。相同 seed 下卷积主干及输出 fc2 的初始化与原模型一致。

**作用：** 保持方形图像几何和二维特征格，避免原 `192×1800` 拉伸。新模型从头训练，旧模型不作为新模型初始化。

训练、资源探测、保存重载、独立离线评估及 ONNX 使用同一工厂；`train.test_model` 的下视分支也调用它。输入仍是原始 float32 RGB / 255、`[N,3,1024,1024]`，网络内部做统一 resize。下视闭环仍留在 H，未借此启动归巢。

[stage08_training_protocol.json](../visual_simulator_docker/CodeInsectInspired/config/experiments/stage08_training_protocol.json)显式声明新变体和参数量。训练控制器新增 `--protocol`，清单冻结模型、协议、数据和源码，校验声明参数量；旧全景变体保留。

## 步骤四：验证、重新采图与正式训练

主机测试 137 项中通过 123 项，因缺少 torch/ONNX 运行依赖跳过 14 项；容器 **137 项全部通过**。新增测试覆盖路标几何、间隔、配色多样性、方形前向、梯度、原模型初始化兼容、方形权重重载及 ONNX。见[容器日志](../visual_simulator_docker/CodeInsectInspired/runs/stage08_validation_01.log)。

实际流程全部完成：按用户确认关闭预览，重新采集 147×8=1176 张照片；完成像素、标签、分组与读取检查；独立工程探测和 3 轮 smoke 通过后，从新初始化训练 **150 轮、19,950 次正式更新**。保持 seed 10、MSE、AdamW、lr 0.0005、batch 8、4 workers，无额外增强、混合精度、早停或学习率调度。交付第 150 轮模型，不按测试集挑权重。

新图 1176 个 SHA-256 均不同，逐张像素读取与在线 RGBA / 原 ToTensor 一致。新旧样本的位置、航向和标签逐元素差值均为 0，位置分组及索引一致，仍为 133/6/8 位置、1064/48/64 图，无同位置跨集合。每帧稳定等待 4–8 次更新，最大接受 RGB MAE 为 0.45944/255，未放宽稳定条件。见[配对数据核对](../visual_simulator_docker/CodeInsectInspired/runs/stage08_optimization/paired_data_comparison.json)。

读取进程 0/2/4 的实际探测均通过，保留原 4 workers。smoke 独立完成 3 轮、12 次有效更新，保存/重载预测差为 0，ONNX 两图最大差约 1.94e-7。正式训练重新初始化，初始参数摘要与 smoke 的新初始化相同，不继承 smoke 权重。24 次 probe 与 12 次 smoke，共 36 次工程更新另行记录。

实际输出为：

- `runs/downward_h10_hfov30_landmarks_v2/revised_01/capture/`：本轮新照片。
- `runs/stage08_data/position_groups_01/`：本轮准备数据，保持相同位置分组。
- `runs/stage08_training/attention384_landmarks_v2_01/`：工程探测、smoke、训练和最终评估。

原始 G 源码与清单已另存 `runs/stage08_optimization/historical_stage07_sources/`，上一轮照片、划分、模型、报告均保留。数据准备的四个核心源码保持原样，避免破坏既有 F 验证。

[实际调度脚本](../visual_simulator_docker/CodeInsectInspired/runs/stage08_optimization/run_retraining.py)已执行完成。每个重阶段使用独立进程，采集退出后释放 Kit；完成状态保存在 `runs/stage08_optimization/progress.json`，总日志为 [stage08_retraining_01.log](../visual_simulator_docker/CodeInsectInspired/runs/stage08_retraining_01.log)。以下为本次使用的容器内执行顺序，从 `SIM/` 运行；重复实验应使用新目录，保留本轮产物：

```bash
../python.sh run_downward_homing.py \
  --config config/experiments/downward_h10_hfov30_landmarks_v2.json \
  --stage prepare --run-id revised_01
../python.sh run_downward_homing.py \
  --config config/experiments/downward_h10_hfov30_landmarks_v2.json \
  --stage capture --run-id revised_01
../python.sh tools/check_stage06_data.py \
  --source-run runs/downward_h10_hfov30_landmarks_v2/revised_01 \
  --output runs/stage08_data/position_groups_01
../python.sh run_downward_training.py \
  --prepared-data runs/stage08_data/position_groups_01 \
  --protocol config/experiments/stage08_training_protocol.json \
  --output runs/stage08_training/attention384_landmarks_v2_01 --stage all
```

## 结果与解释边界

下表全部使用各自**第 150 轮固定权重**重新评估；上一轮同样是下视相机，区别是较小路标与原 `192×1800` 内部 resize。本轮最后一轮更新过程中的平均训练损失为 0.623745 m²，与固定权重重评训练集的 0.356564 m² 是不同统计量。

| 集合 / 方案 | 分量 MSE（m²） | 平均向量误差（m） | 平均距离误差（m） | 平均最短角误差 |
| --- | ---: | ---: | ---: | ---: |
| 训练 / 上一轮 | 0.157106 | 0.466 | 0.289 | 7.01° |
| 训练 / 本轮 | 0.356564 | 0.712 | 0.441 | 8.49° |
| 验证 / 上一轮 | 29.284454 | 6.807 | 2.368 | 91.71° |
| 验证 / 本轮 | 12.114713 | 4.297 | 2.402 | 46.43° |
| 测试 / 上一轮 | 55.824698 | 9.619 | 3.596 | 91.68° |
| 测试 / 本轮 | 32.281675 | 7.166 | 3.157 | 64.31° |

测试向量误差均值下降 **25.5%**，方向误差均值下降 **29.9%**，分量 MSE 下降 **42.2%**。并非所有指标改善：训练集拟合略差，验证距离误差均值略增。初始验证 MSE 12.437225，最终 12.114713；验证误差在中期下降后又回升，仍存在明显过拟合。

| 本轮集合 | 向量误差 p50 / p90 / p95（m） | 距离误差 p50 / p90 / p95（m） | 角误差 p50 / p90 / p95 |
| --- | --- | --- | --- |
| 训练 | 0.637 / 1.305 / 1.548 | 0.359 / 0.888 / 1.162 | 4.44° / 18.44° / 27.98° |
| 验证 | 4.221 / 7.837 / 8.320 | 1.953 / 4.706 / 5.382 | 31.36° / 96.18° / 108.53° |
| 测试 | 6.148 / 12.347 / 14.743 | 2.931 / 5.891 / 6.817 | 55.47° / 140.67° / 161.30° |

三个集合均没有零目标/零预测，方向样本排除数为 0。测试仍有 **16/64 张图的方向误差超过 90°**；因此目前不能把误差下降解释为已具备可靠归巢能力。该计数是相关航向样本上的描述，不是 64 次独立导航实验。

两项改动同时实施，结果仅用于评估组合方案，无法分别归因。测试点位置已用于上一轮诊断，后续独立泛化结论需要另行预留位置或场景；当前仍没有闭环归巢成功率。

## 最终模型、图表与完整性

正式 epoch 用时合计约 **40.60 分钟**，不含采集、数据验收、工程探测和最终评估。训练进程与子进程峰值 PSS 约 2.84 GiB、RSS 汇总约 8.24 GiB，最低主机可用内存约 6.20 GiB；PyTorch CUDA 峰值分配/保留约 145.2 / 176 MiB。RSS 可能重复计算共享页面，CUDA 分配器统计不等于整卡占用。运行环境为 Python 3.10.14、PyTorch 2.4.0+cu118、CUDA 11.8、RTX 4060；没有加载 Kit 进行训练。

最终权重重载预测最大差 **0**。ONNX 两张图的参考计算最大差 **1.91e-6**，通过 `atol=1e-5, rtol=1e-4`。导出的输入是 float32 RGB / 255、`[N,3,1024,1024]`，内部 resize 为 384×384；输出为米制机体系 `[forward,left]`。没有验收 ONNX Runtime 后端。

对实际交付权重再次挂钩检查，第一层收到 `[1,3,384,384]`，末端为 `[1,4,7,7]`，展平 196，且与方形 bilinear resize 逐元素一致。见[实际输入检查图](../visual_simulator_docker/CodeInsectInspired/runs/stage08_training/attention384_landmarks_v2_01/analysis/actual_network_input.png)与[input_inspection.json](../visual_simulator_docker/CodeInsectInspired/runs/stage08_training/attention384_landmarks_v2_01/analysis/input_inspection.json)。F 数据验收工具同时保留原网络的辅助前向检查；它不代表本轮训练仍在使用原长条输入，生产模型以这里的实际交付检查为准。

- [正式模型](../visual_simulator_docker/CodeInsectInspired/runs/stage08_training/attention384_landmarks_v2_01/train/model_attention.pth)：SHA-256 `ad3aaf5d6a7b9d664554cdd8a520cc2119f0b9b36ac73f755dd00fa259050e96`。
- [ONNX 模型](../visual_simulator_docker/CodeInsectInspired/runs/stage08_training/attention384_landmarks_v2_01/evaluate/model_attention.onnx)：SHA-256 `5d0497f8f90c3b0d3287dc536c4c4ecde19326156122b22fab1375ddf4f30086`。
- [完整新旧损失对比](../visual_simulator_docker/CodeInsectInspired/runs/stage08_training/attention384_landmarks_v2_01/analysis/comparison_losses.png)、[SVG](../visual_simulator_docker/CodeInsectInspired/runs/stage08_training/attention384_landmarks_v2_01/analysis/comparison_losses.svg)、[完整数值对比](../visual_simulator_docker/CodeInsectInspired/runs/stage08_training/attention384_landmarks_v2_01/analysis/comparison.json)。
- [最终评估](../visual_simulator_docker/CodeInsectInspired/runs/stage08_training/attention384_landmarks_v2_01/evaluate/evaluation.json)、[测试世界系向量图](../visual_simulator_docker/CodeInsectInspired/runs/stage08_training/attention384_landmarks_v2_01/evaluate/plots/evaluation_test.png)。
- [交付记录](../visual_simulator_docker/CodeInsectInspired/runs/stage08_training/attention384_landmarks_v2_01/result.json)：150 轮、19,950 次正式更新及 36 次工程更新；完整轮检查点与优化器/RNG 位于 `train/checkpoints/`。

本轮采集指纹为 `b32a7896835e64f348a46144bceeac12f94bc484383f8d2d96cee0d1b17a392d`，数据准备指纹为 `d58c54447034f009724d1c3bf2130cdd6b8d80872a0beaa8a6d2639f9c34ed02`，训练指纹为 `4504f8dc0f00f6d0307edce0ab8303f01b82133b42bfd756a20dbb8e52a53d81`。

只读复查（从 `SIM/`，无需启动 Isaac）：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 run_downward_training.py \
  --output runs/stage08_training/attention384_landmarks_v2_01 --verify-only
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_stage06_data.py --verify-only \
  --output runs/stage08_data/position_groups_01
```

加载本轮 `.pth` 时使用 `train/effective_config.json` 或显式 `network_variant="nadir384x384"`，保持原始完整 RGB 输入约定；不能只给旧模型配置更换权重路径。
