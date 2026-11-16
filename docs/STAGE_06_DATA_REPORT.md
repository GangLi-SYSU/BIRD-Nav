# 第六阶段：按位置分组与统一图像预处理

日期：2026-10-27。**阶段 F 已完成：1176 张实际下视图全部通过读取和像素检查，固定划分已保存，验证/测试重复前向结果一致。** 本阶段使用未训练网络检查数据接口，没有执行优化器更新，没有启动 Isaac 或闭环归巢。

## 已确认的划分与预算

用户选择“按位置分组，保留原比例”。[第六阶段协议](../visual_simulator_docker/CodeInsectInspired/config/experiments/stage06_data_protocol.json)冻结 seed 10、原 `split_ratio=0.95` 的取整规则、真实八航向和关闭额外增强。

| 集合 | 位置数 | 图像数 |
| --- | ---: | ---: |
| 训练 | 133 | 1064 |
| 验证 | 6 | 48 |
| 测试 | 8 | 64 |
| 合计 | 147 | 1176 |

先用独立 CPU `torch.Generator().manual_seed(10)` 抽取 147 个位置的排列，再按原规则划分：`pool=int(.95×147)=139`，验证为 `int(.05×139)=6`，训练为 133，测试为 8。每个位置的八张图全部进入同一集合；索引完整覆盖 1176 张，无重复、无丢失，同一位置跨集合数量为 **0**。再次读取时使用固定索引，不重新随机划分。

这是学习区域内的位置分组检验，未设置空间隔离带。[实际划分图](../visual_simulator_docker/CodeInsectInspired/runs/stage06_data/position_groups_01/inspection/split_map.png)显示三组位置分布。验证/测试位置到最近训练位置的最短距离分别约 0.195 m、0.638 m；相邻图像仍可能看到重叠地面，因此这组结果不能作为空间留出泛化结论。

batch 8、150 epochs 等原参数保留。当前固定划分对应每轮 133 个训练批次、150 轮 **19,950 次计划更新**；本阶段实际更新数为 **0**。

## 修改一：离线与在线使用同一预处理

**作用：** 确保训练照片与在线观测采用相同像素约定。

[rgb_preprocessing.py](../visual_simulator_docker/CodeInsectInspired/insect_utils/rgb_preprocessing.py)处理完整 uint8 RGB 或 RGBA 的前三通道，输出连续 float32 CHW 张量，像素除以 255。不混合 alpha、不裁剪、不提前 resize、不循环移位。

[virtual_homing_ANN.py](../visual_simulator_docker/CodeInsectInspired/virtual_homing_ANN.py)的下视取图接口已调用共用函数；Isaac 导入延迟到使用仿真时，离线检查不加载 Kit。1176 张实际 PNG 的结果与构造 RGBA 后的在线处理、原 `ToTensor` 逐元素一致。实际 `get_image` 分支另用回传这些像素的模拟相机验证；本阶段没有重新渲染实景。

## 修改二：独立数据集与固定评测

**作用：** 保持八航向分组，阻止旧全景增强进入下视数据流程。

[downward_data.py](../visual_simulator_docker/CodeInsectInspired/insect_utils/downward_data.py)提供训练、验证、测试三个独立 Dataset。每张 PNG 读取时检查哈希、尺寸和类型，再返回完整 RGB 与原米制机体系标签。

[data_loader.py](../visual_simulator_docker/CodeInsectInspired/data_loader.py)、[train.py](../visual_simulator_docker/CodeInsectInspired/train.py)和[train_utils.py](../visual_simulator_docker/CodeInsectInspired/insect_utils/train_utils.py)的下视分支已接通固定数据集、检查图和评测接口。下视模式显式拒绝全景 wrap 或额外旋转增强，检查器也不再偷偷旋转图像；全景分支保留。验证与测试顺序固定，训练仅打乱样本读取顺序。

已验证：验证集和测试集各重复两次，预测与标签逐元素相同；0 与 2 个读取进程得到相同验证数据。配置中的原 `num_workers=4` 保留，正式训练的资源检查留在 G。

## 修改三：保留网络，展示实际拉伸

**作用：** 落实“网络结构沿用原代码”，并让输入形变可见。

原 `simple_network.py` 未修改，attention 参数量仍为 **10,820**。前向钩子实测第一层收到 `[8,3,192,1800]`，与原 bilinear、`align_corners=False` 的内部 resize 一致；输出为有限的 `[8,2]`。检查前后权重哈希相同。

[resize 前后图](../visual_simulator_docker/CodeInsectInspired/runs/stage06_data/position_groups_01/inspection/network_resize.png)展示 1024 方图被原网络非等比拉伸的结果。相机物理视场仍为 30°×30°；这里记录网络输入变形，尚未评价其对导航精度的影响。

[方向检查图](../visual_simulator_docker/CodeInsectInspired/runs/stage06_data/position_groups_01/inspection/label_directions.png)以中心箭头表示机体系归巢方向：机体 +X 对应图像上，+Y 对应图像左。箭头只表示方向，家可以在视场外；零向量不画方向箭头。[世界系标签图](../visual_simulator_docker/CodeInsectInspired/runs/stage06_data/position_groups_01/inspection/world_home_vectors.png)将标签转回世界系，核对它们指向配置的家。

## 文件与数据来源

新产物位于 `SIM/runs/stage06_data/position_groups_01/`，其中 `SIM=visual_simulator_docker/CodeInsectInspired/`：

- [source_reference.json](../visual_simulator_docker/CodeInsectInspired/runs/stage06_data/position_groups_01/source_reference.json)：阶段 E 四相机数据的原采集指纹与哈希。
- [splits.json](../visual_simulator_docker/CodeInsectInspired/runs/stage06_data/position_groups_01/splits.json)：固定位置分组和完整样本索引。
- [data_config.json](../visual_simulator_docker/CodeInsectInspired/runs/stage06_data/position_groups_01/data_config.json)：下视数据入口与保留的训练参数。
- `records.json`、[preparation_manifest.json](../visual_simulator_docker/CodeInsectInspired/runs/stage06_data/position_groups_01/preparation_manifest.json)：绑定样本、标签、分组、原始图像与预处理代码版本。
- [stage06_report.json](../visual_simulator_docker/CodeInsectInspired/runs/stage06_data/position_groups_01/stage06_report.json)：实际检查结果和诊断图哈希。

[data_preparation.py](../visual_simulator_docker/CodeInsectInspired/insect_utils/data_preparation.py)按采集时冻结的原指纹验证历史数据，再为本阶段单独记录准备指纹。代码进入新阶段不要求重拍照片，也不会跳过旧数据的完成状态或文件哈希检查；阶段 E 的照片、CSV 和清单保持原样。

## 验证与复查命令

容器 Python 3.10 中 **117 项测试全部通过**；主机 Python 3.8 中通过 110 项，因缺少 PyTorch 跳过 7 项运行测试，这 7 项均在容器通过。Black、Flake8、旧基线协议和 `git diff --check` 通过。[最终验收日志](../visual_simulator_docker/CodeInsectInspired/runs/stage06_validation_02.log)包含测试与实际数据检查通过记录。

[阶段完成记录](../visual_simulator_docker/CodeInsectInspired/runs/stage06_data/position_groups_01/completion.json)汇总固定划分、协议与源码哈希、测试结果，以及 422 个既有文件的修改范围和权重保留核对结果。

从 `SIM/` 运行，只读核对已有数据、完成状态和哈希，不需要 PyTorch，也不启动仿真：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_stage06_data.py --verify-only \
  --output runs/stage06_data/position_groups_01
```

需要重新执行完整准备与前向验收时，在现有容器的 `SIM/` 中使用尚不存在的新目录：

```bash
../python.sh tools/check_stage06_data.py \
  --output runs/stage06_data/position_groups_02
```

下一阶段 G 接入独立训练运行，先做少量 epochs 的训练检查，再按冻结参数推进正式训练。H 的闭环归巢仍待实施；本报告没有训练精度或归巢成功率结果。
