# 第七阶段：独立下视训练与模型交付

日期：2026-10-27。**阶段 G 的训练与交付流程已完成：原网络从头训练 150 轮、19,950 次更新，最终权重重载和 ONNX 检查通过。** 模型存在明显过拟合：训练集平均方向误差 7.01°，验证/测试均约 91.7°。工程检查通过不代表导航精度合格；尚未启动阶段 H 闭环归巢。

## 固定的实验设置

[训练协议](../visual_simulator_docker/CodeInsectInspired/config/experiments/stage07_training_protocol.json)沿用已确认设置：

| 项目 | 本次设置 |
| --- | --- |
| 相机与场景 | 10 m AGL，30° 下视相机，完整 1024×1024 RGB；400×400 m 几何路标场景 |
| 数据 | 阶段 E 四相机采集；147 位置，每位置 8 个真实航向，共 1176 图 |
| 固定划分 | 阶段 F 按位置分组：训练 1064、验证 48、测试 64 图；无额外增强 |
| 网络 | 原 attention，10,820 参数；保留内部 H192×W1800 resize |
| 初始化与损失 | seed 10，原 Xavier 初始化，MSELoss |
| 优化器 | AdamW，lr 0.0005，betas (0.9, 0.999)，eps 1e-8，weight_decay 0.01 |
| 训练预算 | batch 8，150 epochs，每轮 133 批，共 19,950 次正式更新 |
| 读取与计算 | num_workers 4，PyTorch 线程 2，float32；记录实际 CUDA/cuDNN 默认设置 |
| 模型选择 | 第 150 轮最终权重；不使用测试集选择模型 |

无学习率调度、混合精度、梯度裁剪或提前停止。短训练的模型单独保存，正式训练重新初始化。原学习路径长约 169.5 m，最远位置距家约 11.0 m；400 m 场景面积不等于训练覆盖范围。当前没有空间隔离带，仅运行 seed 10 一次；同一位置八个航向存在相关性，不能把 64 张测试图当作 64 个独立空间位置，也不能据此声称学习区外泛化或大范围归巢成功。

## 修改一：隔离训练生命周期

新增 [run_downward_training.py](../visual_simulator_docker/CodeInsectInspired/run_downward_training.py)，将一次运行拆成 `probe → smoke → train → evaluate` 四个独立子进程。`train.py` 的下视分支进入新训练实现，全景分支保留。

**作用与结果：** 离线训练不加载 Isaac/Kit，不保留渲染场景。每个阶段有独立日志、状态和产物哈希；运行锁防止同一目录被同时训练。运行清单绑定阶段 F 数据、训练协议、源码和软件版本，配置变化需要新运行目录。

## 修改二：检查内存与实际更新

[downward_training.py](../visual_simulator_docker/CodeInsectInspired/insect_utils/downward_training.py)实现原网络训练、确定性验证和最终离线评估；[training_state.py](../visual_simulator_docker/CodeInsectInspired/insect_utils/training_state.py)实现资源采样、指标和检查点文件校验。

先用相同的 8 个训练位置、64 图分别试运行 0/2/4 个读取进程，每次 8 次更新。三次样本标签顺序一致，均通过有限损失/梯度检查。

| 读取进程 | 含进程启动的耗时 | 进程及子进程峰值 PSS | 最低主机可用内存 |
| ---: | ---: | ---: | ---: |
| 0 | 7.86 s | 1.19 GiB | 8.17 GiB |
| 2 | 1.32 s | 1.39 GiB | 7.84 GiB |
| 4 | 0.96 s | 1.35 GiB | 7.93 GiB |

因此保留原 `num_workers=4`。首轮包含冷启动，表中耗时不用于严格速度比较。资源每 0.25 s 采样，PSS 用于估计进程共享内存后的实际占用；RSS 相加可能重复计算共享页面。训练另有最低 1 GiB 主机可用内存检查，触发时保留最后完整轮的检查点。CUDA 峰值是 PyTorch 分配器统计，不等于整张显卡占用。

## 修改三：可靠保存与恢复

每轮结束保存不可变的 `checkpoints/epoch_XXXX_<id>.pt`，包含模型、AdamW 状态、累计步数、训练历史、Python/NumPy/PyTorch/CUDA 随机状态，以及训练/验证 DataLoader 的生成器状态。写入完成后原子更新 `checkpoint_latest.json`，用指纹和 SHA-256 校验恢复来源。

**作用与结果：** 中断后从最后完整轮恢复，未提交的半轮重新执行，不把优化器状态或读取顺序重置。CPU 含随机输入与 Dropout 的恢复测试逐元素一致；GPU 测试验证 CUDA 随机状态与优化器状态恢复。每轮记录训练/验证 MSE、累计更新、耗时、最大梯度范数与 CUDA 峰值；主机内存峰值在整个训练运行中汇总。

## 短训练验收

独立 smoke 使用 4 个训练位置、32 图，3 轮共 **12 次更新**；每次梯度有效。结束时训练分量 MSE 为 23.374114 m²，验证为 12.435434 m²。它仅检查训练链路，不用于判断正式模型精度。

权重重载前后预测最大差为 **0**；ONNX opset 17 图检查通过，动态 batch 的两张原始尺寸图经 `onnx.reference.ReferenceEvaluator` 计算，与 CPU PyTorch 最大差 **4.77e-7**，低于 `atol=1e-5, rtol=1e-4`。未安装或下载 ONNX Runtime，也未验证其硬件执行后端。

probe 的 24 次与 smoke 的 12 次，共 **36 次工程检查更新**，另行记录，不计入正式 19,950 次。smoke 与正式模型均从相同 seed 的全新权重开始，不继承 smoke 训练结果。

## 正式结果：能拟合训练集，未见位置预测较差

150 轮均完成，19,950 次更新均有非零且有限的梯度。每轮训练用时合计约 **41.83 分钟**，不含前后处理、工程探测和最终评估。初始化验证分量 MSE 为 12.474172 m²；最终验证为 29.284454 m²，训练过程中没有取得持续的验证改善。

下表均为**第 150 轮固定权重重新前向**的指标；训练集评估 MSE 不同于最后一轮更新过程中平均损失 0.147629 m²。

| 集合 | 位置/图像 | 分量 MSE（m²） | 向量误差均值（m） | 距离误差均值（m） | 最短角误差均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 训练 | 133 / 1064 | 0.157106 | 0.466 | 0.289 | 7.01° |
| 验证 | 6 / 48 | 29.284454 | 6.807 | 2.368 | 91.71° |
| 测试 | 8 / 64 | 55.824698 | 9.619 | 3.596 | 91.68° |

| 集合 | 向量误差 p50 / p90 / p95（m） | 距离误差 p50 / p90 / p95（m） | 方向误差 p50 / p90 / p95 |
| --- | --- | --- | --- |
| 训练 | 0.408 / 0.846 / 1.033 | 0.236 / 0.605 / 0.768 | 2.83° / 15.46° / 28.50° |
| 验证 | 6.670 / 12.131 / 13.770 | 2.027 / 4.476 / 6.691 | 76.91° / 158.79° / 171.61° |
| 测试 | 9.499 / 14.918 / 17.562 | 3.315 / 6.825 / 7.332 | 95.54° / 165.66° / 173.04° |

向量分量 MSE 是两分量平方误差的均值，单位 m²；向量误差为预测与真值向量之差的长度，距离误差为两向量模长之差的绝对值。方向使用最短角误差，零目标或零预测不计方向；本次三个集合均无零目标/零预测，因此方向样本排除数均为 0。

额外保存[恒零向量参考](../visual_simulator_docker/CodeInsectInspired/runs/stage07_training/attention_8h_01/analysis/zero_vector_reference.json)：训练/验证/测试分量 MSE 分别为 20.097902 / 12.353389 / 35.694288 m²。最终模型验证与测试 MSE 都高于该参考。恒零向量本身没有有效方向，只用于展示误差尺度，不是可用导航策略。

**判断：** 模型学会了拟合训练图，未在当前未见位置上表现出有效的方向泛化。[完整损失曲线](../visual_simulator_docker/CodeInsectInspired/runs/stage07_training/attention_8h_01/analysis/loss_curves.png)与[测试世界系向量图](../visual_simulator_docker/CodeInsectInspired/runs/stage07_training/attention_8h_01/evaluate/plots/evaluation_test.png)保留该失败证据。目前不能确定窄视场下地物可辨识性、位置/航向覆盖和原网络非等比 resize 各自的影响，需要独立诊断或受控对照，不能仅凭损失曲线归因。本阶段没有自动更换网络、重新划分数据或调参。

验证位置距家约 2.44–6.57 m，测试位置约 4.13–10.11 m；两者覆盖不同，应分别解读。二者仍位于学习区域，且没有空间隔离带。空间留出、学习区外和闭环成功率留在 H/I。

## 资源与交付结果

正式训练进程及读取子进程的采样峰值 PSS 约 **2.81 GiB**，汇总 RSS 约 8.06 GiB，最低主机可用内存约 **5.41 GiB**；未触发内存保护。PyTorch CUDA 峰值分配/保留约 **190.3 / 216 MiB**。系统进程和显示占用不计入该分配器指标。

实际环境为 Python 3.10.14、PyTorch 2.4.0+cu118、CUDA 11.8、cuDNN 90100、RTX 4060。保留运行时默认 `cudnn_deterministic=false`、`cudnn_allow_tf32=true`、`matmul_allow_tf32=false`；记录恢复测试结果不等于承诺跨硬件逐位一致。

最终 `.pth` 重载预测最大差为 **0**。ONNX 两图参考计算最大差为 **1.22e-5**，通过组合容差 `abs(actual-expected) <= 1e-5 + 1e-4*abs(expected)`；输入为 float32、`[batch,3,1024,1024]`、完整 RGB / 255，输出为米制机体系 `[forward,left]` 归巢向量。该检查使用 ONNX 参考解释器，不代表已验证 ONNX Runtime 部署。

运行目录为 `SIM/runs/stage07_training/attention_8h_01/`，其中 `SIM=visual_simulator_docker/CodeInsectInspired/`：

- [正式模型](../visual_simulator_docker/CodeInsectInspired/runs/stage07_training/attention_8h_01/train/model_attention.pth)：第 150 轮权重；SHA-256 `7bd320860e955fef4048f311cb6582d244f51f67a1e5f8dd24f7f9cf285c7298`。
- [ONNX](../visual_simulator_docker/CodeInsectInspired/runs/stage07_training/attention_8h_01/evaluate/model_attention.onnx)：SHA-256 `fd76946f9770a6671d14fc9cbe606abb4e0fd2e2aeee655c71fd83d17e6c94fb`。
- `run_manifest.json`：冻结数据、参数、源码及依赖版本，指纹 `e5b4b530ec6c1fc74dc4663c89a8b27706c7bf7d9152bce6beb7113ed5b4fea0`。
- `logs/train_01.log`、[training_history.json](../visual_simulator_docker/CodeInsectInspired/runs/stage07_training/attention_8h_01/train/training_history.json)：每轮及每 25 次更新的记录。
- [checkpoint_latest.json](../visual_simulator_docker/CodeInsectInspired/runs/stage07_training/attention_8h_01/train/checkpoint_latest.json)：指向第 150 轮完整模型、优化器和 RNG 检查点。
- [evaluation.json](../visual_simulator_docker/CodeInsectInspired/runs/stage07_training/attention_8h_01/evaluate/evaluation.json)：三组指标；同目录 `predictions_*.npz` 保存逐样本预测与编号。
- [result.json](../visual_simulator_docker/CodeInsectInspired/runs/stage07_training/attention_8h_01/result.json)、`completion.json`：运行交付与测试、数据保留核对记录。

原始照片和阶段 F 划分保持原样；已有全景权重、理论分析的 `model.pt` / `model_10.pt` 不由本阶段覆盖。

## 验证与复查

容器内 **130 项测试全部通过**，包括 GPU 恢复检查，见[容器验收日志](../visual_simulator_docker/CodeInsectInspired/runs/stage07_smoke_01.log)。主机通过 119 项，因缺少 PyTorch 跳过 11 项，这 11 项已在容器通过。41 个实验 Python 文件的 Black 检查与 Flake8 通过。旧基线协议检查通过，原 `simple_network.py`、阶段 F 核心预处理及划分文件保持原样；训练后 G 产物校验与 F 数据完整性复核均通过。

从 `SIM/` 查看状态，不加载 PyTorch 或 Isaac：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 run_downward_training.py \
  --output runs/stage07_training/attention_8h_01 --stage status
cat runs/stage07_training/attention_8h_01/state/train.json
```

完成后校验所有阶段产物与主模型哈希：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 run_downward_training.py \
  --output runs/stage07_training/attention_8h_01 --verify-only
```

容器内的训练/恢复入口如下；同一目录的训练锁会拒绝并发启动：

```bash
../python.sh run_downward_training.py \
  --output runs/stage07_training/attention_8h_01 --stage all --resume
```

下一阶段 H 接入下视闭环控制时，应先用真值 oracle 验证相机、坐标与移动方向，再用冻结模型作诊断。当前模型精度不足是明确的后续问题；若改变输入尺寸、采样方案、网络或训练参数，需要先与用户确认并建立独立实验。本阶段未运行闭环，本报告没有归巢成功率结果。
