# 第二阶段：统一配置、独立进程与可靠续跑

日期：2026-10-22。对应总清单 **B1–B6**，已完成。

本阶段把实验配置和运行管理接入了代码。**实际下视相机尚未替换**：新入口可以解析、预览和保存 10 m / HFOV 30°目标配置；下视采集与归巢执行会在启动 Isaac 前明确提示等待 C/D/E 阶段，避免参数写成下视、实际生成全景图。

## B1：用一个文件描述实验

**作用：** 高度、家位置、相机、学习路径、训练与测试不再由多个入口分别指定。

**修改：** 新增 [downward_h10_hfov30.json](../visual_simulator_docker/CodeInsectInspired/config/experiments/downward_h10_hfov30.json)，以及纯标准库配置解析模块 [experiment_config.py](../visual_simulator_docker/CodeInsectInspired/insect_utils/experiment_config.py)。未知字段、非有限数值、非法路径和不支持的组合会被拒绝，拼错参数不会被静默忽略。

**结果：** 目标配置明确为 10 m 相机离地高度、暂定水平视场角 30°、`1024×1024` 原图，以及原 attention / MSE / AdamW、学习率 `0.0005`、batch 8、150 epochs、seed 10。AdamW 的 betas、eps、weight_decay、amsgrad 也显式保存。下视模式禁用不成立的全景 wrap 增强；新的航向增强仍待后续数据阶段。

FOV 定义仍标记为暂定，不等于已确认真实传感器参数。场景路径 `maps/env_downward_flat.usd` 是阶段 C 的待建场景，本阶段没有创建一个同名空场景冒充已完成。

另有 [stage02_pipeline_smoke.json](../visual_simulator_docker/CodeInsectInspired/config/experiments/stage02_pipeline_smoke.json)，明确标记 `engineering_smoke`。它仅使用人工测试图验证流程，不能当作研究数据，也不能进入闭环归巢评测。

## B2：保存实际生效的参数

**作用：** 防止 JSON 写了一种设置，入口或辅助函数又暗中使用另一种设置。

**修改：** 新入口 [run_downward_homing.py](../visual_simulator_docker/CodeInsectInspired/run_downward_homing.py) 从统一配置生成采集、训练和归巢参数；准备运行时保存 `resolved_config.json`，每个实际执行阶段再保存自己的 `effective_config.json`。

**结果：** 相机图像尺寸、采集位置、模型输出路径、worker 数和测试参数都有可复查记录。新流程给旧模块传入明确参数；旧入口省略这些参数时仍使用原默认值。第一阶段冻结的旧参数检查依然通过，网络结构文件未修改。

本次保留兼容性的原文件修改包括：

- `render_dataset.py`：可接收显式 headless 设置。
- `train.py`：可接收显式 AdamW 参数。
- `data_loader.py`：可指定 DataLoader worker 数及划分索引的保存位置。
- `insect_utils/train_utils.py`：可指定模型目录/名称，记录训练过程并原子替换新流程的模型产物。
- `virtual_homing_ANN.py`：可接收同源的 render 配置和动作上限。

这些改动没有把相机投影改为下视，也没有修改原控制器的方向符号、网络层或正式训练轮数。

## B3：区分地面上的家与相机悬停点

**作用：** “家在地面上”与“相机在家上方飞行”使用不同的三维坐标，不能把二者的高度混用。

**修改：** 输入地面家位置 XY、平地 Z 和相机 AGL，由解析模块统一推导物理家位置与相机悬停位置。

**结果：** 当前配置得到地面家位置 `[0,0,0]`、相机在家上方的位置 `[0,0,10]`。采集 `home_position` 与归巢 `virtual_home_point` 都由后者生成，二维标签仍表示水平归巢向量。测试也覆盖了非零家位置和 `ground_z=2 m` 的情况，采集与归巢都得到相机 `z=12 m`。

这仍是平地模型，不代表已经实现地形跟随或实测 AGL 控制；三维场景检查属于阶段 C。

## B4：拆分采集、训练和评测进程

**作用：** 训练时释放 Isaac 场景、渲染和界面资源，并让各阶段能够单独重跑、定位错误。

**修改：** 控制入口本身只使用标准库；通过独立 Python 子进程执行 `capture`、`train`、`evaluate`，具体适配代码在 [experiment_stages.py](../visual_simulator_docker/CodeInsectInspired/insect_utils/experiment_stages.py)。只有实际 Isaac 采集或评测分支才导入并创建 `SimulationApp`。阶段产物在 Kit 关闭前提交，避免第一阶段遇到的退出时报告未保存问题。

**结果：** 实际短训练验收中，数据生成和训练的进程不同；训练进程加载的 `omni` 模块列表为空。训练使用原 attention 模型并进行了真实梯度更新。

每个独立阶段重新初始化 seed 10。因此新流程与旧版“同一进程连续跑多个环境”的随机数消耗顺序可能不同；后续对照应使用同一套新流程，不能宣称与旧串行运行逐位一致。

## B5：隔离每次运行的产物

**作用：** 避免不同高度、相机和训练配置共用旧 `forest/`、`models/`、`results/`，或覆盖上次结果。

**修改与结果：** 每次运行使用独立 run ID，目录结构为：

```text
runs/<experiment_name>/<run_id>/
├── resolved_config.json
├── run_manifest.json
├── capture/
│   ├── effective_config.json
│   ├── dataset_navigation.csv
│   ├── dataset_manifest.json
│   └── Replicator/rgb/
├── train/
│   ├── effective_config.json
│   ├── model_attention.pth
│   ├── model_attention.onnx
│   ├── indices.pkl
│   ├── training_history.json
│   └── summary.json
├── evaluate/                    # 实际评测后才存在
├── state/                       # 阶段状态、输入和产物校验值
├── logs/                        # 按阶段、尝试次数保存日志
└── attempts/                    # 失败重试时保留旧半成品
```

索引放在 `train/`，训练不会修改已经完成并校验过的采集目录。运行产物已加入 Git 和 Docker 构建排除规则；仍存放在当前硬盘中。

## B6：通过状态与指纹续跑

**作用：** 目录存在不能证明数据完整，更不能证明它属于当前相机、配置和代码。

**修改：** 新增 [experiment_state.py](../visual_simulator_docker/CodeInsectInspired/insect_utils/experiment_state.py)，提供原子状态写入、文件校验与运行锁。

**结果：**

- 运行指纹包括规范化配置、推导参数、源代码、入口 USD、冻结协议和 Python/主要库版本。
- 采集后检查数量、可解码性、尺寸、标签代数一致性和高度，生成数据清单。实际相机帧与真实位姿的同步检查仍待 D/E。
- 完成标记同时保存产物内容与依赖指纹；缺文件、内容被改动或输入不一致时，拒绝复用。
- `--resume` 对已完成阶段进行校验后复用；失败阶段先归档半成品再重新执行，保留旧尝试。
- 归档与状态写入之间再次中断，也不会覆盖已有归档。运行锁由工作进程继承，防止父进程退出后另一个进程误接管仍在执行的任务。
- 工作结束前重新核对配置、代码和环境；中途修改代码的运行不能被标为可复现的完成结果。

这里实现的是**阶段级续跑**：失败训练会从头执行该阶段。精确恢复到某个 epoch、优化器状态与 RNG 状态仍属于阶段 G。

当前场景指纹覆盖入口 USD；引用模型、纹理的完整依赖清单要在 C 阶段补齐，不能据此宣称已验证所有外部资产。

## 验证结果

宿主 Python 3.8 与容器 Python 3.10 中，累计 **29 项测试通过**；第一阶段旧参数检查仍通过。新增 Python 文件通过 Black 和 Flake8 检查。

最终流程验收使用 **24 张人工生成图像、1 epoch、3 次梯度更新**，不是正式视觉归巢训练：

- 保持 10,820 个网络参数，权重内容实际发生变化，保存后能够重新加载。
- 训练进程无 Kit，采集与训练进程隔离，CUDA 用于训练。
- 模型、ONNX、索引和训练日志只写入本次运行目录；ONNX 结构检查通过。
- 正常续跑不重复训练；故意损坏产物、改变配置时拒绝复用；人工数据评测被拒绝。
- 已恢复被故意修改的检查产物；日志中的相应 `Error:` 是负向验收的预期结果。

证据位置：

- [最终流程验收报告](../visual_simulator_docker/CodeInsectInspired/runs/stage02_pipeline_smoke/check_20261022T124815Z_67dd16/checks/integration_report.json)
- [训练进程与参数摘要](../visual_simulator_docker/CodeInsectInspired/runs/stage02_pipeline_smoke/check_20261022T124815Z_67dd16/train/summary.json)
- [实际训练记录](../visual_simulator_docker/CodeInsectInspired/runs/stage02_pipeline_smoke/check_20261022T124815Z_67dd16/train/training_history.json)
- [10 m 目标配置快照](../visual_simulator_docker/CodeInsectInspired/runs/downward_h10_hfov30/stage02_prepared_target/resolved_config.json)
- [容器最终验收日志](../visual_simulator_docker/CodeInsectInspired/runs/stage02_final_verification.log)

ONNX 本次仅验证结构，动态 batch 和 PyTorch/ONNX 数值对照仍待 G8。没有在本阶段重复启动 Isaac 场景或进行闭环导航，真实采集/评测适配仍需后续验收。

## 当前可用命令与下一步

宿主机无需启动 Isaac 即可预览配置和运行标准库测试。从 `Bee-Nav/` 根目录执行：

```bash
python3 visual_simulator_docker/CodeInsectInspired/run_downward_homing.py --stage plan
python3 -m unittest discover -s visual_simulator_docker/CodeInsectInspired/tests -p 'test_stage*.py'
```

实际准备和运行应使用同一个项目容器与 Python 环境，避免不同环境指纹混用。按 `DISK_BUILD.md` 进入容器后：

```bash
cd /isaac-sim/CodeInsectInspired
../python.sh run_downward_homing.py --stage plan
../python.sh run_downward_homing.py --stage prepare --run-id my_exp_001
../python.sh run_downward_homing.py --stage status --run-id my_exp_001
```

`prepare` 只保存配置，不采图、不训练。当前目标配置的 `capture/evaluate` 会提示相机适配尚未完成，`train` 则需要先有经过校验的采集结果。

后续 C/D/E 适配完成后，阶段命令的接口为：

```bash
../python.sh run_downward_homing.py --stage capture --run-id <run_id>
../python.sh run_downward_homing.py --stage train --run-id <run_id>
../python.sh run_downward_homing.py --stage evaluate --run-id <run_id>
# 完成阶段复用或失败阶段重试时，给相应命令增加 --resume。
```

`--stage all` 按顺序调度三个独立进程；目前对下视目标仍会先检查并报告未实现条件。每次代码或场景发生改变后使用新 run ID，不继续使用本阶段的演示运行。

下一阶段 C 将建立适合 10 m 下视观察、资源完整的平坦大场景；D 阶段再实际替换并校准相机。
