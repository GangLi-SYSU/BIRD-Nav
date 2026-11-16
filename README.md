# BIRD-Nav

BIRD-Nav 研究大尺度外出后的两阶段归巢：先利用理想地磁航向辅助路径积分返回起点附近，
再通过下视相机和神经网络完成局部视觉归巢。项目提供批量实验、训练流程及可手动外出的 Isaac Sim 交互演示。

## 最终方案

第一阶段通过里程计累计位移，地磁提供无噪声绝对航向；控制器返回内部估计的起点后切入视觉导航。
第二阶段根据下视 RGB 图像预测机体坐标系中的归巢向量，按预测距离调整移动步长。

| 设置 | 数值或方法 |
| --- | --- |
| 仿真环境 | Isaac Sim 4.2.0；真实尺寸房屋、树木、水池、道路与草地 |
| 飞行高度与相机 | 离地 40 m，垂直下视，水平与垂直视场角均为 30° |
| 图像与网络 | 原始 1024×1024 RGB，网络输入 384×384，注意力网络共 10,628 个参数 |
| 学习轨迹 | 半径 20 m 内约五圈螺旋，147 个位置，每位置八个航向，共 1,176 张图像 |
| 数据划分 | 按位置分组；训练 1,064 张、验证 48 张、预留测试 64 张 |
| 训练设置 | 随机旋转图像及向量标签、固定圆形掩膜；AdamW，学习率 0.0005，批大小 8，600 轮 |
| 参考模型 | 按验证集误差选出的第 484 轮；导出模型包含预处理及米制输出换算 |
| 视觉控制 | 步长为 `min(0.1 + 0.4 × 预测距离 / 3, 2.5)` m，最多 50 步 |
| 到达判定 | 标准测试在动作完成后距起点不超过 2 m 时停止 |

## 参考实验结果

### 视觉归巢

每个初始半径使用 16 个方位和四个初始航向，共 64 次测试。
成功条件为首次进入 2 m 范围；平均停止距离包含成功及失败的全部测试。

| 初始半径 | 成功次数 | 成功率 | 平均停止距离 |
| --- | ---: | ---: | ---: |
| 10 m | 64 / 64 | 100.00% | 1.840 m |
| 15 m | 59 / 64 | 92.19% | 2.351 m |
| 20 m | 54 / 64 | 84.38% | 6.773 m |

在独立的 **15 m 起点、固定执行 50 步**实验中，进入 2 m 后仍继续导航：
第 50 步有 63 / 64 次位于 2 m 内，平均距离 0.839 m、中位数 0.618 m。
该结果衡量第 50 步的位置，与首次进入 2 m 即停止的成功率分别统计。

### 地磁辅助路径积分

名义外出终点距家 2,000 m，折线外出航程 4,000 m、名义返程 2,000 m，
距离噪声系数为 0.1 m/√m。两组各 1,000 次配对实验，在内部估计到家后停止，
按真实终点是否位于起点 15 m 内评分。

| 方法 | 终点位于 15 m 内 | 比例 | 平均终点误差 |
| --- | ---: | ---: | ---: |
| 理想地磁辅助 | 977 / 1,000 | 97.7% | 6.677 m |
| 无地磁，累积陀螺仪航向漂移 | 2 / 1,000 | 0.2% | 597.631 m |

完整数值见[参考指标](docs/results/reference_metrics.json)。上述结果来自单一场景/训练种子的运动学仿真，
不代表实飞成功率。视觉到达判定使用仿真真值；15 m 指标只衡量第一阶段进入局部归巢范围的比例，
两类独立测试没有合并为端到端成功率。

## 代码结构

| 目录 | 内容 |
| --- | --- |
| `visual_simulator_docker/` | Isaac 环境构建、场景准备、训练、测试与演示启动脚本 |
| `visual_simulator_docker/CodeInsectInspired/` | 图像采集、网络、导航控制器、配置和自动化测试 |
| `path_integration_noise_simulation/magnetic_guidance/` | 可调外出距离与噪声的地磁路径积分实验 |
| `docs/REPRODUCTION_RESOURCES.md` | 素材下载、校验、场景重建及完整复现步骤 |
| `code_theoretical_simulation_analysis/` | 上游理论分析代码 |
| `robot_network_training/`、`robot_onboard/` | 上游机器人训练与机载程序 |

## 复现流程

### 1. 获取代码与环境资源

```bash
git clone https://github.com/GangLi-SYSU/BIRD-Nav.git
cd BIRD-Nav
```

视觉仿真需要 Linux、NVIDIA RTX GPU、兼容驱动、Docker 和 NVIDIA Container Toolkit。
先按[复现指南](docs/REPRODUCTION_RESOURCES.md)下载树木/地面素材和四种房屋，
再构建 Isaac Sim 4.2.0 容器并生成场景。专用 Docker 存储位于仓库所在硬盘，首次需要至少 42 GiB 可用空间。

参考网络和演示用鸟模型随代码提供。大型素材从指南列出的来源下载；场景与训练图像按命令生成。
预生成场景和自采图像尚无公开下载包，克隆代码后需要完成对应生成步骤。

### 2. 重新采集、训练与测试

环境和场景就绪后，在仓库根目录运行：

```bash
# 检查采样计划。
bash visual_simulator_docker/run_compact_spiral_on_disk.sh \
  --run-dir runs/reproduce_compact20_01 --plan-only

# 按同一计划采图、训练 600 轮，并测试 10 m / 20 m 起点。
bash visual_simulator_docker/run_compact_spiral_on_disk.sh \
  --run-dir runs/reproduce_compact20_01

# 查看运行日志。
bash visual_simulator_docker/disk_docker.sh docker logs -f bee-nav-stage23-compact20
```

每次独立实验使用新的输出目录。结果保存在
`visual_simulator_docker/CodeInsectInspired/runs/reproduce_compact20_01/`，
包含采集清单、检查点、损失曲线、导航轨迹及结果报告。默认流程测试 10 m、20 m；
上表的 15 m 和固定 50 步结果属于同一参考模型的独立评估。
重新训练的最佳轮次及指标可能随 GPU、依赖或渲染版本变化。

### 3. 启动两阶段交互演示

在桌面终端、仓库根目录运行：

```bash
bash visual_simulator_docker/run_interactive_homing_demo.sh
```

演示默认加载随仓库提供的参考网络。主界面为第三人称，另有俯视总览和下视相机子窗口。

| 按键 | 操作 |
| --- | --- |
| `T` | 起飞到 40 m |
| `W / A / S / D` | 手动外出，方向相对机头 |
| `Q / E` | 调整机头方向 |
| `H` | 开始路径积分与视觉两阶段归巢 |
| 空格 | 暂停或恢复 |
| `R` | 重置飞行 |
| `V` | 恢复子窗口 |

归巢结束后自动降落，标出起点、降落点及水平距离，再显示两阶段的轨迹结果。
详细操作与输出格式见[演示说明](docs/INTERACTIVE_TWO_STAGE_DEMO.md)。

### 4. 运行纯 CPU 路径积分实验

该实验只需要 NumPy 和 Matplotlib。在仓库根目录创建 Python 环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy matplotlib
cd path_integration_noise_simulation/magnetic_guidance
../../.venv/bin/python run_experiment.py --config config.json --output runs/magnetic_01
```

默认配置测试多种距离/噪声组合，每组 50 次。2,000 m、0.1 噪声、1,000 次实验的命令及配置说明见
[路径积分说明](path_integration_noise_simulation/magnetic_guidance/README.md)。
Docker 磁盘、下载缓存、生成数据及实验输出由 `.gitignore` 排除，不随代码上传。

## 来源与许可

本项目基于 [TU Delft Bee-Nav](https://github.com/tudelft/Bee-Nav) 及论文
[*Efficient robot navigation inspired by honeybee learning flights*](https://www.nature.com/articles/s41586-026-10461-3)。
保留原代码的 [MIT 许可](LICENSE)与版权声明；第三方场景素材和鸟模型分别遵循各自附带许可。
