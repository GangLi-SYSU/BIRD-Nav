# 下视视觉归巢代码

本目录实现图像采集、注意力网络训练、视觉导航控制和两阶段演示。
环境安装与素材准备见[复现指南](../../docs/REPRODUCTION_RESOURCES.md)，
最终实验结果见[仓库首页](../../README.md)。

## 最终实验设置

- 相机离地 40 m、垂直下视、30° 方形视场，输出 1024×1024 RGB。
- 半径 20 m 内的螺旋轨迹包含 147 个采样位置，每位置八个航向；按位置划分训练、验证和预留测试。
- 网络输入为 384×384，10,628 个参数；训练使用随机旋转、同步向量标签及固定圆形掩膜。
- AdamW、学习率 0.0005、批大小 8、600 轮；归巢向量的归一化尺度为 44 m。
- 标准导航在首次进入 2 m 后停止，最多 50 步，按网络预测距离缩短步长。

## 模块与资源

| 路径 | 用途 |
| --- | --- |
| `run_compact_spiral_experiment.py` | 完整采集、训练与 10 m / 20 m 测试入口 |
| `generate_real_world_environment.py` | 从房屋、树木及纹理生成真实尺寸场景 |
| `insect_utils/` | 相机、数据处理、模型、控制器与轨迹绘制 |
| `tools/interactive_homing_multiview.py` | 手动外出和两阶段归巢演示 |
| `config/experiments/` | 采样、训练、导航与演示协议 |
| `assets/models/stage23_epoch484/` | 第 484 轮参考权重、导出模型及 SHA256 |
| `assets/birds/quaternius_eagle/` | 演示用鸟模型及许可 |
| `tests/` | 数据、几何、导航与演示回归测试 |

## 运行方式

以下命令在仓库根目录执行，使用 Isaac 容器自带的 Python 和依赖：

```bash
bash visual_simulator_docker/run_compact_spiral_on_disk.sh \
  --run-dir runs/reproduce_compact20_01
bash visual_simulator_docker/run_interactive_homing_demo.sh
```

采集与训练输出保存至本目录 `runs/reproduce_compact20_01/`；
演示输出保存至 `runs/interactive_demo/<时间戳>/`。
`config/experiments/demo_release.json` 指定参考模型、相机、控制器和哈希。
导出的 TorchScript 接收原始 RGB 张量，内部执行尺寸调整、圆形掩膜和米制输出换算。

## 代码规范

Python 使用四空格缩进；格式化与静态检查配置为 `pyproject.toml` 和 `.flake8`。
运行仿真需使用 Isaac Sim 自带 Python；完整命令与检查步骤以复现指南为准。
