# 第十三阶段：下视图随机旋转与固定圆形掩膜

## 目的与范围

用既有八航向照片生成连续航向的训练样本，检查是否改善航向变化下的预测。
保留原384×384网络、10,628个可训练参数及原始数据的位置分组。
现有Stage11权重和Stage12的0.5m/50步导航基线保留；新输出单独保存。

本阶段增加训练与部署接口，不重新采集场景，也不自动启动导航。
旋转不会补充新的空间位置，尤其不能替代起点附近的数据。

## 当前验收状态

2026-10-30已通过15项回归测试及RTX4060 GPU工程验收：两种模式各2轮、4次更新，
使用与正式配置相同的4个数据工作进程。TorchScript、ONNX和原始RGB部署路径均通过数值容差检查。
用户现已选择两组正式150轮训练，输出为`runs/stage13_circular_01/`。
训练完成后自动衔接[仅20m归巢对照](STAGE_14_ROTATION_R20_COMPARISON.md)，
每组64次，不运行其他起始半径。当前验收数据不代表正式训练精度或导航成功率。

- [实际训练图的增强效果](../visual_simulator_docker/CodeInsectInspired/runs/stage13_circular_acceptance_02/preview/augmentation_grid.png)
- [完整验收记录](../visual_simulator_docker/CodeInsectInspired/runs/stage13_circular_acceptance_02/IMPLEMENTATION_ACCEPTANCE.md)

## 图像与标签怎样变换

1. 读取原始1024×1024 RGB，校验原文件SHA256，按原方式双线性缩至384×384。
2. 旋转组每次读取训练图时，从`[0,360)`均匀随机选择角度α，绕图像中心逆时针旋转。
3. 归巢向量采用当前机体系：X向前、Y向左。标签同步变换为
   `v' = R(α) v`；对应虚拟机头航向为`yaw' = yaw - α`，位置和距离不变。
   例如`(10,0)`配合图片逆时针90°，变为`(0,10)`。
4. 模型统一保留中心`(191.5,191.5)`、半径190像素的圆盘，圆外RGB固定为0。
   半径略小于内切圆，为双线性插值留出边界；约77%的方形像素保留。
   不对旋转后的画面裁剪、放大、镜像填充或水平循环平移。
5. 验证、测试、导航部署不执行随机旋转，但使用同一固定掩膜。

掩膜写在`CircularAttention`预处理层中，并包含在TorchScript/ONNX导出里。
部署仍接收`[N,3,1024,1024]`、RGB/255的原始浮点图像，内部完成缩放与掩膜，
输出仍是机体系米制归巢向量。归一化尺度44只在输出端恢复一次。
后续接入导航须建立新的模型/协议冻结记录，不覆盖Stage12的固定模型。

## 对照与预算

配置：`config/experiments/stage13_circular_protocol.json`。

| 组别 | 固定圆形掩膜 | 训练随机旋转 |
|---|---|---|
| `mask_only` | 是 | 否 |
| `mask_rotation` | 是 | 是 |

用户已选择同时运行两组，以区分圆形掩膜和随机旋转增强的作用。
默认正式预算沿用上一轮：512训练位置/4096图、40验证位置/320图，
150轮、每组76,800次更新、batch=8、seed=10、AdamW学习率0.0005、weight_decay=0.01。
归巢向量除44训练，两组重新以相同参数初始化，不从冒烟测试或旧模型微调。
在线增强不增加数据集长度、不额外写出数十万张照片，也不增加优化器更新预算。

每轮验证不随机增强；仅按完整验证集米制分量MSE选最佳轮次。
所有参与组的选择完成并冻结后，才允许显式执行测试比较。
现有两个测试集已经参与开发分析，新的分数必须称为开发对比，不能称为新的盲测。

## 运行命令

在`Bee-Nav/`运行；容器和结果继续使用本硬盘。正式GPU运行前关闭Isaac预览。

```bash
# 准备独立验收目录，不启动正式训练。
bash visual_simulator_docker/run_circular_on_disk.sh prepare \
  --run-dir runs/stage13_circular_acceptance_02

# 使用真实训练图生成原图、0/22.5/45/90/137度旋转及标签箭头对照。
bash visual_simulator_docker/run_circular_on_disk.sh preview \
  --run-dir runs/stage13_circular_acceptance_02

# GPU工程验收：每组2个训练位置、2个验证位置，2轮/4次更新。
# 同时验证最佳权重选择、原始RGB部署路径、TorchScript和ONNX导出。
bash visual_simulator_docker/run_circular_on_disk.sh smoke \
  --run-dir runs/stage13_circular_acceptance_02
```

正式训练使用新目录，按最终组别准备。例如两组对照：

```bash
bash visual_simulator_docker/run_circular_on_disk.sh prepare \
  --run-dir runs/stage13_circular_01 --arms mask_only mask_rotation
bash visual_simulator_docker/run_circular_on_disk.sh train \
  --run-dir runs/stage13_circular_01
```

`train`后台顺序运行所选组，自动从最后完整轮次继续；`--arm mask_rotation`
可单独运行冻结方案中的一组。仅旋转组的方案应在准备时使用`--arms mask_rotation`。
训练完成后，显式运行`evaluate --run-dir ...`才进行已有测试图的开发比较。

## 验收材料与复现

- `preview/augmentation_grid.png`与单张PNG：实际图像效果；箭头只表示方向，不按米制像素比例绘制。
- `preview/completion.json`：源图、标签、角度、虚拟航向及图像哈希；仅使用训练集。
- `smoke/<组>/completion.json`：GPU训练与部署一致性结果，不是正式训练精度。
- `arms/<组>/engine_scaled_units/`：逐轮日志和含优化器、随机数状态的断点。
- `arms/<组>/deployment/`：自带固定掩膜的米制TorchScript与ONNX模型。
- `frozen.json`：原始数据、共享依赖、新代码和配置的哈希；修改后使用新的运行目录。

回归测试在现有PyTorch容器的项目目录执行：
`python -m unittest discover -s tests -p 'test_stage13_circular.py' -v`。
覆盖相机投影/旋转符号、向量长度、缺失四角隔离、留出集保护、随机数恢复、
断点续训、原网络参数及部署预处理一致性。新文件沿用Black和Flake8配置。
