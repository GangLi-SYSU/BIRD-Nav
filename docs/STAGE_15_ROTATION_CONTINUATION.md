# 增强组150→300轮续训与相同20m测试

## 本次唯一实验变量

用户已授权增强组从150轮完整断点续训至总计300轮，并重做相同20m的64次归巢。
保留既有数据与位置划分、384×384网络、圆形掩膜、先缩放后旋转的增强方式、目标尺度44、
学习率0.0005、AdamW参数、batch8、4个数据工作进程和控制器。
不加入学习率衰减、早停、新数据、损失加权或其他半径测试。

原150轮训练与20m结果完整保留；本次目录为
`visual_simulator_docker/CodeInsectInspired/runs/stage15_rotation_300_01/`。

本轮已于2026-10-31 19:37（北京时间）全部完成。选中验证最佳第298轮权重，
20m测试42/64成功（65.625%），原150轮为18/64（28.125%）。
6项CPU回归检查、完整状态移交、相机/部署验收和结果完整性检查均通过。

- [正式对照报告](../visual_simulator_docker/CodeInsectInspired/runs/stage15_rotation_300_01/result_summary.md)
- [当前结果详细分析](../visual_simulator_docker/CodeInsectInspired/runs/stage15_rotation_300_01/analysis/detail_01/RESULT_ANALYSIS.md)

## 为什么是续训

父断点为`stage13_circular_01/arms/mask_rotation/engine_scaled_units/`的第150轮，
累计76,800次更新，SHA256：`fd8d2b477307b439f4226a3eb991acbc3c4806fac43abec568bec22c47f8496b`。
新实验仅重新绑定断点的实验编号，保存和回读后逐项核验其余内容完全一致：
网络、AdamW动量/步数、Python/NumPy/Torch/CUDA随机状态、数据加载器生成器和训练历史。
训练引擎恢复这些状态后从第151轮开始，新增76,800次更新，到第300轮累计153,600次。

CPU夹具回归检查验证跨实验编号的续训与连续训练具有相同权重、优化器及随机数状态；
正式执行还会检查实际GPU/PyTorch/CUDA/cuDNN设置与父实验一致。

## 自动流程

1. 校验和冻结父训练、父导航、断点、新代码与配置。
2. 在新目录移交断点，顺序完成151–300轮GPU训练；中断后从新目录最近完整轮次继续。
3. 按相同完整验证集的米制分量MSE，在1–300轮中选择最佳权重，导航前冻结。
   若150轮仍最好，继续使用它并在报告中明确说明，不依据导航结果改选模型。
4. 导出自带固定圆形掩膜、输出米制向量的TorchScript/ONNX并核验一致性。
5. 原20m工程验收通过后，运行同样的16方位×4初始航向=64次神经导航。
   成功半径0.5m、50步上限和动态步长不变。
6. 与冻结的150轮18/64成功结果进行配对比较，保存训练曲线和轨迹。

训练进程退出后才启动Isaac进程；没有额外的静态测试集评估。
本次是同一增强方法增加训练预算的对照，不是与无增强150轮等预算的对照。
实时渲染和闭环可能存在波动，单次结果不能作为跨场景或稳定悬停保证。

## 启动、续跑和结果

从`Bee-Nav/`运行：

```bash
bash visual_simulator_docker/run_continuation_on_disk.sh --plan-only
bash visual_simulator_docker/run_continuation_on_disk.sh
```

后台容器名：`bee-nav-stage15-continuation`。已有该容器时不重复启动。
容器结束后，同一命令验证并复用完成阶段，继续尚未完成的阶段。

## 实时查看损失

在`visual_simulator_docker/CodeInsectInspired/`目录，从本机桌面终端运行：

```bash
/media/lg/lg-512/conda_envs/bird-nav/bin/python tools/watch_training_loss.py
```

只读曲线窗口每5秒刷新，包含完整历史和最近50轮；新损失仍按每轮结束时记录。
默认显示米制MSE，与状态分析一致；加`--normalized`可显示原始日志中的归一化损失。
日志归一化损失乘`44²=1936`才是米制MSE。关闭查看窗口不影响后台训练。
无图形终端时，加`--save-only`生成一次PNG快照（默认`monitor/loss_snapshot.png`）。

也可在同一目录实时跟随训练日志：

```bash
tail -n 20 -f runs/stage15_rotation_300_01/logs/train_*.log
```

- `progress.json`：当前训练轮次或导航完成数。
- `logs/`：训练与导航的独立进程日志。
- `training/checkpoint_import.json`：实际150轮断点的完整状态移交核验。
- `training/engine_scaled_units/`：续训断点、逐轮进度与历史。
- `training/physical_history.json`：原150轮加新150轮的米制损失。
- `training/selection_frozen_before_navigation.json`：测试前选模。
- `navigation/formal/episodes/`：64次图像、稳定性凭据和执行轨迹。
- `result_summary.md`：最终对照结论。
- `analysis/continuation_learning_curves.png`、`analysis/r20_epoch_comparison.png`：曲线和轨迹。

回归测试在现有PyTorch镜像的项目目录执行：
`python -m unittest discover -s tests -p 'test_stage15_continuation.py' -v`。
