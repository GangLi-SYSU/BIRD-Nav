# 两组圆形输入训练与20m归巢对照

## 本轮范围

用户选择`mask_only`（圆形掩膜无旋转）与`mask_rotation`（圆形掩膜随机旋转）两组。
训练使用既有512个训练位置/4096张图、40个验证位置/320张图，各150轮、76,800次更新。
两组重新以同一网络参数初始化，384×384、10,628参数、目标除44、AdamW设置保持一致。
验证集没有随机旋转，按各组完整验证集米制分量MSE选择最佳轮次。

**本轮仅测试从20m半径归巢，不运行5/10/30/40m归巢。**
每组16个圆周方位×4初始航向（0/90/180/270°）=64次，两组合计128次，起点一一对应。
沿用0.5m水平成功圈、50步、动态步长`min(0.1+0.4*预测距离/3,2.5)m`以及机头转向。
不因增强组表现改变阈值、模型选择或控制器。完整半径测试留待用户后续决定。

## 自动执行顺序

1. GPU顺序完成两组150轮训练，保存各组完整断点和逐轮验证损失。
2. 在任何神经网络导航之前冻结两组各自验证最佳模型及部署文件。
3. 执行20m纯几何oracle、20m真实渲染oracle和圆形模型实时图像/单位/批量一致性检查。
4. 每组运行64次20m神经网络归巢，保存每步原始RGB、相机稳定性凭据、预测、执行和轨迹。
5. 汇总成功率、配对成功/失败情况、步数、最终距离和近家轨迹，生成报告。

自动等待器检测到训练失败会停止并写出错误，训练未完成时不会提前测试。
同一时刻只有一个Isaac应用；每组使用四个相机并行执行独立轨迹。
即使Isaac错误地以exit0退出，也必须有完整阶段凭据和64次结果才能标记完成。
中断后的未完成轨迹保留在独立attempt目录，续跑从原定起点重新开始，不接续不完整轨迹。

## 输出与查看进度

- 正式训练：`visual_simulator_docker/CodeInsectInspired/runs/stage13_circular_01/`。
- 20m测试：`visual_simulator_docker/CodeInsectInspired/runs/stage14_rotation_r20_01/`。
- 训练进度：`arms/<组>/engine_scaled_units/progress.json`。
- 自动流程进度：测试目录`progress.json`；等待时包含两组训练轮次，测试时包含当前组和完成次数。
- 最终报告：测试目录`result_summary.md`。
- 对照轨迹：`analysis/r20_comparison_trajectories.png`；训练曲线：`analysis/training_curves.png`。
- 配对明细：`analysis/comparison.json`。

从`Bee-Nav/`可查看后台日志：

```bash
bash visual_simulator_docker/disk_docker.sh docker logs --tail 30 bee-nav-stage13-circular
bash visual_simulator_docker/disk_docker.sh docker logs --tail 30 bee-nav-stage14-rotation
```

首次准备/声明和启动自动流程：

```bash
bash visual_simulator_docker/run_circular_on_disk.sh prepare \
  --run-dir runs/stage13_circular_01 --arms mask_only mask_rotation
bash visual_simulator_docker/run_circular_on_disk.sh train --run-dir runs/stage13_circular_01
bash visual_simulator_docker/run_rotation_comparison_on_disk.sh --plan-only
bash visual_simulator_docker/run_rotation_comparison_on_disk.sh --wait-for-training
```

已有运行容器时不重复启动。训练容器退出后，同一`train`命令可从最近完整轮次继续；
测试控制器退出后，同一`--wait-for-training`命令验证并复用已完成轨迹。
全部输出仍在项目硬盘，容器无网络，不使用/home存储。

## 结果边界

两组共享场景、起点与控制条件，但闭环中后续位姿不同，实时渲染图也可能存在微小差异。
64次中的航向、相邻视野相关；一次训练种子下的成功率差异仅作为20m增强收益的初步证据。
不把上千个轨迹帧当作独立试验，也不把最终入圈距离解释为自主到家检测或稳定悬停精度。
本轮不执行另两个静态测试集的评估；训练验证集仍按原划分使用。
