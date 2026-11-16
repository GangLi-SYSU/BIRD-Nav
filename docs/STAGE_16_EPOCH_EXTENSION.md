# 增强组从300轮继续训练

## 目的与已确认预算

检查增加训练轮数能否继续降低验证误差，并改善20m归巢。
用户已确认总计600轮，新增300轮。正式实验目录为
`visual_simulator_docker/CodeInsectInspired/runs/stage16_rotation_600_01/`。

已于2026-11-01启动后台容器`bee-nav-stage16-extension`。启动验收确认第301轮完整落盘，
累计154,112次更新，继承的300轮历史与源断点哈希保持一致；凭据见新实验的
`launch_verified.json`。首个新增轮次耗时204.58秒，当前读盘较慢，总耗时需根据后续速度估算。
后续进度以`progress.json`和完整轮次的断点为准。

最近251–275轮的平均验证MSE为17.1299m²，276–300轮为15.5696m²，
仍有改善；最后两组各10轮的均值只下降约0.53%，尚不能据此认定已经收敛。
300轮基线选择第298轮：验证MSE13.4173m²，20m归巢42/64成功。

## 本次修改的作用

1. **完整续训**：从第300轮、153,600次更新的断点恢复模型、AdamW、随机数、
   加载器状态和原300轮历史。450轮累计230,400次更新；600轮累计307,200次。
2. **保留历史最优**：把第298轮验证最优断点复制为选模候选，保持恢复指针仍指向300轮。
   最终在全部历史和新增轮次中按相同验证MSE选优，先冻结权重再做导航。
3. **保持实验变量单一**：数据仍为4,096张训练图和320张验证图，位置划分不变；
   网络、384×384输入、圆形掩膜、先缩放后随机旋转、尺度44、学习率0.0005、
   AdamW参数、batch8、workers4和GPU运行设置沿用原实验。
4. **自动重复相同测试**：训练进程结束后，单独启动Isaac，先完成相机和模型检查，
   再运行20m的16方位×4初始航向=64次归巢。成功半径0.5m、上限50步不变。
5. **保护旧结果**：新增入口和模块使用`epoch_extension`名称；旧实验代码、断点和结果保持原哈希。

## 启动与续跑

从`Bee-Nav/`运行：

```bash
bash visual_simulator_docker/run_epoch_extension_on_disk.sh \
  --run-dir runs/stage16_rotation_600_01 --target-epochs 600 --plan-only
bash visual_simulator_docker/run_epoch_extension_on_disk.sh \
  --run-dir runs/stage16_rotation_600_01
```

首次命令校验并冻结计划；第二条启动后台容器`bee-nav-stage16-extension`。
已有容器运行时不重复启动。容器退出后用第二条命令继续，读取最近完整轮次，
并复用已通过的阶段。图形预览需先关闭以释放GPU。
模型、日志、图像和缓存均继续存放在当前硬盘。

在`visual_simulator_docker/CodeInsectInspired/`实时查看损失：

```bash
/media/lg/lg-512/conda_envs/bird-nav/bin/python tools/watch_training_loss.py \
  --run-dir runs/stage16_rotation_600_01
```

查看器每5秒刷新，新损失按完整轮次记录；关闭窗口不影响训练。
默认显示米制MSE；原始日志的归一化MSE需乘44²=1936。

## 验证与产物

已通过6项CPU回归检查，以及新增代码的格式、静态检查和启动脚本语法检查。
回归检查覆盖真实300轮状态移交、保留298轮而不移动恢复指针、配置约束、
两种计划的一致性和随机数据下续训与连续训练的逐项一致性。
在项目镜像中运行：

```bash
python -m unittest discover -s tests -p 'test_stage16_epoch_extension.py' -v
```

新实验目录包含`progress.json`、`logs/`、`training/`、`navigation/formal/episodes/`。
完成后生成`result_summary.md`、`analysis/comparison.json`、
`analysis/epoch_budget_learning_curves.png`和`analysis/epoch_budget_trajectories.png`。

本次检验增加计算预算的收益；单场景、单训练种子且渲染存在波动。
若仍选中298轮，导航差异属于同一模型的重复测量。更多轮次也可能无法解决近家数据不足的问题。
