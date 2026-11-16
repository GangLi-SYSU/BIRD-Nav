# 第589轮模型的10m与5m归巢实验

用户要求查看10m、5m起始半径下的归巢效果。本轮复用600轮训练中按验证集选出的第589轮模型，
各16个方位×4个初始航向（0°、90°、180°、270°），共128次正式导航。
先运行10m的64次，再运行5m的64次。

已于2026-11-02 11:29（北京时间）完成全部128次导航与结果核验：10m成功40/64（62.5%），
5m成功46/64（71.875%）。5项回归检查、256项纯几何检查、8次真值向量返回及8张部署
一致性验收均通过。启动凭据见实验目录的`launch_verified.json`，最终状态见`completion.json`。

- [完整结果](../visual_simulator_docker/CodeInsectInspired/runs/stage17_radius_5_10_01/result_summary.md)
- [完整轨迹](../visual_simulator_docker/CodeInsectInspired/runs/stage17_radius_5_10_01/analysis/r10_r5_trajectories.png)
- [单侧偏移诊断](../visual_simulator_docker/CodeInsectInspired/runs/stage17_radius_5_10_01/analysis/bias_diagnosis_01/BIAS_DIAGNOSIS.md)

## 保持的设置与新增内容

- 同一房屋、树木、水池场景，40m离地高度、70°下视相机、原始1024² RGB、网络384²圆形输入。
- 同一权重、尺度44和确定性推理；本轮没有训练或新增推理增强。
- 原动态步长`min(0.1+0.4×预测距离/3,2.5)`m，最多50步；每步结束后真值水平距离≤0.5m计成功。
- 新入口仅冻结5m/10m的初始条件并生成分半径轨迹，控制和成像复用已验收的原运行器。
- 先做256项纯几何检查，以及5m/10m共8次真值向量工程返回。原运行器另在20m固定位置拍8张
  工程验收照片，检查模型部署一致性；这些不计入神经导航，没有新增20m归巢测试。

## 运行与产物

从`Bee-Nav/`运行：

```bash
bash visual_simulator_docker/run_radius_navigation_on_disk.sh --plan-only
bash visual_simulator_docker/run_radius_navigation_on_disk.sh
```

后台容器为`bee-nav-stage17-radius`。默认目录为
`visual_simulator_docker/CodeInsectInspired/runs/stage17_radius_5_10_01/`，
保存`frozen.json`、`episode_plan.json`、`progress.json`、`logs/`、
`formal/episodes/`中的逐步照片与轨迹，以及完成后的`result_summary.md`、
`analysis/r10_r5_trajectories.png`和`analysis/radius_comparison.json`。
图中绿色为成功、红色为失败，并单独放大起点附近区域。

测试进行时，独立CPU容器`bee-nav-stage17-plot`每15秒更新
`analysis/partial_trajectories.png`；图上显示北京时间和已完成数量，采用原子替换避免半张图片。
它只读取实验记录，实验结束后生成最后一帧并自动退出。
`analysis/partial_trajectories.status.json`记录图片时间与数量，
`analysis/plot_watcher.status.json`记录绘图进程状态。
聊天中的旧图片预览不会自动刷新，可重新打开PNG；若希望持续观看，
用本机浏览器打开`analysis/live_trajectories.html`，页面每15秒自动重新加载PNG。

在容器退出后，第二条命令从未完成的试验继续，完整已提交的试验会先核验后复用。
旧20m的46/64仅作为历史参考；此次结果为同场景开发对照，不验证飞行动力学、悬停或自主到家确认。

回归检查从`CodeInsectInspired/`运行：
`python -m unittest discover -s tests -p 'test_stage17_radius_navigation.py' -v`。
