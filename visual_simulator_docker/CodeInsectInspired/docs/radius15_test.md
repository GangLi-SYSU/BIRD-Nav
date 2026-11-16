# 第484轮模型的15 m补充测试

使用 `stage23_compact_spiral_h40_fov30_01` 的同一验证最优模型：离地40 m、30°下视相机、384×384圆形输入，训练数据来自20 m范围内的147个螺旋位置。不重新训练或选择权重，不修改原10 m／20 m结果。

本次正式神经导航仅为15 m：16个起始方位×4个初始航向（0°／90°／180°／270°）=64次。成功半径2 m，每步执行后按真值水平距离判定，最多50步，动态步长仍为 `min(0.1+0.4×预测距离/3,2.5)` m。

先执行纯几何检查及15 m的4次渲染真值向量检查。原模型8张实时图像的部署一致性检查已核验并复用；这些工程检查不计入正式神经导航成功率。

从 `Bee-Nav/` 启动：

```bash
bash visual_simulator_docker/run_radius15_on_disk.sh \
  --run-dir runs/stage24_radius15_01
```

输出独立保存在 `runs/stage24_radius15_01/`：

- `pipeline_progress.json`：当前阶段和正式测试进度。
- `analysis/r15_trajectories.png`：完整轨迹与近家放大图，每30秒更新已完成轨迹。
- `analysis/live_results.html`：自动刷新的轨迹页面。
- `formal/results.json`、`result_summary.md`：全部64次结束后生成的最终结果及10／15／20 m对照表；10 m、20 m为原实验历史结果。

测试使用相同场景和训练种子，属于额外开发评估，不代表实机自主到家识别或2 m以内的定位精度。
