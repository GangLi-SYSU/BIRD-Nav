# 第十阶段：真实地物场景训练

日期：2026-10-28。**1176 张新图、完整数据验收、150 轮正式训练及最终评估全部完成，共 19,950 次正式更新。** 测试平均方向误差由上一轮 64.31° 降至 14.06°；按各自学习半径归一化的向量误差由 65.17% 降至 25.46%。原始向量误差为 11.20 m，学习范围已从约11 m扩大到44 m，不能仅按米制误差直接比较。仍有距离低估和后期过拟合，未运行闭环归巢。

结果见[完整摘要](../visual_simulator_docker/CodeInsectInspired/runs/stage10_real_training/result_summary.md)和[最终核对记录](../visual_simulator_docker/CodeInsectInspired/runs/stage10_real_training/completion.json)。流程状态见 [progress.json](../visual_simulator_docker/CodeInsectInspired/runs/stage10_real_training/progress.json)，总日志为 [training_02.log](../visual_simulator_docker/CodeInsectInspired/runs/stage10_real_training/training_02.log)。

## 训练前的相机修复与验收

预览的连续帧差约 0.6/255，未通过既有 0.5/255 门槛。无界面测试表明，仅关闭自动曝光或重新写入全局抗锯齿设置仍然不够：场景初始化会恢复 DLSS，而 Kit 的 `/rtx-transient/post/aa/limitedOps` 限制会覆盖关闭抗锯齿的请求。

本机 Replicator 官方接口 `set_render_rtx_realtime(antialiasing="Off")` 同时解除该限制。正式采集在场景和相机初始化后执行该设置，逐张记录并校验实际抗锯齿、限制开关和自动曝光，后续若被覆盖则停止。未修改相机光学参数、网络、标签或稳定阈值。

独立工程验收使用真实房屋、树木、水池场景，在路口、住宅、水池、公园四个位置各取八个航向，分别验证单相机和四相机，**64 张全部通过**。单相机最大接受帧间 MAE 为 0.37295/255，四相机为 0.36246/255。每张保留递增帧、实际位姿、真实投影和两次连续 MAE≤0.5 的证据，仍使用两渲染子帧、最多 90 次更新。

- [真实相机验收](../visual_simulator_docker/CodeInsectInspired/runs/stage10_real_training/camera_check_03/camera_report.json)
- [工程照片目录](../visual_simulator_docker/CodeInsectInspired/runs/stage10_real_training/camera_check_03)

工程照片不进入训练数据。原预览报告保持其当时未通过的状态。采集修改前的源码保存在 `runs/stage10_real_training/historical_sources/`，附 SHA-256。四个冻结的数据/原网络核心源码和上一轮模型校验值保持不变。

新增两项回归测试验证渲染初始化覆盖和后续意外覆盖拒绝；已有 23 项采集、标签及文件完整性测试也通过，共 25 项。新源码 Flake8 检查通过。

## 已确认的学习范围

相机保持已预览的 40 m AGL、70°×70°、1024×1024。单张在平地覆盖约 56×56 m；原学习轨迹的最大半径只有约 11 m。

| 选项 | 原蜂式轨迹参数 | 实际最大半径 | 作用 |
| --- | --- | --- | --- |
| 推荐：放大四倍 | `[5,150,1.4]` | 43.98 m | 学习范围扩大至更多住宅、水池区域 |
| 保留原范围 | `[5,150,0.35]` | 11.00 m | 先在原学习范围观察新场景与相机效果 |

用户已选择第一项。最终配置为 `config/experiments/downward_h40_hfov70_real_r44.json`，训练协议为 `stage10_training_protocol.json`。路径最大半径 43.9823 m，位置数、航向数、分组比例、网络及训练参数均按确认设置执行。

正式方案保持 147 位置×8 个真实航向，共 1176 张新图；保留按位置分组、133/6/8 位置的训练/验证/测试划分。使用上一轮 384×384 模型（10,628 参数）、从头初始化、150 轮、batch 8、AdamW、学习率 0.0005；无新增增强、早停或学习率调度。数据逐像素、标签、分组及读取验收均已通过，最终模型、ONNX、误差和学习曲线已输出。

最终结果属于当前学习区域内的位置留出测试，不等于场景外泛化或闭环导航。如果扩大半径，米制误差不能直接与上一轮较小范围作单一因果比较，应同时报告角度和按学习尺度归一化的误差。

## 四相机批量等待与实际执行

顺序调用四个相机的等待函数，会在每次等待时渲染所有产品，实测约 5 秒/图。新增批量等待让四个产品共享渲染推进，但各自独立核对递增帧、位姿、光学参数和连续稳定证据；单个产品过旧或错误会使整批失败。未更改两子帧、90 次上限、0.5/255 阈值、采样位置或照片尺寸。

批量回归新增 3 项测试，另有 23 项相机/稳定性测试及 23 项采集/文件/标签测试通过。独立[批量实机验收](../visual_simulator_docker/CodeInsectInspired/runs/stage10_real_training/camera_check_05/camera_report.json)的 64 张照片全部通过；四相机 32 张用时约 55 秒，最大接受 MAE 为 0.34181/255。早先工程验收 `camera_check_03/` 验证的是顺序方式；两者均不进入训练集。

首次顺序采集 `real_01/` 已中止，文件保留，修改前源码保存在 `sequential_capture_sources/`；这批未完成数据不与新数据混合。正式新数据为 `runs/downward_h40_hfov70_real_r44/real_02/capture/`。先前重叠实例造成的工程启动显存不足已通过退出旧实例解决，后续只运行一个 Isaac 实例。

本次输出路径均在 `CodeInsectInspired/` 下：

- `runs/downward_h40_hfov70_real_r44/real_02/`：1176 张新图、真实位姿、标签和稳定凭证。
- `runs/stage10_real_training/position_groups_01/`：按位置分组的数据和完整性验收。
- `runs/stage10_real_training/attention384_real_r44_01/`：读取探测、独立短训练、150 轮正式训练和评估。
- `runs/stage10_real_training/analysis_01/`：完成训练后生成学习曲线、实际 384×384 输入及与前次的对比。

后台容器名为 `bee-nav-stage10-training`，使用本硬盘 Docker 服务；调度脚本 `runs/stage10_real_training/run_training.py` 将采集、数据验收和离线训练放在独立进程，采集退出后释放 Isaac 显存。短训练权重不继承到正式训练。正式训练交付第 150 轮模型，保持 19,950 次正式更新，不按测试集选择权重。

## 正式完成结果与实际限制

| 集合 | 图像数 | 最终向量误差/m | 最终方向误差/° | 最终组件MSE/m² |
| --- | ---: | ---: | ---: | ---: |
| 训练 | 1064 | 1.665 | 4.85 | 2.033 |
| 验证 | 48 | 5.996 | 13.05 | 26.501 |
| 测试 | 64 | 11.199 | 14.06 | 82.032 |

表格使用同一个第150轮模型重新评估。训练日志的逐轮训练损失是当轮更新期间的在线平均，与最终模型的训练集评估值定义不同。

测试方向误差降低约78.1%，按学习半径归一化的向量误差降低约60.9%。本轮测试64张中，方向误差超过90°的数量为0；上一轮为16。测试平均真实距离为32.58 m，预测距离为27.69 m，75%的测试图低估距离，平均带符号距离偏差为-4.90 m。因此主要改善是方向与相对尺度精度，距离估计仍需优化。

验证MSE在第38轮最低为11.376，第150轮为26.501；后期过拟合依旧存在。没有按测试集改选权重或追加训练。若下一步比较更早模型，应按验证集预先确定选择规则。

实际采集最大接受MAE为0.49936/255，等待4–6次更新。1176张图片哈希均不同，XY与距离标签均按原轨迹放大四倍，最大数值差约2.1e-13 m，分组成员和航向保持一致。数据逐像素、在线RGBA与磁盘RGB以及标签检查均通过；额外36次优化仅用于读取探测和独立短训练，未继承至正式模型。

PyTorch保存重载最大预测差为0；ONNX参考最大差约2.96e-5，按原`atol=1e-5,rtol=1e-4`通过。外部输入仍是1024×1024 RGB，网络内部统一缩放为384×384。通过第一层实际前向钩子核对为`[1,3,384,384]`，见[实际输入图](../visual_simulator_docker/CodeInsectInspired/runs/stage10_real_training/analysis_01/actual_network_input.png)。F阶段沿用的原网络兼容性诊断图不代表本轮正式模型输入。

图片与权重：

- [完整学习曲线](../visual_simulator_docker/CodeInsectInspired/runs/stage10_real_training/analysis_01/learning_curves.png)、[尺度归一化对比](../visual_simulator_docker/CodeInsectInspired/runs/stage10_real_training/analysis_01/final_comparison.png)、[固定顺序测试样例](../visual_simulator_docker/CodeInsectInspired/runs/stage10_real_training/analysis_01/test_examples.png)。
- [PyTorch模型](../visual_simulator_docker/CodeInsectInspired/runs/stage10_real_training/attention384_real_r44_01/train/model_attention.pth)、[ONNX模型](../visual_simulator_docker/CodeInsectInspired/runs/stage10_real_training/attention384_real_r44_01/evaluate/model_attention.onnx)。

对照408个原有文件核对，除README及两处采集修复源码外均保持原样；修改前源码已归档并验证，四个冻结数据/原网络核心源码、原理论模型权重和上一轮模型均未变。

后续只读诊断见[剩余误差分析](STAGE_10_ERROR_DIAGNOSIS.md)：验证集缺少30 m以外的位置；直接采用验证最优第38轮并未改善外圈为主的测试向量误差，未替换原交付模型。
