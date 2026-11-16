# 第四阶段：下视相机接入与标定

日期：2026-10-23。**阶段 D 的相机层验收已完成：标定板 12 张、真实路标场景 8 张图像通过几何与颜色一致性检查，动态稳定和超时保护通过，72 项测试通过。** 未采集正式训练集、未训练网络、未运行闭环归巢；完整采样与控制仍待 E/H。

## 已获用户确认的实验定义

用户回复“全部采用推荐”后，以下定义已写入四个下视配置与独立的 [stage04_camera_protocol.json](../visual_simulator_docker/CodeInsectInspired/config/experiments/stage04_camera_protocol.json)：

| 设置 | 确认值 |
| --- | --- |
| 投影 | 理想针孔，无畸变、无景深 |
| 图像 | `1024×1024`，完整 RGB，不使用全景条带裁剪 |
| 视场 | **水平**视场 30°；方形成像面对应垂直视场 30° |
| 高度 | 相机光心距平坦地面 10 m；不随路标顶部改变高度 |
| 安装 | 固定垂直下视，图像上方对应机头，随偏航旋转 |
| 姿态 | 机体保持水平，相机安装偏移暂为零 |
| 网络 | 原 attention，10,820 参数，内部 `192×1800` resize 保留 |

这确认的是仿真模型定义，不代表实机相机已经完成内参标定。第一阶段冻结的历史协议没有重写。

## D1、D3：共用内参与单位换算

**作用：** 确保采集端、推理端使用同一视场和像素坐标定义。

新增 [camera_geometry.py](../visual_simulator_docker/CodeInsectInspired/insect_utils/camera_geometry.py) 与 [downward_camera.py](../visual_simulator_docker/CodeInsectInspired/insect_utils/downward_camera.py)。前者仅使用标准库，推导内参、位姿、投影与坐标变换；后者负责真实相机、渲染产物和图像读取。

核对安装版本源码后，明确区分了：

- USD/Replicator 的焦距与 filmback 数值单位是 `0.1×场景长度单位`。
- Isaac Camera 的相关 setter 还会乘以 10，不能直接复用同一个数值。
- clipping range 使用场景长度单位；本场景为米。

两端现在由同一模块直接配置相机 USD 属性，关闭景深，设置零光圈偏移；相机 API 读取出的内参也与共用矩阵核对。

当前等效虚拟焦距为 100 mm，成像面约 `53.5898×53.5898 mm`；米制场景的原始 USD 数值分别为 `1.0` 与 `0.5358984`。这些是实现理想投影的**数值尺度**，不是选用了某款 100 mm 实机镜头。关闭景深时，焦距与成像面同时缩放不改变目标视场。先前较小尺度方案虽然写入了参数，实际渲染视场却偏窄，因此最终结果以标尺测量为准。

共同内参约为：

```text
fx = fy = 1910.810013 px
cx = cy = 512 px
near/far = 0.05 / 1000 m
```

## D2：采集与推理的实际相机接口

**作用：** 同时适配两端，避免训练图片和在线图片采用不同相机模型。

- [render_dataset.py](../visual_simulator_docker/CodeInsectInspired/render_dataset.py) 新增 `create_downward_capture_camera`：共用 USD 针孔相机，加 Replicator render product 与 RGB 输出。
- [virtual_homing_ANN.py](../visual_simulator_docker/CodeInsectInspired/virtual_homing_ANN.py) 新增 `create_downward_inference_camera`：实际使用 `omni.isaac.sensor.Camera`，显式指定 USD 相机坐标轴。
- 推理 `get_image` 已有下视适配器分支，返回完整图像；`downward_rgb_tensor` 的 RGB/CHW/`[0,1]` 数值逐元素验证通过。

GPU 标定调用的正是这两个入口文件中的工厂函数及推理取图分支。

**完整实验入口仍有阶段限制。** 原全景采集、训练和归巢代码保留；传入下视配置时，完整数据采集会明确要求完成 E，闭环控制会要求完成 H。统一控制入口也保持对应限制，避免旧尾数处理、标签关联、全景裁剪或旧控制符号被当作已完成的下视流程。

## D4：统一外参与归巢向量

世界系 Z 向上；机体系 X 前、Y 左、Z 上，正偏航从世界 X 朝 Y。下视相机的图像右方对应机体 `-Y`，图像上方对应机体 `+X`。

因此在 USD 相机系中，相机绕世界 Z 的旋转是 `body_yaw−90°`，而不是把所有接口都填成 `pitch=-90°`。Camera API 使用 scalar-first `wxyz`，并显式传 `camera_axes="usd"`；渲染返回的相机矩阵必须与请求的世界位姿匹配。

水平归巢标签和还原仍为：

```text
target_B = Rz(-yaw) × (home_xy - position_xy)
target_W = Rz(+yaw) × target_B
```

四方位、非零家位置、任意 yaw 和像素反投影已有数学测试。原全景控制器的额外负号没有搬进新数学模块；真正动作链和 oracle 验收留在 H。

## D5、D6：有色标记与实际视场

**作用：** 用真实图像确认方向和覆盖，不能只检查 JSON。

标定场景仅保存在运行目录，没有给正式路标场景增加网格或专属标记。网格中心在非零坐标 `(12,-8)`；红/绿/蓝/黄分别位于世界 `+X/+Y/-X/-Y`，白色标记为中心。相机测试 0/90/180/270°、平移及返回原位，共 **12 张有效图像**。

[最终标定报告](../visual_simulator_docker/CodeInsectInspired/runs/stage04_camera/calibration_settled_02/camera_report.json)和[完整验收日志](../visual_simulator_docker/CodeInsectInspired/runs/stage04_camera_settled_validation_02.log)记录：

[四个航向的实际标定图](../visual_simulator_docker/CodeInsectInspired/runs/stage04_camera/calibration_settled_02/calibration_contact_sheet.png)展示了两端相同的朝向与画幅。

| 指标 | 实测范围/最大值 | 检查标准 |
| --- | --- | --- |
| 水平视场 | `29.98765–30.00932°` | 相对目标误差 <1% |
| 地面覆盖宽度 | `5.35667–5.36073 m` | 理论约 5.35898 m |
| 标记投影误差 | 最大约 `0.620 px` | ≤1.5 px |
| 标定板两端平均 RGB 差 | 最大约 `0.766/255` | ≤2/255 |

原网络完成了未训练的前向检查，参数量仍为 10,820，输出为 `[1,2]` 且有限；这不是导航精度结果。

## D8：等待真实的新帧

**作用：** 过滤默认位姿帧、初始化黑图和上一姿态的缓存。

两端统一推进完整渲染图，不再让一个路径只推进 `World.step` 而另一个推进 Replicator。先要求帧号推进、CameraParams 位姿矩阵与目标相同，再等待下一张同样匹配且帧号继续推进的有效 RGB。初始化全黑缓冲被拒绝；不使用固定 sleep 作为正确性保证。

每张图保存 pose ID、帧号、相机和 render-product 路径、请求位置/航向、实际相机矩阵、实际 USD 光学数值、投影矩阵和等待次数。四航向与移动标记的像素检测进一步验证了图像内容，不能仅凭元数据就宣布画面正确。

在原有新帧与位姿条件之外，现已加入用户确认的画面稳定条件：每次取图单独建立比较序列，连续两次有效 RGB 平均变化 ≤0.5/255 才返回，最多 90 次渲染更新。重复/倒退帧不计数，位姿或图像异常会重置序列；超时明确报错并清除旧图像记录。每次更新仍使用原来的 2 个渲染子帧。

[frame_stability.py](../visual_simulator_docker/CodeInsectInspired/insect_utils/frame_stability.py)记录并执行统一策略；每张返回图保存完整比较历史、连续达标次数、更新次数和耗时。两次变化需要至少三张有效 RGB 图像，第一张仅作为比较基准。

图像内容、位姿或元数据无效时会清除当前比较序列；每次新的取图独立建立基准。专门将单次更新预算设为 1 的负向检查在两个后端均得到明确超时，且没有返回旧图像记录。

## D7：真实路标场景的画面收敛与最终验收

真实 400 m 场景首次比较，两端形状与位姿对齐，但平均 RGB 差约 `4.54/255`，超过 2/255 检查线，因此[该次验收](../visual_simulator_docker/CodeInsectInspired/runs/stage04_camera/landmarks_01/camera_report.json)明确标记失败，没有修改成通过。

进一步保持同一相机、同一位姿、相同 2 个渲染子帧，测量重复图像，得到[诊断数据](../visual_simulator_docker/CodeInsectInspired/runs/stage04_camera/noise_01/diagnostics.json)：

可直接查看[初始与稳定后图像对比](../visual_simulator_docker/CodeInsectInspired/runs/stage04_camera/noise_01/initial_vs_settled.png)。

| 轮次 | 采集端相对上一轮差 | 推理端相对上一轮差 | 两端之差 |
| --- | ---: | ---: | ---: |
| 0 | — | — | 6.239 |
| 1 | 5.900 | 2.239 | 0.809 |
| 2 | 1.941 | 1.949 | 0.480 |
| 3 | 0.999 | 0.750 | 0.253 |
| 4 | 0.430 | 0.377 | 0.211 |
| 5 | 0.331 | 0.327 | 0.207 |

单位均为 8-bit RGB 的平均绝对差。该结果说明主要问题是初始渲染尚未收敛，不能把第一次对比的差异全部归因于相机模型。

用户已确认并实现动态稳定策略。最终比较先将两个相机都放到同一目标位姿，再分别按相同稳定条件取图，避免把不同启动状态混入路径对比；等价的重复位姿命令保留渲染历史。每次取图仍必须取得新帧、重新建立 RGB 比较基准，不能复用上一次的稳定计数。

原来的 **2 个渲染子帧、RGB 稳定阈值 0.5/255、连续两次、最多 90 次更新、两端平均差 ≤2/255** 全部保持，未通过放宽阈值获得验收结果。

[最终真实场景报告](../visual_simulator_docker/CodeInsectInspired/runs/stage04_camera/landmarks_settled_02/camera_report.json)覆盖四种路标和 0/90/180/270°，共 4 对、8 张图：

| 路标 / 航向 | 两端平均 RGB 差，单位为 8-bit 档位 |
| --- | ---: |
| 立方体 / 0° | 0.615 |
| 圆柱 / 90° | 0.634 |
| 圆锥 / 180° | 0.520 |
| 球体 / 270° | 0.817 |

所有图像均满足连续稳定条件，两端仍有小幅渲染差异，最大平均差约 **0.817/255**，低于 2/255。可查看[最终两端对比图](../visual_simulator_docker/CodeInsectInspired/runs/stage04_camera/landmarks_settled_02/paired_contact_sheet.png)。缩略图仅用于展示，全部数值按原始 `1024×1024` 图像计算。

本次取图开销：标定板均等待 4 次更新，约 0.624–0.833 s/张；真实场景等待 4–8 次，约 0.872–2.113 s/张。这不含 Kit 启动与 PNG 保存，后续多相机采集的吞吐需另行实测。

## 当前验证范围

累计 **72 项标准库测试通过**。原网络结构、已有权重和正式场景资源保留；没有生成正式 147 点数据集，也没有运行 150 epochs 或归巢控制。失败尝试与诊断图均保留在 `runs/stage04_camera/`，不得混入后续训练数据。

最终 GPU 验收按同一份代码依次验证真实场景、标定板、图像哈希与稳定记录，再运行容器 Python 3.10 中的 72 项测试。图像侧每条记录包含所用策略、连续变化历史、实际位姿与光学数值；两个后端的超时保护均通过。

[阶段完成记录](../visual_simulator_docker/CodeInsectInspired/runs/stage04_camera/completion.json)汇总最终报告哈希、39 个运行源码文件的哈希、场景依赖校验、主机与容器测试结果，以及已有文件的保留核对结果。

## 复查命令与下一阶段

在项目容器的 `/isaac-sim/CodeInsectInspired/` 中，以下命令只复查已保存的最终报告、稳定记录与图像哈希，不启动 Kit：

```bash
../python.sh tools/check_stage04_camera.py --verify-only --output runs/stage04_camera/landmarks_settled_02
../python.sh tools/check_stage04_camera.py --verify-only --output runs/stage04_camera/calibration_settled_02
../python.sh -m unittest discover -s tests -p 'test_*.py'
```

需要重新渲染时使用尚不存在的新目录，例如：

```bash
../python.sh tools/check_stage04_camera.py --kind calibration --output runs/stage04_camera/my_calibration
../python.sh tools/check_stage04_camera.py --kind landmarks --output runs/stage04_camera/my_landmarks
```

早期报告保留为历史几何/诊断证据，不满足当前稳定策略的新报告格式；最终验收以本文链接的 `*_settled_02` 为准。下一阶段 E 处理学习路径、完整采样数及图像—位姿—标签关联，F 处理数据划分和增强，H 才接入完整下视归巢控制。
