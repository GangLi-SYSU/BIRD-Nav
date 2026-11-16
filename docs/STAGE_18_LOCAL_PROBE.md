# 第一步：起点周围固定位置、多航向诊断

## 作用与已确认设置

用户要求逐步改进。本步先保持第589轮模型、场景、40m高度、70°下视相机、圆形掩膜和
控制器不变，通过新拍的固定位置图像检查原点输出与局部向量场；没有训练或闭环导航。

用户已确认`cross13`：原点加四个方向各0.5、1、2m，共13位置×8航向=104张。
航向均为0°、45°……315°。正式目录为
`visual_simulator_docker/CodeInsectInspired/runs/stage18_local_cross13_01/`。

共同准备已完成，6项回归检查、代码静态检查和启动脚本语法检查通过。

已于2026-11-02 12:05（北京时间）完成104/104张及全部采图一致性核验。
原点8航向预测距离均值2.828m、最大4.586m，航向离散度2.643m；
新拍0°/90°照片与历史照片的平均像素差约1.99/255。
产物见`runs/stage18_local_cross13_01/result_summary.md`和`analysis/`。
用户随后授权本轮按推荐设置自主推进；下一步见[对称近家补数](STAGE_19_NEAR_DATA.md)。

## 如何保证结果可用

1. 核验并冻结上一轮模型、场景、成像代码、配置与新诊断代码的哈希。
2. 沿用4相机、独立位姿/帧校验、两次连续RGB变化≤0.5/255和原渲染设置保护。
3. 每批核对实时与保存后的RGB、TorchScript与原权重输出，以及单张/批量推理一致性。
4. 记录实际相机位置、航向、真值向量、预测向量、反推的家坐标及图像凭据。
   仅将完整提交样本计入结果；中断后保留旧尝试，从缺失样本继续。
5. 比较新拍原点0°/90°图像与两张历史工程照片，报告像素差和输出差。

## 看哪些结果

- 原点8航向的预测模长：正确目标均为零。
- 同一位置换航向后，预测向量转回世界坐标，反推的家是否保持一致。
- 原点四周的箭头是否朝向真实原点，是否存在共同偏移或随航向旋转的偏差。
- 原点及零预测的方向角无定义，分别报告向量误差，避免错误统计方向正确率。

产物为`result_summary.md`、`analysis/local_vector_fields.png`、`analysis/local_summary.png`、
`analysis/home_views.png`、`analysis/diagnostics.json`以及逐张图像与凭据。
本步没有设置模型精度通过阈值，也不把这些开发探针当作独立盲测或自动加入训练集。

## 执行与续跑

从`Bee-Nav/`运行：

```bash
bash visual_simulator_docker/run_local_probe_on_disk.sh \
  --run-dir runs/stage18_local_cross13_01 --design cross13 --plan-only
bash visual_simulator_docker/run_local_probe_on_disk.sh \
  --run-dir runs/stage18_local_cross13_01
```

后台容器名为`bee-nav-stage18-probe`。完整样本、运行日志与断点均保存在当前硬盘的`runs/`中。

回归检查从`CodeInsectInspired/`执行：
`python -m unittest discover -s tests -p 'test_stage18_local_probe.py' -v`。

本步结果出来后，再确定近家补数的密度、占比、空间留出规则和训练预算；
原1m隔离规则与0.5m内密集验证冲突，需在下一步修改前明确约定。
