# 第三阶段：100 m 平坦场景、资源完整性与空间预算

完成日期：2026-10-23。对应总清单 **C1–C8**。本阶段建立场景并打开 Isaac 浏览，未采集正式数据、训练网络或运行归巢。

**现在看到的场景：** 一块 `100×100 m`、位于 `z=0 m` 的平坦地面，带有草土色、多尺度的程序纹理。没有建筑、树木、屋顶，也没有给“家”添加特殊标记。这是先排除遮挡和几何问题的工程场景，不是已经验证过导航能力的真实城市环境。

## C1、C2、C8：独立生成，使用真实米制尺寸

**作用：** 为 10 m 下视观察提供尺度明确、飞行平面无障碍的起点，保留原场景供对照。

**修改：** 新增 [场景生成器](../visual_simulator_docker/CodeInsectInspired/generate_downward_environment.py)。它直接生成 ASCII USD、纹理和资源清单，不启动 Kit，也不放大旧建筑。输出目录必须不存在；再次使用同一目录会拒绝覆盖。

**结果：** `metersPerUnit=1`、Z-up、XY 各为 `[-50,50] m`，地面为一个带碰撞属性的平面网格。场景没有高出地面的几何体，10 m 相机平面与可见表面的垂直间距为 10 m。以后扩大场景时调整地面尺寸和纹理覆盖范围，并重新验证，不改变现有物体的比例。

生成产物位于当前硬盘的：

```text
visual_simulator_docker/CodeInsectInspired/maps/downward_flat_v1/
├── distinct.usda / distinct.png / distinct.manifest.json
├── repeated.usda / repeated.png / repeated.manifest.json
├── low_texture.usda / low_texture.png / low_texture.manifest.json
└── *_texture_overview.png
```

整个目录约 21 MB，作为可重建产物加入 Git 排除规则；源码、配置和生成命令保留。默认研究配置现指向 `maps/downward_flat_v1/distinct.usda`。

## C3、C5：检查边界、足迹和连续线段

**作用：** 相机中心位于地面上，并不足以保证整幅图像都在有效地表内。

**修改：** 新增 [scene_geometry.py](../visual_simulator_docker/CodeInsectInspired/insect_utils/scene_geometry.py)，计算任意偏航下图像足迹的外接圆，再收缩有效相机区域；检查计划位置、高度和相邻线段。

**结果：** 在暂定 10 m、HFOV/VFOV 30°、方形传感器条件下：

| 项目 | 数值与含义 |
| --- | --- |
| 图像地面足迹 | 约 `5.359×5.359 m`，仍待 D 阶段实测成像校准 |
| 足迹外接圆半径 | 约 `3.789 m`，涵盖任意水平偏航 |
| 额外边界余量 | 5 m |
| 相机中心有效边界 | X/Y 均约为 `[-41.211,41.211] m` |
| 学习路径 | 原 bee `[5,150,0.35]`，147 点及 146 条相邻线段通过 |
| 测试起点 | 原 `5/10/15/20/25 m ×16`，80 个起点通过 |

检查器对凸矩形内的线段使用端点判定；对当前唯一的平坦网格，净空检查是精确的几何判断，不需要用射线近似。测试起点之间的连线检查不等于实际闭环轨迹验收。

**尚需接入控制器的条件：** 25 m 起点加上最坏方向的 `50×2.5 m` 动作，中心可能到达 150 m；若完全不限制运动，场景边长需要至少约 **317.58 m**。因此目前 100 m 场景只覆盖计划学习区域和测试起点，后续 D/H 必须调用边界检查，越界时记录明确终止原因，不能让黑色视场继续送入网络。本阶段实现了检查函数，未冒称已经接入尚未实现的下视闭环。

## C4、C7：提供三种信息条件，记录纹理物理尺度

**作用：** 区分方法能够利用视觉差异的情况，以及重复图案、缺少纹理造成的观测歧义。

| 场景 | 外观与目的 |
| --- | --- |
| `distinct` | 单张全场纹理，混合约 16/4/1/0.25/0.0625 m 尺度变化及均匀随机细线；用于首个平面实验 |
| `repeated` | 周期约 2.002 m 的棋盘状地面；故意制造不同位置的相似观测 |
| `low_texture` | 空间恒定颜色；用于测量缺少位置线索时的行为 |

纹理均为 `4096×4096`、seed 10、sRGB，UV `[0,1]` 覆盖完整 100 m 地面，边缘为 clamp。纹理不读取家位置或学习路径，也不使用标注坐标或专属地标作为归巢提示。

每纹素约 2.441 cm；在 10 m / 30°下，约 5.36 m 足迹对应约 **220 个纹理像素**。原始相机计划输出 1024 像素不代表地表具有 1024 个独立纹理细节；将来提高纹理分辨率或采用实景素材属于明确的场景版本变化。

局部纹理检查确认：distinct 的四块局部区域都有空间变化；repeated 平移一个实际周期后像素完全一致；low_texture 的空间标准差为零。这些检查说明诊断条件成立，不证明训练或导航成功。

另增加了 [重复纹理配置](../visual_simulator_docker/CodeInsectInspired/config/experiments/downward_h10_hfov30_repeated.json)和[低纹理配置](../visual_simulator_docker/CodeInsectInspired/config/experiments/downward_h10_hfov30_low_texture.json)，网络、训练预算、控制设置均与主配置一致。

## C6：审查旧场景，明确迁移阻碍

**作用：** 避免在资源缺失、存在屋顶或飞行平面穿过物体时，直接把旧相机高度改成 10 m。

**修改：** [场景审查工具](../visual_simulator_docker/CodeInsectInspired/tools/check_stage03_scene.py) 使用容器内 OpenUSD 读取依赖和三维包围盒，不启动另一个 Kit。旧 USD 均保留。

| 旧场景 | 未解析资源数 | 中心约 ±30 m 区域内与 z=10 m 相交的几何包围盒数 | 迁移结论 |
| --- | ---: | ---: | --- |
| park | 7 | 118 | 先修资源，再选择开放区域 |
| urban | 18 | 44 | 先修模型/HDR，再检查建筑与局部表面 |
| courtyard | 5 | 5 | 先修资源，再检查庭院边界与高物体 |
| warehouse | 2 | 24 | 不能直接升高；检查结构、封闭空间和屋顶 |
| greenhouse | 0 | 0 | 资源可解析，但最高表面约 6.575 m；仍需检查俯视时屋顶遮挡与真实 AGL |

这些数值是当前本地依赖可解析部分的**保守包围盒筛查**，不是三角网格碰撞或射线自由空间证明。缺失资源可能使高度/净空统计不完整。greenhouse 的零相交不能解释为适合下视导航：相机可能只看到屋顶。

## 场景依赖进入实验指纹

第二阶段只记录入口 USD；现在对新场景同时记录 sidecar manifest 和其中所有 USD/纹理的实际内容哈希。修改或删除纹理会改变运行指纹，旧运行不能继续复用。OpenUSD 独立解析出的完整文件依赖与 manifest 一致，每个新场景恰好只有自己的 USD 和 PNG，未解析资源数均为零。

这是本次生成场景的完整文件闭包，不表示旧场景的外部 MDL、网络资源或所有未来场景都已得到相同保证。

## Isaac 预览与操作

按用户要求，已实际打开 Isaac GUI，并检查了 10 m 局部地面显示。预览器不会调用数据采集、训练或归巢入口。它创建的 `StageCOverview`、`StageCAt10m` 是仅用于浏览的临时相机，原场景文件没有因此重写。

窗口内的 `Stage C: terrain preview` 面板提供全景、10 m 局部视角和三种地面切换。如果关闭了面板，也可从视口相机下拉菜单选择这两个相机。纵向窗口中图像上下的留黑是视口显示区域与相机画幅比例不同，不能当作地面越界的判断依据。

预览器和正式实验相机的区别仍然明确：**下视采集与推理适配尚未实现**。机体系方向、偏航、真实 FOV、采集/推理一致性及帧同步均留在 D/E 阶段验收。

从宿主桌面终端重新打开预览（先关闭已开的同名预览）：

```bash
cd /media/lg/lg-512/new_workspace/Bird_Nav_ws/Bee-Nav/visual_simulator_docker
bash view_downward_scene_on_disk.sh
```

仅生成新版本场景，需使用有 NumPy/Pillow 的 Python，不依赖 Isaac：

```bash
cd CodeInsectInspired
/media/lg/lg-512/conda_envs/bird-nav/bin/python generate_downward_environment.py --output-dir maps/downward_flat_v2
```

在容器 `CodeInsectInspired/` 目录中检查场景：

```bash
../python.sh tools/check_stage03_scene.py --legacy --output runs/my_scene_audit
../python.sh -m unittest discover -s tests -p 'test_stage*.py'
```

`tools/render_stage03_preview.py` 另提供离线诊断视图导出接口；本阶段为保持用户预览流畅，没有同时启动第二个渲染器，因此该独立导出工具尚未做 GPU 运行验收。当前图像证据来自实际打开的 GUI，而非该工具。

## 验证证据与下一阶段

- 宿主 Python 3.8 与容器 Python 3.10 中，累计 **42 项测试通过**；包括纹理漂移、资源缺失、非零家位置边界、错误高度、低净空和线段越界。
- OpenUSD 新场景几何/依赖检查通过，旧五场景检查完成；详见[最终场景报告](../visual_simulator_docker/CodeInsectInspired/runs/stage03_final_audit/scene_report.json)及[最终容器日志](../visual_simulator_docker/CodeInsectInspired/runs/stage03_final_verification.log)。日志中旧场景的缺失资源警告是审查发现，不是新场景缺失。
- 当前生成器重新生成的 6 个 USD/纹理文件逐字节一致，覆盖现有目录的操作被拒绝，见[重建验证](../visual_simulator_docker/CodeInsectInspired/runs/stage03_scene_audit/reproducibility_report.json)。原始 manifest 保留格式化前的生成器文件哈希，重建报告记录最终源码及实际一致性。
- [Isaac 实际窗口截图](../visual_simulator_docker/CodeInsectInspired/runs/stage03_scene_audit/isaac_preview.png)显示局部地面正常渲染；它包含 GUI 叠加层与视口留边，不属于训练数据或标定图。

下一阶段 D 将共用下视相机内外参、校准朝向和真实 30°覆盖，并比较采集端与推理端的输出；随后才进入正式数据流程。
