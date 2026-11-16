# 第一阶段：基线冻结与实验协议

日期：2026-10-22。对应总清单的 **A1–A5**。这一阶段建立后续修改的比较依据，还没有把运行中的相机改成下视，也没有开始 10 m 实验训练。

## A1：保存当前代码与环境

**作用：** 保留可追溯的改造起点。后续网络表现改变时，可以区分代码、参数、数据和运行环境的影响。

**已做修改：** 新建本地基线目录，记录当前 commit、未提交修改、375 个受跟踪文件的 SHA-256；复制 11 个关键入口/配置/辅助源文件。环境私密文件不复制，现有权重不重置或覆盖。

**结果：** Git 基线仍为 `e1ef00b1e3636466c19e326b326811e43384a039`。原有两个理论模型权重修改及 Dockerfile 修改均保留。项目镜像 ID 为 `sha256:110deb27e96333b194ee0f680b9d3253e35a06e80075e16905c1983359202980`，独立 Docker 使用 `/mnt/bee-nav-docker/docker`。检查时 ext4 存储内部约余 17 GiB，硬盘分区约余 195 GiB；这些是检查时的读数，会随使用变化。

本地记录：[source_manifest_before.json](../visual_simulator_docker/CodeInsectInspired/baselines/stage01_20261022T111712Z/source_manifest_before.json)。该目录是本机证据，不纳入 Git，也不进入 Docker 构建上下文。

## A2：分开必要适配与可选扩展

**作用：** 防止同时更换相机、网络、采样数量和控制规则后，无法解释实验结果。

**已做修改：** 新增 [stage01_protocol.json](../visual_simulator_docker/CodeInsectInspired/config/experiments/stage01_protocol.json)，分别列出原代码基线、下视目标、必须适配项、可控扩展项及论文与当前代码的差异。

**结果：** 下视投影、10 m 高度、全景裁剪/平移失效、数据对齐和坐标链归入必要适配。更多航向、更密路径、不同半径/步长/步数、光照与网络变化归入需要单独比较的扩展。协议明确标记 `execution_ready=false`，不会让尚未实现的下视实验被误认为已经可运行。

## A3：冻结实际生效的训练与控制参数

**作用：** 入口会覆盖 JSON 默认值，单看配置文件容易冻结错误的实验设置。

**已做修改：** 新增 [check_stage01_baseline.py](../visual_simulator_docker/CodeInsectInspired/tools/check_stage01_baseline.py)。程序通过语法树读取原入口、bee 参数分支和配置覆盖关系，不导入或执行原入口；将结果与协议比较，发现偏离时报出字段名。

**结果：** 已冻结 attention 网络、MSE、AdamW、学习率 `0.0005`、batch 8、150 epochs、seed 10、原学习路径、无路径噪声、原测试圈与成功阈值。实际动态步长公式也单独核对，避免误把 JSON 中的 `0.5` 当作始终生效的恒定步长。

原始 `config_training.json` 中的 `simple`、采样配置中的朝北/噪声默认值均保留；检查程序正确识别主入口覆盖后的 `attention`、随机航向和无噪声设置。这一阶段不更改旧实验行为。

检查证据：[static_report.json](../visual_simulator_docker/CodeInsectInspired/baselines/stage01_20261022T111712Z/static_report.json)、[协议快照](../visual_simulator_docker/CodeInsectInspired/baselines/stage01_20261022T111712Z/protocol_snapshot.json)。

## A4：明确暂定的下视成像条件

**作用：** 把“10 m”和“30°”落实为有物理含义的配置，避免把世界 Z、机体高度、相机高度或不同 FOV 定义混用。

**已做修改：** 协议暂按相机光心距平坦地面 10 m、光轴垂直向下、水平视场角 30°、原始分辨率 `1024×1024`、方形像素记录。物理家的地面位置与后续悬停位置分开定义。

**结果：** 该假设下地面覆盖约 `5.359×5.359 m`。协议明确保留“FOV 定义未获确认”“传感器形状暂定”和“相机安装偏移未知”的状态，不把它们写成已确定硬件参数。正式下视采集前仍须落实这些信息；本阶段全景检查不依赖它们。

## A5：建立原全景处理链检查

**作用：** 在改造前验证原相机、预处理、标签函数和网络前向能够工作，为后续相机适配提供参照。

**已做修改：** 检查程序提供显式的 `--render` 选项。在既有 urban 场景中使用原鱼眼相机、原 1.5 m 高度与原条带裁剪，限定采集 3 个位姿。使用 seed 10 初始化的原 attention 网络做前向计算，训练更新步数固定为 0。

**检查范围：** 原始 RGB 尺寸、裁剪后的张量尺寸、图像变化、有限数值输出、网络参数量、原路径实际点数、安装环境版本与 CUDA。此检查不训练网络、不预测已训练模型的归巢能力，也不统计成功率。正式全景/下视对照需要后续场景、相机阶段完成后，使用相同场景和评测条件开展。

**运行结果：** 3 个位姿的全景原图与裁剪图已生成。原图为 `1024×1024×3`，裁剪后张量为 `1×3×400×1024`；网络保持 10,820 个参数，输出为 `1×2` 且数值有限。标签分别为 `[-3,0]`、约 `[0,3]` 和 `[0,-3]`，均可恢复配置中的家位置。图像随位姿变化，裁剪内容与旧预处理逐像素一致。CUDA 运算通过，训练更新次数为 0。

首次渲染启动较慢，客户端返回超时；3 张检查图已实际输出，但最终报告留在 `in_progress`。该尝试不记作完整成功，原始记录保留在 [initial_capture_outcome.json](../visual_simulator_docker/CodeInsectInspired/baselines/stage01_20261022T111712Z/initial_capture_outcome.json)。随后在相同镜像中对已保存图像进行无 Kit、无重新渲染的独立复核，全部通过。检查工具已改为在 Kit 关闭之前写入测量报告；这一新的关闭顺序本次没有再次启动 Kit 验证。

最终复核证据：[verification/runtime_report.json](../visual_simulator_docker/CodeInsectInspired/baselines/stage01_20261022T111712Z/verification/runtime_report.json)。报告中的 `renderer_exercised=false` 表示**复核进程**不启动渲染器，`verified_existing_captures=true` 表示验证先前实际采集的图像，并非声称没有进行过渲染。[查看一张原全景裁剪检查图](../visual_simulator_docker/CodeInsectInspired/baselines/stage01_20261022T111712Z/smoke/panorama_00_crop.png)。

**发现的场景问题：** urban USD 有 17 个无法打开的资产引用，并缺少 `urban_sky.hdr`，涉及部分路灯、长椅等。可见画面能够输出，但场景资源不完整，不能直接作为资源完整的正式对照。本阶段只记录缺失清单，不下载替代模型、不修改原 USD；应在阶段 C 中补齐或建立资源完整的新场景。

实际运行版本：Isaac Sim `4.2.0-rc.18+release.16044.3b2ed111.gl`，Python `3.10.14`，PyTorch `2.4.0+cu118`，NumPy `1.26.0`，SciPy `1.10.1`，scikit-image `0.24.0`。AdamW 的实际默认 `weight_decay=0.01` 及其他参数已保存，后续不依赖记忆或库的新默认值。

路径函数实测生成 147 个位置，最大离家半径为 `10.995574 m`；按旧四相机调度只会输出 144 帧。完整捕获 147 点时旧划分公式得到 133/6/8，旧 144 帧则为 130/6/8。本次没有修复这些既有行为，而是把差异作为后续采样阶段的明确变更。

## 验证与复查命令

以下命令从 `Bee-Nav/` 根目录运行，只需 Python 标准库，不会启动仿真：

```bash
python3 visual_simulator_docker/CodeInsectInspired/tools/check_stage01_baseline.py
python3 -m unittest discover -s visual_simulator_docker/CodeInsectInspired/tests -p test_stage01_baseline.py -v
```

7 项回归检查覆盖：JSON 与入口覆盖优先级、源代码或协议参数偏离、实际动态步长、禁止把未实现实验标为可运行、不执行入口调用、拒绝覆盖已有记录。宿主 Python 3.8.10、Conda Python 3.9.25 和容器 Python 3.10.14 的静态检查均通过。新增工具和测试通过 Black 格式检查、Flake8 检查及 `git diff --check`，原业务代码没有整体格式化。

检查器仅针对当前旧入口的直接赋值与 bee 参数分支，不是通用 Python 配置解释器。后续重构采用新配置入口时应保留此旧基线，不能为了让检查通过而自动更新冻结协议。

若需要重复保存静态报告，指定一个新的输出目录；已有报告不覆盖：

```bash
python3 visual_simulator_docker/CodeInsectInspired/tools/check_stage01_baseline.py \
  --output visual_simulator_docker/CodeInsectInspired/baselines/<new_record_id>
```

GPU/渲染检查必须使用项目专用 Docker 和 Isaac Python。源码可只读挂载；输出目录单独以可写方式挂载，使用新的目录。当前三帧检查使用 `--network none`、`--gpus all`、只读源码和独立输出挂载。普通复查无需再渲染。

仅复核现有图像时，在具有相应挂载的容器中执行如下命令；`/baseline-output` 是显式挂载的本地基线目录：

```bash
cd /isaac-sim/CodeInsectInspired
../python.sh tools/check_stage01_baseline.py \
  --verify-captures /baseline-output/smoke \
  --output /baseline-output/<new_verification_id>
```

## 文件变化与当前状态

- 新增协议 JSON、基线检查工具、7 项回归测试和限定于新文件的 Black/Flake8 配置。
- 新增本阶段报告，更新总清单的 A 项与根 README 的复查命令。
- 更新 `.gitignore` 和 Docker 构建排除规则，隔离 `baselines/` 下的图像、日志和快照。
- 原采集、训练、网络、归巢控制代码及旧默认 JSON 均保持原样；既有权重修改保留。检查结束后，检查容器均已退出并自动删除。

下一阶段为 B：将统一实验配置接入采集、训练和评测，并拆分进程与产物目录。当前协议只负责冻结要求，不会自动改变旧入口的行为。
