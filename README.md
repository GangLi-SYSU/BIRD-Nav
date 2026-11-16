# BIRD-Nav

BIRD-Nav 基于 [TU Delft Bee-Nav](https://github.com/tudelft/Bee-Nav)，增加了理想地磁指导的路径积分、
下视相机视觉归巢，以及可手动外出的两阶段 Isaac Sim 演示。下文保留原项目与论文的说明；
当前扩展实验请优先阅读本节。

## 实验室复现与大文件获取

克隆代码后进入仓库目录：

```bash
git clone https://github.com/GangLi-SYSU/BIRD-Nav.git
cd BIRD-Nav
```

完整安装、素材下载、校验、场景重建和训练命令见
[复现资源与操作指南](docs/REPRODUCTION_RESOURCES.md)。
[大文件清单](docs/GITHUB_LARGE_FILES.md)记录了本机历史资源的体积。

| 内容 | 提供方式 |
| --- | --- |
| 代码、协议配置、测试、鸟模型与参考网络 | 随 Git 提供；参考网络为 Stage23 验证集选出的 epoch 484 |
| Isaac Sim 4.2.0、原项目树木/地面素材、四种房屋 | 从指南列出的原始来源下载，保留完整纹理与许可文件 |
| 当前真实建筑场景、1,176 张训练/验证/预留测试图像 | 按指南重新生成；目前未发布预生成场景或历史图像的下载包 |
| Docker 工作磁盘、镜像归档、缓存、历史训练状态与导航截图 | 不纳入 Git；通过安装与实验脚本重新生成 |

当前视觉协议为 **40 m 高度、30° 方形下视视场、384×384 网络输入、20 m 内密集螺旋、
圆形掩膜与随机旋转、600 轮训练、2 m 成功半径**。没有加入近家网格补充训练数据。
原始 RGB 为 1024×1024；模型导出包含缩放、掩膜和米制输出换算。

完成指南中的环境与场景准备后，在仓库根目录运行：

```bash
# 重新采集、训练并测试 10 m / 20 m 起点；先关闭占用 GPU 的 demo。
bash visual_simulator_docker/run_compact_spiral_on_disk.sh \
  --run-dir runs/reproduce_compact20_01

# 使用随仓库提供的参考网络启动交互演示，无需下载旧 runs/。
bash visual_simulator_docker/run_interactive_homing_demo.sh
```

演示按 `T` 起飞，`W/A/S/D` 外出，`Q/E` 转向，`H` 开始两阶段归巢，空格暂停，`R` 重置。
落地后显示路径积分对照和视觉归巢轨迹。第一阶段的理想地磁只提供方向，
第二阶段的 2 m 到达判定使用仿真真值，当前控制属于运动学演示。

CPU 路径积分批量实验另见
[magnetic_guidance/README.md](path_integration_noise_simulation/magnetic_guidance/README.md)；
不需要 Isaac、场景素材或训练图像。历史阶段报告中的本机 `runs/` 路径不是克隆后附带的数据。

## Upstream: Efficient robot navigation inspired by honeybee learning flights

This repository contains all code for the paper:

> **Efficient robot navigation inspired by honeybee learning flights**
> D. Ou¹, J. J. Hagenaars¹, M. Jankowski¹, M. V. M. Firlefyn¹, C. De Wagter¹, F. T. Muijres², J. Degen³, G. C. H. E. de Croon¹
> *Nature*, 2026.
> DOI: [10.1038/s41586-026-10461-3](https://www.nature.com/articles/s41586-026-10461-3)

Affiliations:

1. MAVLab, Control and Operations department, Faculty of Aerospace Engineering, Delft University of Technology (TU Delft), the Netherlands.
2. Experimental Zoology Group, Wageningen University, the Netherlands.
3. Navigation Biology Group, Institute of Biology and Environmental Sciences, Carl von Ossietzky University of Oldenburg, Germany.

## Repository Structure

| Folder | Description | Paper Reference |
|--------|-------------|-----------------|
| `code_theoretical_simulation_analysis/` | Theoretical analysis and ANN simulations | Supplementary Information 1–5 |
| `path_integration_noise_simulation/` | Path integration noise modelling | Figure 2 a,b |
| `visual_simulator_docker/` | Isaac Sim 4.2 visual homing experiments (Docker) | Figures 2 c,d,e |
| `robot_network_training/` | Offline network training (laptop and Raspberry Pi) | Figure 3,4 |
| `robot_onboard/` | Onboard drone code (ROS 2 flight control + camera) | Figure 3,4 |

## Quick Start

Each folder contains its own `README.md` with setup instructions. In brief:

1. **Theoretical analysis** — `cd code_theoretical_simulation_analysis && pip install -r requirements.txt && python run_all_experiments.py`
2. **Path integration** — `cd path_integration_noise_simulation && pip install -r requirements.txt && python run_path_integration_experiments.py`
3. **Visual simulator** — Requires Docker + NVIDIA GPU. See `visual_simulator_docker/README.md` for setup.
4. **Robot network training** — `cd robot_network_training/home-learning_laptop` and follow `README.md`.
5. **Robot onboard** — Requires Raspberry Pi 4 + PX4 flight controller. See `robot_onboard/README.md`.

## External Data

- **Isaac Sim 3D assets** (~2.4 GB): Download from [SURFdrive](https://surfdrive.surf.nl/s/qardanQkDLwe7m2) and place in `visual_simulator_docker/isaac-sim_assets/`.
- **Robot training data**: Download from the link in `robot_network_training/README.md`.

## Downward-camera experiment development

The [matched-update near-home fine tuning](docs/STAGE_20_NEAR_FINETUNE.md) compares
original and supplemented data from the same complete epoch-589 checkpoint,
then repeats 64 homing trials at each of 5 m and 10 m for both selected models.

The [symmetric near-home supplement](docs/STAGE_19_NEAR_DATA.md) adds 193 training
positions and 16 validation positions within 5 m, retaining all old training and
validation images. Another 18 local positions remain reserved and uncaptured.

The [local multi-heading probe](docs/STAGE_18_LOCAL_PROBE.md) captures 104 stationary
observations at 13 positions around home to diagnose the epoch-589 model's local
bias and heading consistency. No training or navigation is added.

The [5 m and 10 m navigation tests](docs/STAGE_17_RADIUS_NAVIGATION.md) evaluate
the same validation-selected epoch-589 model in 64 trials per starting radius,
with the original dynamic steps, 0.5 m success circle and 50-step limit.

The [extension to 600 epochs](docs/STAGE_16_EPOCH_EXTENSION.md) adds 300 epochs,
preserving the complete epoch-300 resume state and the historical epoch-298
validation candidate, then repeats the same 64 trials at 20 m.

The [150-to-300 epoch continuation](docs/STAGE_15_ROTATION_CONTINUATION.md) restores
the augmentation arm's complete epoch-150 training state, adds 150 epochs with
unchanged settings, and automatically repeats the same 64 trials at 20 m.

The [20 m rotation comparison](docs/STAGE_14_ROTATION_R20_COMPARISON.md) trains both
circular-input controls and runs 64 matched 20 m homing trials per arm, with the
same 0.5 m success radius and 50-step controller. Other initial radii are excluded.

The [stage 13 circular-input protocol](docs/STAGE_13_CIRCULAR_AUGMENTATION.md)
adds online nadir-image rotation with coupled body-vector labels and a fixed
circular mask shared by training and exported inference. It reuses the stage 11
position splits in separate runs; stage 12 remains the historical navigation baseline.

The [stage 12 navigation protocol](docs/STAGE_12_NAVIGATION_TEST.md) tests the frozen
stage 11 validation-selected model with a 0.5 m horizontal success radius. It runs
oracle and live-camera/model checks, 48 pilot trials, then 320 fixed navigation
trials. This is discrete kinematic navigation; it does not simulate quadrotor
braking, hovering, or landing. Runtime state is recorded in
`visual_simulator_docker/CodeInsectInspired/runs/stage12_navigation_01/progress.json`.
The [stage 9 scene preview](docs/STAGE_09_REAL_SCENE_PREVIEW.md) adds a separate
normal-scale house/tree/pool scene with a 40 m AGL, 70° square nadir camera and
60°/80° comparison views. It is a visual preview; no new training is started.
The [stage 11 coverage and normalization protocol](docs/STAGE_11_COVERAGE_NORMALIZATION.md)
defines the 600-position expansion, preserved historical test images, balanced-distance
validation/new test partitions, and paired meter/normalized-target training controls.
The [stage 10 training report](docs/STAGE_10_REAL_SCENE_TRAINING.md) records the
completed 1,176-image capture at a 44 m learning radius and 150 fresh training
epochs. Mean test angle error is 14.06°; mean vector error is 11.20 m, or 25.46%
of the learning radius. Training, validation and test remain grouped by position;
closed-loop homing has not been evaluated.

See the [implementation plan](docs/DOWNWARD_CAMERA_VISUAL_HOMING_PLAN.md) and
[stage 1 report](docs/STAGE_01_BASELINE_REPORT.md). The
[stage 2 report](docs/STAGE_02_PIPELINE_REPORT.md) documents unified configuration,
isolated capture/train/evaluate processes and artifact-verified resume. The
[stage 3 report](docs/STAGE_03_SCENE_REPORT.md) documents three independent 100 m
flat scenes, complete local assets, spatial budgets and the Isaac preview. The
downward camera geometry and both camera endpoints are implemented and calibrated;
[stage 4](docs/STAGE_04_CAMERA_REPORT.md) has passed geometry and settled-image
checks on both the calibration board and the real landmark scene, with 72 tests
passing.
The [stage 5 report](docs/STAGE_05_CAPTURE_REPORT.md) describes exact sample
scheduling, measured-pose labels and footprint diagnostics. The user selected
eight real headings at each of 147 positions (1176 images). GPU capture acceptance
passed for single- and four-camera captures and the five-frame tail probe, with 95
tests passing in both host and container environments.
The [stage 6 report](docs/STAGE_06_DATA_REPORT.md) documents the user-approved
position-group split (1064/48/64 images), shared full-frame RGB preprocessing and
deterministic validation/test loading. All 1176 images passed pixel checks and all
117 tests passed in the container.
The [stage 7 report](docs/STAGE_07_TRAINING_REPORT.md) documents isolated loader
probes, three-epoch smoke training, epoch checkpoints and final-model delivery.
All 130 tests passed in the container, including GPU checkpoint restoration.
The fresh 150-epoch run completed 19,950 updates under
`runs/stage07_training/attention_8h_01/`; final PyTorch reload and ONNX reference
checks passed. Prediction quality shows substantial overfitting: mean angle error
is 7.01° on training images and about 91.7° on validation/test images. These are
within-region offline checks; reliable closed-loop homing is not established and
remains stage H.
The [stage 8 optimization report](docs/STAGE_08_OPTIMIZATION_REPORT.md) records
the user-approved larger, denser landmarks and the square 384×384 attention
variant (10,628 parameters). Scene geometry, preview and all 137 container tests
passed. The new 1,176-image capture, data acceptance and 150-epoch training run
completed 19,950 formal updates. Mean test vector error improved from 9.62 to
7.17 m, and mean test direction error from 91.68° to 64.31°. Overfitting and large
errors remain; closed-loop homing has not run. The original stage 7 data, model
and reports are preserved.
[The landmark scene preview](docs/LANDMARK_SCENE_PREVIEW.md) adds a 400 m field
with 192 colored cubes, cylinders, cones and spheres for the next visual inspection.
Planning and geometry tests do not start Isaac Sim. PNG/CSV fixture tests use
NumPy and Pillow and are skipped when those packages are absent. Eleven runtime
tests additionally require torch/torchvision; GPU restoration requires CUDA. They
run in the existing container:

```bash
python3 visual_simulator_docker/CodeInsectInspired/tools/check_stage01_baseline.py
python3 visual_simulator_docker/CodeInsectInspired/run_downward_homing.py --stage plan
python3 -m unittest discover -s visual_simulator_docker/CodeInsectInspired/tests -p 'test_*.py' -v
```

New Python files use four-space indentation and Python 3.8-compatible syntax.
Black and Flake8 configuration is scoped to the new experiment files. When these
development tools are installed, run the following from the repository root:

```bash
cd visual_simulator_docker/CodeInsectInspired
black --check --config pyproject.toml \
  run_downward_homing.py run_downward_training.py \
  generate_downward_environment.py generate_landmark_environment.py \
  insect_utils/experiment_*.py insect_utils/scene_geometry.py \
  insect_utils/camera_geometry.py insect_utils/downward_camera.py insect_utils/frame_stability.py \
  insect_utils/capture_geometry.py insect_utils/downward_capture.py \
  insect_utils/rgb_preprocessing.py insect_utils/data_preparation.py \
  insect_utils/downward_data.py insect_utils/data_diagnostics.py \
  insect_utils/downward_training.py insect_utils/training_state.py \
  tools/*stage0*.py tools/check_landmark_scene.py tests/test_*.py
flake8 --jobs=1 --config .flake8 \
  run_downward_homing.py run_downward_training.py \
  generate_downward_environment.py generate_landmark_environment.py \
  insect_utils/experiment_*.py insect_utils/scene_geometry.py \
  insect_utils/camera_geometry.py insect_utils/downward_camera.py insect_utils/frame_stability.py \
  insect_utils/capture_geometry.py insect_utils/downward_capture.py \
  insect_utils/rgb_preprocessing.py insect_utils/data_preparation.py \
  insect_utils/downward_data.py insect_utils/data_diagnostics.py \
  insect_utils/downward_training.py insect_utils/training_state.py \
  tools/*stage0*.py tools/check_landmark_scene.py tests/test_*.py
```

Runtime/render checks use Isaac Sim's own Python inside the existing disk-backed
Docker environment; instructions are in the stage report.

From a desktop terminal, browse the generated terrain without starting an experiment:

```bash
cd visual_simulator_docker
bash view_downward_scene_on_disk.sh
# Larger field with geometric landmarks:
bash view_downward_scene_on_disk.sh --scene maps/downward_landmarks_v1/landmarks.usda
```

The scene generator needs NumPy and Pillow with `Image.Resampling` support. It
refuses existing output directories; generated scene bundles live on the project
disk and are excluded from Git. See the stage 3 report for regeneration commands.

## Citation

```bibtex
@article{ou2026efficient,
  title   = {Efficient robot navigation inspired by honeybee learning flights},
  author  = {Ou, D. and Hagenaars, J. J. and Jankowski, M. and Firlefyn, M. V. M. and De Wagter, C. and Muijres, F. T. and Degen, J. and de Croon, G. C. H. E.},
  journal = {Nature},
  year    = {2026},
  doi     = {10.1038/s41586-026-10461-3}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
