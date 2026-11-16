# 视觉归巢参考模型

本目录随代码提供约 1.32 MiB 的模型文件，可独立加载，不需要训练输出目录。
参考模型为验证集误差最低的第 484 轮；文件哈希记录在 `SHA256SUMS`。

- `best_validation.pth`：PyTorch 权重。
- `home_vector_m.torchscript.pt`：演示使用的推理模型，输出机体坐标系中的米制归巢向量。
- `home_vector_m.onnx`：对应的 ONNX 推理模型。

运行配置位于 `config/experiments/demo_release.json`。
相机离地 40 m，30° 方形视场，原始 RGB 为 1024×1024；
导出模型内部执行 384×384 尺寸调整、圆形掩膜和 44 m 输出尺度换算。
训练使用 20 m 半径内 147 个螺旋位置、八个航向、随机种子 10 和 600 轮训练。

在本目录运行 `sha256sum -c SHA256SUMS` 核对文件。
重新采集或使用不同 GPU/依赖版本训练时，最佳轮次与导航指标可能变化。
完整素材与数据生成命令见仓库根目录的 `docs/REPRODUCTION_RESOURCES.md`。
