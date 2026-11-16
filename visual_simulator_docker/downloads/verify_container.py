"""Check project dependencies, bundled inputs, and GPU execution without a simulation."""
import importlib
import json
from pathlib import Path
import sys

modules = ["numpy", "scipy", "matplotlib", "pandas", "tqdm", "skimage", "onnx", "torch", "torchvision", "osqp"]
versions = {}
for name in modules:
    module = importlib.import_module(name)
    versions[name] = getattr(module, "__version__", "unknown")

import numpy as np
import torch

assert versions["scipy"] == "1.10.1"
assert versions["skimage"] == "0.24.0"

# Exercise the NumPy/PyTorch ABI and an actual CUDA operation, not just imports.
assert torch.from_numpy(np.ones(3, dtype=np.float32)).numpy().sum() == 3
assert torch.cuda.is_available(), "CUDA is unavailable inside the container"
assert torch.ones(3, device="cuda").sum().item() == 3

project = Path("/isaac-sim/CodeInsectInspired")
assets = Path("/isaac-sim/isaac-sim_assets")
assert (project / "run_ANN_homing_simulations.py").is_file()
assert (project / "maps/env3_clean.usd").is_file()
assert (assets / "noon_grass_4k.hdr").is_file()
trees = list((assets / "Trees").rglob("*.usd"))
assert len(trees) == 44, f"Unexpected tree count: {len(trees)}"

print(json.dumps({
    "python": sys.version.split()[0], "packages": versions,
    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
    "tree_models": len(trees), "numpy_torch_interop": True,
    "cuda_operation": True, "project_inputs": True,
}, indent=2))
