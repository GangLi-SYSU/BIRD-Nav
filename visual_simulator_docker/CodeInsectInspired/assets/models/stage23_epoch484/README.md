# Reference visual-homing model

This small bundle is included with the code; no historical `runs/` directory is needed
to load the reference model. Files are copied byte-for-byte from Stage23's selected
epoch 484. `SHA256SUMS` records their hashes.

- `best_validation.pth`: PyTorch weights for training/evaluation comparisons.
- `home_vector_m.torchscript.pt`: demo inference; physical body-frame vector in meters.
- `home_vector_m.onnx`: equivalent exported inference graph.

The associated runtime configuration is `config/experiments/demo_release.json`.
The camera is nadir at 40 m AGL with 30° square FOV and raw 1024×1024 RGB.
The model owns 384×384 resizing, the circular mask and the 44 m output scaling.
Training used 147 spiral positions × 8 headings within a 20 m radius, seed 10,
600 epochs and validation-based checkpoint selection.

From this directory, verify all files with `sha256sum -c SHA256SUMS`.
These weights are a reference result; fresh captures and different GPU/library
versions can produce different validation-selected epochs and navigation results.
See `docs/REPRODUCTION_RESOURCES.md` at repository root for external scene/data inputs.
