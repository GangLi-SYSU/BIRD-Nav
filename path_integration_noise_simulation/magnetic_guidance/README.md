# Magnetic-guided path integration

This isolated CPU experiment extends `../determine_LHA_odometry.py`, the repository's
Fig. 2 path-integration simulation. Existing visual experiments and original PI
scripts remain unchanged. No network training, image rendering or GPU is required.

## Model and assumptions

As in the original implementation, the controller integrates nominal displacement;
actual movement differs by zero-mean Gaussian distance increments. Standard
deviation is `sigma * sqrt(step_length_m)`, so sigma has units m/√m, not percent.
This is an effective odometry/motion mismatch, not a full inertial mechanization.
It excludes persistent scale bias, wind, acceleration integration, magnetic
declination error and magnetic disturbances. The ideal compass gives heading in
the navigation frame every step, with zero noise, delay or initial alignment error.
It supplies direction, never position. The gyro-only control accumulates heading
noise at 0.63°/√s, inherited from the original robot configuration.

`distance_m` is **nominal outbound endpoint displacement**. Straight outbound length
is D. Dogleg headings 0°, 90°, 0°, −90° have lengths D/2 each, total 2D and endpoint
(D, 0). Both return nominally D. Random global bearings vary independently per trial.
Actual outbound radii can differ because of simulated error and are saved in CSV.
Return control ends at **estimated home**, including a fractional final step;
true endpoint distance ≤15 m is scored afterwards, never used to control stopping.
Original code instead stops within one nominal step of estimated home.
Gaussian signed negative increments are retained and counted, as in the original
model; large sigmas are stress tests, not calibrated physical sensor specifications.

## Run and inspect

From this directory, using the existing disk environment:

```bash
MPLCONFIGDIR=/tmp/bee-nav-pi /media/lg/lg-512/conda_envs/bird-nav/bin/python -B \
  -m unittest discover -s tests -v
MPLCONFIGDIR=/tmp/bee-nav-pi /media/lg/lg-512/conda_envs/bird-nav/bin/python -B \
  run_experiment.py --config config.json --output runs/magnetic_01
```

Dependencies: NumPy and Matplotlib (already installed). The runner refuses an
existing output directory. `config.json` controls distances, sigma levels, speeds,
time step, gyro coefficient, radius, seed and trial IDs. Extend a selected condition
with a new config/output and `trial_start: 50`; do not duplicate IDs 0–49.
Each combination has 50 independent trials; arms and sigma levels deliberately
share random draws for paired comparisons. Different distances use different draws.

Outputs: protocol snapshot and input hashes, per-trial CSV, grouped CSV with Wilson
95% intervals, all trajectories in NPZ (initial pose included), success heatmaps,
endpoint scatter, Chinese report and completion receipt. NPZ keys are `magnetic`,
`gyro_only`, `trial_ids`. Code uses four-space indentation, Black (100 columns) and
Flake8; tests verify fractional steps, compass correction, paired controls,
diffusion variance and stable trial indexing. Run `black --check .` and
`flake8 --jobs=1 .` when those developer tools are available.

The paper's Nature URL was inaccessible during this implementation. Provenance
therefore refers to the local published-code implementation, not a newly verified
reading of the paper text. Entering 15 m establishes only geometric eligibility
for the visual stage, not end-to-end navigation success.
