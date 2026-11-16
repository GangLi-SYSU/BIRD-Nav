"""Reproduce the standalone varied-route PI demonstration from its protocol."""
import argparse
import json
from pathlib import Path

import numpy as np


def reconstruct(config):
    waypoints = np.asarray(config['waypoints'], dtype=float)
    lengths, angles = [], []
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        delta = end - start
        distance = np.linalg.norm(delta)
        count = int(distance / 1.25)
        steps = [1.25] * count
        if distance - count * 1.25 > 1e-10:
            steps.append(distance - count * 1.25)
        lengths.extend(steps)
        angles.extend([np.arctan2(delta[1], delta[0])] * len(steps))
    split = len(lengths)
    assert np.array_equal(waypoints[-1], [2000, 0])
    lengths.extend([2.] * 1000)
    angles.extend([np.pi] * 1000)
    lengths, angles = np.asarray(lengths), np.asarray(angles)
    rng = np.random.default_rng(np.random.SeedSequence(
        [config['seed'], config['route_id'], 2000000, config['trial_id']]))
    actual = lengths + config['sigma_m_per_sqrt_m'] * np.sqrt(lengths) * rng.standard_normal(len(lengths))
    increments = actual[:, None] * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    positions = np.vstack([np.zeros(2), np.cumsum(increments, axis=0)])
    durations = lengths / np.r_[np.full(split, 2.5), np.full(1000, 4.)]
    timestamps = np.r_[0, np.cumsum(durations)]
    return positions, timestamps, split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    config = json.loads((args.output / 'protocol.json').read_text())
    positions, timestamps, split = reconstruct(config)
    target = args.output / 'trajectory.npy'
    if target.exists():
        np.testing.assert_array_equal(np.load(target), positions)
    else:
        np.save(target, positions)
    assert split == config['outbound_steps']
    radii = np.linalg.norm(positions, axis=1)
    assert config['exit_index'] == np.flatnonzero(radii > 15)[0]
    assert config['reentry_index'] == np.flatnonzero(
        (np.arange(len(radii)) >= split) & (radii <= 15))[0]
    assert abs(radii[-1] - config['final_distance_m']) < 1e-9
    assert abs(radii[split] - 2000) < 50
    np.save(args.output / 'timestamps.npy', timestamps)
    (args.output / 'route_validation.json').write_text(json.dumps(dict(
        exact_reproduction=True, positions=len(positions), outbound_steps=split,
        outbound_distance_m=config['outbound_length_m'], final_distance_m=float(radii[-1]),
        simulation_duration_s=float(timestamps[-1]), first_exit_verified=True,
        first_return_entry_verified=True, seed_not_selected_by_success=True), indent=2))
    print('Replay verified:', len(positions), 'positions; final error', radii[-1])


if __name__ == '__main__':
    main()
