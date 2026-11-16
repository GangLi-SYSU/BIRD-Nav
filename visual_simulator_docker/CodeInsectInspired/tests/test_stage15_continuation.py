"""Checkpoint lineage transfer, epoch-only configuration and full-state continuation."""

import copy
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.continuation_protocol import require_epoch_only_change
from insect_utils.continuation_training import (
    import_checkpoint,
    require_runtime,
    select_checkpoint,
    states_equal,
)
from insect_utils.downward_training import seed_all, train_epochs
from insect_utils.experiment_config import digest
from insect_utils.experiment_state import read_json
from insect_utils.training_state import checkpoint_pointer, save_checkpoint


class StochasticSamples(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        value = torch.rand(2) + torch.tensor([np.random.uniform(), random.random()])
        return value, torch.stack((value.sum(), value[0] - value[1]))


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_only_epoch_budget_change_is_accepted(self):
        original = read_json(PROJECT / "runs/stage13_circular_01/arms/mask_rotation/contract.json")[
            "configuration"
        ]
        require_epoch_only_change(original, dict(original, epochs=300))
        for key, value in (
            ("learning_rate", 0.0001),
            ("batch_size", 16),
            ("num_workers", 2),
            ("epochs", 301),
        ):
            changed = dict(original, epochs=300)
            changed[key] = value
            with self.assertRaises(ValueError):
                require_epoch_only_change(original, changed)

    def test_actual150_checkpoint_transfer_preserves_every_training_state(self):
        engine = PROJECT / "runs/stage13_circular_01/arms/mask_rotation/engine_scaled_units"
        original = read_json(engine / "checkpoint_latest.json")
        source, _ = checkpoint_pointer(engine, original["fingerprint"])
        before = digest(source)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "engine"
            receipt = import_checkpoint(
                source, destination, "new-lineage-fixture", original["fingerprint"], 150, 76800
            )
            self.assertTrue(receipt["all_payload_except_experiment_fingerprint_exactly_equal"])
            self.assertEqual(receipt["imported_optimizer_steps"], 76800)
            self.assertEqual(digest(source), before)
            other = import_checkpoint(
                source, destination, "new-lineage-fixture", original["fingerprint"], 150, 76800
            )
            self.assertEqual(other, receipt)

    def test_wrong_parent_identity_epoch_or_update_budget_is_rejected(self):
        engine = PROJECT / "runs/stage13_circular_01/arms/mask_rotation/engine_scaled_units"
        original = read_json(engine / "checkpoint_latest.json")
        source, _ = checkpoint_pointer(engine, original["fingerprint"])
        for fingerprint, epoch, steps in (
            ("wrong", 150, 76800),
            (original["fingerprint"], 149, 76800),
            (original["fingerprint"], 150, 0),
        ):
            with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
                import_checkpoint(
                    source, Path(temporary) / "engine", "new", fingerprint, epoch, steps
                )

    def test_rebound_resume_matches_uninterrupted_weights_optimizer_and_rng(self):
        def run(engine, fingerprint, epochs, resume=False):
            seed_all(10)
            model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2))
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
            loaders = [
                DataLoader(
                    StochasticSamples(),
                    batch_size=2,
                    shuffle=True,
                    generator=torch.Generator().manual_seed(10),
                )
                for _ in range(2)
            ]
            config = dict(epochs=epochs, validation_interval_updates=0)
            train_epochs(model, optimizer, *loaders, config, engine, fingerprint, resume=resume)
            path, _ = checkpoint_pointer(engine, fingerprint)
            return torch.load(path, map_location="cpu", weights_only=True)

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            full = run(base / "full", "full", 4)
            run(base / "parent", "parent", 2)
            source, _ = checkpoint_pointer(base / "parent", "parent")
            import_checkpoint(source, base / "child", "child", "parent", 2, 8)
            continued = run(base / "child", "child", 4, resume=True)
        for key in ("model", "optimizer", "rng"):
            self.assertTrue(states_equal(full[key], continued[key]), key)
        self.assertEqual(continued["optimizer_steps"], 16)

    def test_validation_selection_can_retain_imported_parent_epoch(self):
        rows = [
            dict(epoch=1, validation_component_mse_m2=2),
            dict(epoch=2, validation_component_mse_m2=1),
        ]
        payload = dict(
            epoch=2, optimizer_steps=8, model={"weight": torch.tensor([3.0])}, history=rows
        )
        final = dict(history=rows + [dict(epoch=3, validation_component_mse_m2=1.5)])
        with tempfile.TemporaryDirectory() as temporary:
            engine = Path(temporary)
            save_checkpoint(engine, payload, "child")
            best, state = select_checkpoint(engine, "child", final)
        self.assertEqual(best["epoch"], 2)
        self.assertTrue(torch.equal(state["weight"], payload["model"]["weight"]))

    def test_runtime_changes_are_not_silently_accepted(self):
        original = read_json(PROJECT / "runs/stage13_circular_01/arms/mask_rotation/contract.json")[
            "runtime"
        ]
        require_runtime(original, original)
        changed = copy.deepcopy(original)
        changed["cudnn_allow_tf32"] = not original["cudnn_allow_tf32"]
        with self.assertRaises(ValueError):
            require_runtime(changed, original)


if __name__ == "__main__":
    unittest.main()
