"""Latest-state continuation with an older validation-best candidate retained."""

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

from insect_utils.continuation_training import import_checkpoint, select_checkpoint, states_equal
from insect_utils.downward_training import seed_all, train_epochs
from insect_utils.epoch_extension import build_extension_plan, preserve_best, require_config
from insect_utils.experiment_config import digest
from insect_utils.experiment_state import read_json
from insect_utils.training_state import checkpoint_pointer


class StochasticSamples(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        value = torch.rand(2) + torch.tensor([np.random.uniform(), random.random()])
        return value, torch.stack((value.sum(), value[0] - value[1]))


class EpochExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.parent = PROJECT / "runs/stage15_rotation_300_01"
        cls.engine = cls.parent / "training/engine_scaled_units"
        cls.pointer = read_json(cls.engine / "checkpoint_latest.json")
        cls.source, _ = checkpoint_pointer(cls.engine, cls.pointer["fingerprint"])
        cls.best = next((cls.engine / "checkpoints").glob("epoch_0298_*.pt"))

    def test_only_the_chosen_epoch_budget_changes(self):
        original = read_json(self.parent / "training/contract.json")["configuration"]
        for target in (450, 600):
            require_config(original, dict(original, epochs=target), target)
            for key, value in (
                ("learning_rate", 0.0001),
                ("batch_size", 16),
                ("num_workers", 2),
                ("epochs", target + 1),
            ):
                changed = dict(original, epochs=target)
                changed[key] = value
                with self.assertRaises(ValueError):
                    require_config(original, changed, target)
        for target in (300, 451):
            with self.assertRaises(ValueError):
                require_config(original, dict(original, epochs=target), target)

    def test_both_proposed_plans_preserve_data_controller_and_all64_trials(self):
        parent = read_json(self.parent / "frozen.json")
        navigation = read_json(self.parent / "navigation/frozen.json")
        for target in (450, 600):
            with self.subTest(target=target):
                plan = build_extension_plan(target)
                require_config(parent["training_config"], plan["training_config"], target)
                self.assertEqual(plan["episodes"], navigation["episodes"])
                self.assertEqual(plan["budget"], navigation["budget"])
                self.assertEqual(plan["navigation_protocol"], navigation["protocol"])
                self.assertEqual(plan["preprocessing"], parent["preprocessing"])
                self.assertEqual(plan["historical_best_epoch"], 298)
                self.assertEqual(plan["protocol"]["target_optimizer_steps"], target * 512)
                dataset = "runs/stage11_coverage_02/dataset.json"
                self.assertEqual(plan["files"][dataset], parent["files"][dataset])

    def test_latest300_restores_all_states_and_best298_does_not_move_resume_pointer(self):
        source_hash, best_hash = digest(self.source), digest(self.best)
        source = torch.load(self.source, map_location="cpu", weights_only=True)
        with tempfile.TemporaryDirectory() as temporary:
            engine = Path(temporary) / "engine"
            receipt = import_checkpoint(
                self.source, engine, "child", self.pointer["fingerprint"], 300, 153600
            )
            self.assertTrue(receipt["all_payload_except_experiment_fingerprint_exactly_equal"])
            pointer_hash = digest(engine / "checkpoint_latest.json")
            best_receipt = preserve_best(
                self.best, engine, "child", self.pointer["fingerprint"], 298
            )
            self.assertTrue(best_receipt["used_for_validation_selection_only"])
            self.assertEqual(digest(engine / "checkpoint_latest.json"), pointer_hash)
            path, pointer = checkpoint_pointer(engine, "child")
            imported = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(pointer["epoch"], 300)
            self.assertEqual(pointer["optimizer_steps"], 153600)
            for key in source:
                if key != "fingerprint":
                    self.assertTrue(states_equal(source[key], imported[key]), key)
            best, state = select_checkpoint(engine, "child", imported)
            self.assertEqual(best["epoch"], 298)
            weights = torch.load(
                self.parent / "training/best_validation.pth", map_location="cpu", weights_only=True
            )
            self.assertTrue(states_equal(state, weights))
            repeated = preserve_best(self.best, engine, "child", self.pointer["fingerprint"], 298)
            self.assertEqual(repeated, best_receipt)
            self.assertEqual(digest(engine / "checkpoint_latest.json"), pointer_hash)
        self.assertEqual(digest(self.source), source_hash)
        self.assertEqual(digest(self.best), best_hash)

    def test_wrong_source_identity_epoch_or_optimizer_count_is_rejected(self):
        for fingerprint, epoch, steps in (
            ("wrong", 300, 153600),
            (self.pointer["fingerprint"], 298, 153600),
            (self.pointer["fingerprint"], 300, 0),
        ):
            with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
                import_checkpoint(
                    self.source, Path(temporary) / "engine", "child", fingerprint, epoch, steps
                )
        for fingerprint, epoch in (("wrong", 298), (self.pointer["fingerprint"], 300)):
            with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
                preserve_best(self.best, Path(temporary) / "engine", "child", fingerprint, epoch)

    def test_conflicting_historical_candidate_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            engine = Path(temporary) / "engine"
            preserve_best(self.best, engine, "child", self.pointer["fingerprint"], 298)
            path = engine / "checkpoints/epoch_0298_historical_best.pt"
            changed = torch.load(path, map_location="cpu", weights_only=True)
            changed["optimizer_steps"] += 1
            torch.save(changed, path)
            with self.assertRaises(ValueError):
                preserve_best(self.best, engine, "child", self.pointer["fingerprint"], 298)

    def test_continuation_matches_uninterrupted_optimizer_weights_and_all_rng(self):
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
            train_epochs(
                model,
                optimizer,
                *loaders,
                dict(epochs=epochs, validation_interval_updates=0),
                engine,
                fingerprint,
                resume=resume,
            )
            path, _ = checkpoint_pointer(engine, fingerprint)
            return torch.load(path, map_location="cpu", weights_only=True)

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            run(base / "parent", "parent", 3)
            source, _ = checkpoint_pointer(base / "parent", "parent")
            for target in (5, 6):
                with self.subTest(target=target):
                    full = run(base / str(target) / "full", "full", target)
                    child = base / str(target) / "child"
                    import_checkpoint(source, child, "child", "parent", 3, 12)
                    continued = run(child, "child", target, resume=True)
                    for key in ("model", "optimizer", "rng"):
                        self.assertTrue(states_equal(full[key], continued[key]), key)
                    self.assertEqual(continued["optimizer_steps"], target * 4)


if __name__ == "__main__":
    unittest.main()
