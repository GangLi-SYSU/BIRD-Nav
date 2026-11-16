"""Source-state identity, equal update sampling, shared validation and exact CPU resume."""

import copy
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.continuation_training import states_equal
from insect_utils.downward_training import seed_all, train_epochs
from insect_utils.experiment_config import digest
from insect_utils.experiment_state import read_json
from insect_utils.near_finetune_engine import block_loader, train_blocks
from insect_utils.near_finetune_training import select_state, validation_metrics
from insect_utils.training_state import checkpoint_pointer, save_checkpoint


class StochasticSamples(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        x = torch.tensor([index / 12, (index + 1) / 12]) + torch.rand(2) * 0.01
        x = x + float(np.random.uniform(0, 0.01)) + random.random() * 0.01
        return x, torch.stack((x.sum(), x[0] - x[1]))


class FineTuneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actual589_source_matches_selected_weights_and_optimizer_counter(self):
        spec = read_json(PROJECT / "config/experiments/stage20_near_finetune_protocol.json")
        source = PROJECT / spec["source_checkpoint"]
        self.assertEqual(digest(source), spec["source_checkpoint_sha256"])
        value = torch.load(source, map_location="cpu", weights_only=True)
        selected = torch.load(
            PROJECT / spec["source_training_run"] / "training/best_validation.pth",
            map_location="cpu",
            weights_only=True,
        )
        self.assertTrue(states_equal(value["model"], selected))
        self.assertEqual(value["epoch"], 589)
        self.assertEqual(value["optimizer_steps"], 301568)
        self.assertTrue(
            all(float(s["step"]) == 301568 for s in value["optimizer"]["state"].values())
        )

    def test_equal_size_distinct_subsets_and_shared_worker_seeds(self):
        cfg = dict(updates_per_block=4, batch_size=2, num_workers=0, seed=10)
        a = block_loader(list(range(8)), cfg, 1)
        b = block_loader(list(range(12)), cfg, 1)
        for loader in (a, b):
            self.assertEqual(len(loader), 4)
            self.assertEqual(len(set(loader.dataset.indices)), 8)
        self.assertEqual(a.generator.initial_seed(), b.generator.initial_seed())
        self.assertEqual(b.dataset.indices, block_loader(list(range(12)), cfg, 1).dataset.indices)
        self.assertNotEqual(
            b.dataset.indices, block_loader(list(range(12)), cfg, 2).dataset.indices
        )

    def test_common_validation_bands_are_macro_averaged_in_physical_units(self):
        targets = torch.tensor([[0.25, 0], [1, 0], [3, 0], [10, 0], [10, 0]])
        loader = DataLoader(TensorDataset(torch.zeros(5, 2), targets / 44), batch_size=2)
        model = nn.Linear(2, 2)
        nn.init.zeros_(model.weight)
        nn.init.zeros_(model.bias)
        result = validation_metrics(model, loader)
        self.assertEqual([v["images"] for v in result["bands"].values()], [1, 1, 1, 2])
        expected = (0.25 ** 2 / 2 + 1 / 2 + 9 / 2 + 100 / 2) / 4
        self.assertAlmostEqual(result["selection_score"], expected, places=4)
        self.assertNotAlmostEqual(result["selection_score"], result["component_mse_m2"], places=2)

    def test_original_control_and_supplement_share_all448_validation_images(self):
        records = read_json(PROJECT / "runs/stage19_near_data_01/dataset.json")["records"]
        validation = [r["image"] for r in records if r["split"] == "validation"]
        self.assertEqual(len(validation), 448)
        self.assertEqual(
            sum(r["split"] == "train" and r["origin"] == "original" for r in records), 4096
        )
        self.assertEqual(sum(r["split"] == "train" for r in records), 5640)
        self.assertFalse(any("test" in r["split"] for r in records))

    def test_full_state_block_resume_matches_uninterrupted_training(self):
        def evaluate(model, loader):
            mode = model.training
            model.eval()
            with torch.inference_mode():
                values = [float(torch.nn.functional.mse_loss(model(x), y)) for x, y in loader]
            model.train(mode)
            return dict(
                component_mse_m2=sum(values) / len(values),
                selection_score=sum(values) / len(values),
            )

        cfg = dict(blocks=2, updates_per_block=4, batch_size=2, num_workers=0, seed=10, scale_m=1)

        def model_opt():
            model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2))
            return model, torch.optim.AdamW(model.parameters(), lr=0.0005)

        def loaders():
            return [
                DataLoader(
                    StochasticSamples(),
                    batch_size=3,
                    shuffle=False,
                    generator=torch.Generator().manual_seed(10),
                )
                for _ in range(2)
            ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed_all(10)
            model, opt = model_opt()
            train, val = loaders()
            train_epochs(
                model,
                opt,
                train,
                val,
                dict(epochs=1, validation_interval_updates=0),
                root / "source",
                "source",
            )
            path, _ = checkpoint_pointer(root / "source", "source")
            origin = torch.load(path, map_location="cpu", weights_only=True)
            results = []
            for name, interrupted in (("full", False), ("split", True)):
                model, opt = model_opt()
                _, val = loaders()
                train_blocks(
                    model,
                    opt,
                    StochasticSamples(),
                    val,
                    cfg,
                    root / name,
                    name,
                    copy.deepcopy(origin),
                    evaluate,
                    stop_after_block=1 if interrupted else None,
                )
                if interrupted:
                    model, opt = model_opt()
                    _, val = loaders()
                    train_blocks(
                        model,
                        opt,
                        StochasticSamples(),
                        val,
                        cfg,
                        root / name,
                        name,
                        copy.deepcopy(origin),
                        evaluate,
                        resume=True,
                    )
                path, _ = checkpoint_pointer(root / name, name)
                results.append(torch.load(path, map_location="cpu", weights_only=True))
            for key in ("model", "optimizer", "rng"):
                self.assertTrue(states_equal(results[0][key], results[1][key]), key)
            self.assertEqual(results[1]["optimizer_steps"], 8)
            self.assertEqual(results[1]["total_optimizer_steps"], 12)
            self.assertTrue(
                all(float(s["step"]) == 12 for s in results[1]["optimizer"]["state"].values())
            )

    def test_initial_source_model_can_remain_validation_best(self):
        rows = [dict(block=0, selection_score=1), dict(block=1, selection_score=2)]
        payload = dict(
            epoch=0, optimizer_steps=0, history=rows[:1], model={"weight": torch.tensor([3.0])}
        )
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp)
            save_checkpoint(engine, payload, "fixture")
            best, state = select_state(engine, "fixture", dict(history=rows))
            self.assertEqual(best["block"], 0)
            self.assertTrue(torch.equal(state["weight"], payload["model"]["weight"]))

    def test_resume_rejects_completion_from_another_arm(self):
        from insect_utils.near_finetune_training import train_arm

        frozen = dict(fingerprint="fixture", spec=dict(arms=["original_data", "near_supplement"]))
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            base = folder / "arms/original_data"
            base.mkdir(parents=True)
            (base / "completion.json").write_text('{"fingerprint":"wrong"}')
            with patch("insect_utils.near_finetune_training.verify_finetune", return_value=frozen):
                with self.assertRaises(ValueError):
                    train_arm(folder, "original_data")


if __name__ == "__main__":
    unittest.main()
