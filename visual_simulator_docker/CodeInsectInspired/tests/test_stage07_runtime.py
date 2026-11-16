"""Exact CPU epoch resume with stochastic inputs and dropout; torch is lazy-loaded."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

HAS_TORCH = all(importlib.util.find_spec(name) for name in ("torch", "numpy", "torchvision"))


@unittest.skipUnless(HAS_TORCH, "Training runtime tests need torch and its dataset dependencies")
class TrainingResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def setup_training(self, seed=17, cuda=False):
        import random
        import numpy as np
        import torch
        from insect_utils.downward_training import seed_all

        torch.set_num_threads(2)
        seed_all(seed, cuda)

        class TrainingData(torch.utils.data.Dataset):
            def __len__(self):
                return 8

            def __getitem__(self, index):
                noise = random.random() * 0.01 + float(np.random.random()) * 0.01
                return torch.tensor([index / 8 + noise, 1.0]), torch.tensor([index / 4, -index / 8])

        model = torch.nn.Sequential(torch.nn.Dropout(0.25), torch.nn.Linear(2, 2)).to(
            "cuda" if cuda else "cpu"
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        train = torch.utils.data.DataLoader(
            TrainingData(), batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(10)
        )
        val = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.ones(4, 2), torch.zeros(4, 2)),
            batch_size=2,
            generator=torch.Generator().manual_seed(11),
        )
        return model, optimizer, train, val

    def test_epoch_resume_restores_model_optimizer_random_streams_and_loader_order(self):
        import torch
        from insect_utils.downward_training import train_epochs

        config = {"epochs": 3, "validation_interval_updates": 5, "minimum_host_available_bytes": 0}
        full = self.setup_training()
        expected = train_epochs(*full, config, self.folder / "full", "same")
        interrupted = self.setup_training()
        part = train_epochs(
            *interrupted, config, self.folder / "resume", "same", stop_after_epoch=1
        )
        self.assertEqual(part["status"], "interrupted")
        restored = self.setup_training(seed=999)
        actual = train_epochs(*restored, config, self.folder / "resume", "same", resume=True)
        self.assertEqual(actual["optimizer_steps"], 12)
        self.assertEqual(expected["final_parameter_sha256"], actual["final_parameter_sha256"])
        for a, b in zip(full[0].parameters(), restored[0].parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_nonfinite_loss_does_not_publish_a_completed_epoch_checkpoint(self):
        import torch
        from insect_utils.downward_training import train_epochs

        model, optimizer, _, val = self.setup_training()
        bad = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.ones(2, 2), torch.full((2, 2), float("nan"))),
            batch_size=2,
            generator=torch.Generator().manual_seed(10),
        )
        with self.assertRaisesRegex(ValueError, "Non-finite training loss"):
            train_epochs(model, optimizer, bad, val, {"epochs": 1}, self.folder / "bad", "bad")
        self.assertFalse((self.folder / "bad/checkpoint_latest.json").exists())

    def test_resume_requires_explicit_flag_and_matching_fingerprint(self):
        from insect_utils.downward_training import train_epochs

        config = {"epochs": 2, "validation_interval_updates": 0}
        train_epochs(
            *self.setup_training(), config, self.folder / "run", "original", stop_after_epoch=1
        )
        with self.assertRaisesRegex(ValueError, "explicitly use"):
            train_epochs(*self.setup_training(), config, self.folder / "run", "original")
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            train_epochs(
                *self.setup_training(), config, self.folder / "run", "changed", resume=True
            )

    def test_gpu_resume_restores_cuda_rng_and_optimizer_device(self):
        import torch
        from insect_utils.downward_training import train_epochs

        if not torch.cuda.is_available():
            self.skipTest("CUDA checkpoint roundtrip needs a GPU")
        config = {"epochs": 2, "validation_interval_updates": 0}
        full = self.setup_training(cuda=True)
        train_epochs(*full, config, self.folder / "gpu_full", "gpu")
        train_epochs(
            *self.setup_training(cuda=True),
            config,
            self.folder / "gpu_resume",
            "gpu",
            stop_after_epoch=1,
        )
        restored = self.setup_training(seed=999, cuda=True)
        train_epochs(*restored, config, self.folder / "gpu_resume", "gpu", resume=True)
        for a, b in zip(full[0].parameters(), restored[0].parameters()):
            self.assertTrue(torch.allclose(a, b, rtol=1e-6, atol=1e-6))
        self.assertTrue(all(state["exp_avg"].is_cuda for state in restored[1].state.values()))


if __name__ == "__main__":
    unittest.main()
