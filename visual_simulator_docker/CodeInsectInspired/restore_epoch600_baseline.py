"""Restore the literal epoch600 weights without changing frozen historical runs.

Run with Isaac's Python for CPU export; --show needs only standard Python.
"""

import argparse
import json

from insect_utils.experiment_config import PROJECT, canonical_hash, digest
from insect_utils.experiment_state import atomic_json, now, read_json


SOURCE = PROJECT / "runs/stage16_rotation_600_01"
DESTINATION = PROJECT / "runs/stage21_epoch600_rollback_01"
ACTIVE = PROJECT / "config/experiments/active_baseline.json"
FINAL_SHA256 = "1dd0960dcf74796179e151556dcb165a028f2b124b9c47aabdaadcb1d24a7017"


def relative(path):
    return str(path.relative_to(PROJECT))


def verify(manifest):
    payload = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if canonical_hash(payload) != manifest["fingerprint"]:
        raise ValueError("Baseline manifest fingerprint changed")
    for name, expected in manifest["files"].items():
        if digest(PROJECT / name) != expected:
            raise ValueError("Baseline artifact changed: " + name)
    return manifest


def restore():
    receipt = DESTINATION / "baseline.json"
    if receipt.exists():
        manifest = verify(read_json(receipt))
    else:
        import torch

        from insect_utils.circular_augmentation import CircularAttention
        from insect_utils.coverage_model import PhysicalHomeVector
        from insect_utils.coverage_training import export_physical
        from insect_utils.epoch_extension import verify_extension
        from insect_utils.rgb_preprocessing import load_rgb_tensor

        frozen = verify_extension(SOURCE)
        completed = read_json(SOURCE / "training/completion.json")
        weights = SOURCE / "training/final.pth"
        if (
            completed["epochs_total"] != 600
            or completed["files"]["final.pth"] != FINAL_SHA256
            or digest(weights) != FINAL_SHA256
        ):
            raise ValueError("The original epoch600 weights did not pass integrity checks")
        torch.set_num_threads(2)
        model = CircularAttention(frozen["training_config"]).eval()
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        engine = SOURCE / "training/engine_scaled_units"
        pointer = read_json(engine / "checkpoint_latest.json")
        checkpoint_path = engine / pointer["file"]
        if pointer["epoch"] != 600 or digest(checkpoint_path) != pointer["sha256"]:
            raise ValueError("The original epoch600 full checkpoint changed")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["epoch"] != 600 or checkpoint["fingerprint"] != frozen["fingerprint"]:
            raise ValueError("Checkpoint belongs to another training run")
        if set(state) != set(checkpoint["model"]) or any(
            not torch.equal(state[k], checkpoint["model"][k]) for k in state
        ):
            raise ValueError("Final weights differ from the full epoch600 checkpoint")
        dataset_path = PROJECT / frozen["protocol"]["dataset_run"] / "dataset.json"
        dataset = read_json(dataset_path)
        if canonical_hash(dataset["records"]) != dataset["records_sha256"]:
            raise ValueError("Original dataset records changed")
        examples = [r for r in dataset["records"] if r["split"] == "validation"][:2]
        raw = torch.stack(
            [load_rgb_tensor(PROJECT / r["image"], [1024, 1024], r["sha256"]) for r in examples]
        )
        physical = PhysicalHomeVector(model, frozen["training_config"]["scale_m"]).eval()
        exported = export_physical(physical, raw, DESTINATION / "deployment")
        if exported["status"] != "passed":
            raise ValueError("Deployment export check failed")
        paths = [
            weights,
            checkpoint_path,
            SOURCE / "frozen.json",
            SOURCE / "training/completion.json",
            dataset_path,
            PROJECT / "restore_epoch600_baseline.py",
        ] + sorted((DESTINATION / "deployment").iterdir())
        manifest = dict(
            schema_version=1,
            created_at=now(),
            status="restored",
            source_run=relative(SOURCE),
            trained_epochs=600,
            selected_epoch=600,
            selection="User requested literal epoch600 final weights; no new optimizer updates",
            model=dict(
                epoch=600,
                weights=relative(weights),
                torchscript=relative(DESTINATION / "deployment/home_vector_m.torchscript.pt"),
                onnx=relative(DESTINATION / "deployment/home_vector_m.onnx"),
                scale_m=44.0,
                output_units="body-frame home vector in meters",
            ),
            dataset=dict(
                run=frozen["protocol"]["dataset_run"],
                records_sha256=dataset["records_sha256"],
                train_images=4096,
                validation_images=320,
                stage19_near_supplement_used=False,
                stage20_finetuning_used=False,
                spiral_only=False,
                caveat="Historical dataset already contains Stage11 2D coverage samples",
            ),
            future_collection_constraint="Training capture must follow a spiral flight trajectory",
            former_navigation_model=dict(
                trained_epochs=600,
                selected_epoch=589,
                weights=relative(SOURCE / "training/best_validation.pth"),
                torchscript=relative(SOURCE / "training/deployment/home_vector_m.torchscript.pt"),
                note="Historical 5/10/20m navigation scores belong to epoch589, not final epoch600",
            ),
            navigation_tests_run_after_restore=0,
            success_criterion_modified=False,
            export_check=exported,
            files={relative(p): digest(p) for p in paths},
        )
        manifest["fingerprint"] = canonical_hash(manifest)
        atomic_json(receipt, manifest)
    verify(manifest)
    atomic_json(ACTIVE, dict(manifest, manifest=relative(receipt)))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="Verify and print the active baseline")
    args = parser.parse_args()
    if args.show:
        active = read_json(ACTIVE)
        manifest = verify(read_json(PROJECT / active["manifest"]))
        if {k: v for k, v in active.items() if k != "manifest"} != manifest:
            raise ValueError("Active baseline differs from its verified restore receipt")
    else:
        manifest = restore()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
