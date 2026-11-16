"""Regression checks for protocol drift and preservation of existing records."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "stage01", PROJECT / "tools/check_stage01_baseline.py"
)
STAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGE)


class Stage01ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(STAGE.DEFAULT_PROTOCOL.read_text())

    def test_entrypoint_overrides_take_precedence_over_json_defaults(self):
        settings = STAGE.check_protocol(PROJECT, self.protocol)
        self.assertEqual(settings["training"]["model_type"], "attention")
        self.assertFalse(settings["sampling"]["only_point_north"])
        self.assertFalse(settings["sampling"]["use_noisy_spiral"])

    def test_changed_training_protocol_is_rejected_with_field_name(self):
        self.protocol["legacy_reference"]["training"]["epochs"] = 151
        with self.assertRaisesRegex(ValueError, r"training.epochs"):
            STAGE.check_protocol(PROJECT, self.protocol)

    def test_changed_source_override_is_detected(self):
        original = STAGE.parse_source

        def changed_source(project, relative):
            if relative == "run_ANN_homing_simulations.py":
                source = (project / relative).read_text()
                return STAGE.ast.parse(
                    source.replace(
                        'config_training["epochs"] = epochs', 'config_training["epochs"] = 151'
                    )
                )
            return original(project, relative)

        with patch.object(STAGE, "parse_source", side_effect=changed_source):
            with self.assertRaisesRegex(ValueError, r"training.epochs"):
                STAGE.check_protocol(PROJECT, self.protocol)

    def test_dynamic_step_formula_cannot_be_replaced_by_configured_constant(self):
        self.protocol["legacy_reference"]["evaluation"]["effective_step_expression"] = "0.5"
        with self.assertRaisesRegex(ValueError, "effective_step_expression"):
            STAGE.check_protocol(PROJECT, self.protocol)

    def test_unimplemented_downward_protocol_cannot_be_marked_ready(self):
        self.protocol["target_experiment"]["execution_ready"] = True
        with self.assertRaisesRegex(ValueError, "must not advertise"):
            STAGE.check_protocol(PROJECT, self.protocol)

    def test_literal_reader_does_not_execute_entrypoint_calls(self):
        tree = STAGE.ast.parse("epochs = 150\nrun_expensive_experiment()\n")
        self.assertEqual(STAGE.literal_assignments(tree.body), {"epochs": 150})

    def test_existing_baseline_record_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "record.json"
            STAGE.write_new_json(destination, {"preserved": True})
            with self.assertRaises(FileExistsError):
                STAGE.write_new_json(destination, {"preserved": False})
            self.assertEqual(json.loads(destination.read_text()), {"preserved": True})


if __name__ == "__main__":
    unittest.main()
