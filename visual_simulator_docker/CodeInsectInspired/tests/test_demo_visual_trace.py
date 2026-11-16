import unittest
from insect_utils.demo_visual_trace import VisualTrace, recover_trace


class VisualTraceTests(unittest.TestCase):
    def test_only_committed_neural_endpoints_and_snapshot_is_independent(self):
        trace = VisualTrace()
        trace.begin([10, 0, 40], 90)
        trace.completed_action(1, [8, 0, 40], 180)
        snapshot = trace.snapshot(2, "SUCCESS", "complete")
        trace.completed_action(2, [6, 0, 40], 180)
        self.assertEqual(len(snapshot["positions"]), 2)
        self.assertEqual(snapshot["positions"][0]["distance_m"], 10)
        self.assertEqual(snapshot["positions"][-1]["z_m"], 40)
        with self.assertRaises(ValueError):
            trace.completed_action(4, [0, 0, 40], 0)

    def test_recovery_excludes_pi_and_vertical_landing(self):
        def event(name, position, steps=0, **fields):
            state = dict(
                position_m=position,
                yaw_deg=0,
                visual_steps=steps,
                homing_outcome="SUCCESS",
                reason="complete",
                landing_measurement={"landing_position_m": [1, 0, 0.3], "horizontal_distance_m": 1},
            )
            return dict(event=name, episode=0, state=state, **fields)

        events = [
            event("motion", [500, 0, 40]),
            event("stage_transition", [3, 0, 40], to_stage="VISUAL_RETURN"),
            event("motion", [2, 0, 40]),
            event("neural_action_completed", [1, 0, 40], 1),
            event("landing_complete", [1, 0, 0.3], 1),
        ]
        snapshot = recover_trace(events, 0)
        self.assertEqual([p["x_m"] for p in snapshot["positions"]], [3, 1])
        self.assertTrue(all(p["z_m"] == 40 for p in snapshot["positions"]))

    def test_initial_success_has_one_handoff_and_no_made_up_action(self):
        trace = VisualTrace()
        trace.begin([1, 0, 40], 0)
        self.assertEqual(len(trace.snapshot(0, "SUCCESS", "initial arrival")["positions"]), 1)


if __name__ == "__main__":
    unittest.main()
