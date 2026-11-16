"""Regression: stage initialization must not silently restore temporal DLSS."""

from pathlib import Path
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.downward_capture import configure_capture_rendering, require_capture_rendering


class RenderingContractTests(unittest.TestCase):
    def setUp(self):
        class Settings:
            def __init__(self):
                self.values = {
                    "/rtx/post/aa/op": 3,
                    "/rtx-transient/post/aa/limitedOps": True,
                    "/rtx/post/histogram/enabled": True,
                }

            def get(self, key):
                return self.values.get(key)

            def set(self, key, value):
                self.values[key] = value

        self.settings = Settings()
        settings = self.settings

        class ReplicatorSettings:
            def set_render_rtx_realtime(self, antialiasing):
                if antialiasing != "Off":
                    raise AssertionError("The declared capture mode is AA off")
                # Simulate Kit's reset while enabling the renderer. Enforcement
                # must happen after this call, rather than before it.
                settings.set("/rtx/post/aa/op", 3)
                settings.set("/rtx-transient/post/aa/limitedOps", True)
                settings.set("/rtx/rendermode", "RayTracedLighting")

        self.replicator = ReplicatorSettings()

    def test_renderer_initialization_cannot_leave_dlss_enabled(self):
        configure_capture_rendering(self.settings, self.replicator)
        actual = require_capture_rendering(self.settings)
        self.assertEqual(actual["antialiasing"], 0)
        self.assertIs(actual["aa_limited_ops"], False)
        self.assertIs(actual["auto_exposure"], False)

    def test_later_renderer_overrides_are_rejected(self):
        configure_capture_rendering(self.settings, self.replicator)
        for key, value in [
            ("/rtx/post/aa/op", 3),
            ("/rtx-transient/post/aa/limitedOps", True),
            ("/rtx/post/histogram/enabled", True),
        ]:
            configure_capture_rendering(self.settings, self.replicator)
            self.settings.set(key, value)
            with self.assertRaisesRegex(ValueError, "overwritten"):
                require_capture_rendering(self.settings)


if __name__ == "__main__":
    unittest.main()
