#!/usr/bin/env python3

import importlib.util
import pathlib
import tempfile
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("boot-score.py")
SPEC = importlib.util.spec_from_file_location("boot_score", MODULE_PATH)
BOOT_SCORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOOT_SCORE)


class BootScoreTests(unittest.TestCase):
    def score(self, lines):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as log:
            log.writelines(lines)
            log.flush()
            return BOOT_SCORE.score(log.name)

    def test_uses_whole_cpu_exact_gpu_population_and_elapsed_window_time(self):
        result = self.score(
            [
                "OFF drain_duty win_ms=500 duty=0.8 draws=100 draw_us=400 proc_us=1000 t=1000\n",
                "OFF gpu_span busy_us=900 retired_draws=90 t=1002\n",
                "OFF window_publish win_ms=500 fresh=10 t=999\n",
                "OFF host_window_cadence window_ms=500 presents=9 offered=10 present_hz=18 offered_hz=20 t=800\n",
                "OFF drain_duty win_ms=1500 duty=0.9 draws=300 draw_us=600 proc_us=6000 t=2000\n",
                "OFF gpu_span busy_us=3300 retired_draws=110 t=1999\n",
                "OFF window_publish win_ms=1500 fresh=30 t=2001\n",
                "OFF host_window_cadence window_ms=1500 presents=27 offered=30 present_hz=18 offered_hz=20 t=1800\n",
            ]
        )

        self.assertIn("cpu=17.50", result)
        self.assertIn("gpu=21.00", result)
        self.assertIn("sum=38.50", result)
        self.assertIn("fps= 18.0", result)
        self.assertIn("offered= 20.0", result)
        self.assertIn("draws/s=   200", result)
        self.assertIn("occ=0.01", result)
        self.assertIn("d/frame=    10", result)

    def test_refuses_a_log_without_the_gpu_owned_denominator(self):
        result = self.score(
            [
                "OFF drain_duty win_ms=1000 duty=0.8 draws=100 proc_us=1000 t=1000\n",
                "OFF gpu_span busy_us=900 t=1000\n",
            ]
        )

        self.assertIn("log predates the exact GPU denominator", result)

    def test_excludes_windows_below_the_declared_driven_duty_band(self):
        result = self.score(
            [
                "OFF drain_duty win_ms=1000 duty=0.45 draws=100 proc_us=1000 t=1000\n",
                "OFF gpu_span busy_us=900 retired_draws=100 t=1000\n",
                "OFF drain_duty win_ms=1000 duty=0.50 draws=100 proc_us=2000 t=2000\n",
                "OFF gpu_span busy_us=1000 retired_draws=100 t=2000\n",
            ]
        )

        self.assertIn("n=1", result)
        self.assertIn("cpu=20.00", result)

    def test_cadence_median_rejects_a_partial_boundary_window(self):
        result = self.score(
            [
                "OFF host_window_cadence window_ms=4000 presents=2 offered=2 present_hz=0.5 offered_hz=0.5 t=900\n",
                "OFF drain_duty win_ms=1000 duty=0.8 draws=100 proc_us=2000 t=1000\n",
                "OFF gpu_span busy_us=1000 retired_draws=100 t=1000\n",
                "OFF host_window_cadence window_ms=1000 presents=40 offered=40 present_hz=40 offered_hz=40 t=1800\n",
                "OFF host_window_cadence window_ms=1000 presents=40 offered=40 present_hz=40 offered_hz=40 t=2800\n",
            ]
        )

        self.assertIn("fps= 40.0", result)
        self.assertIn("offered= 40.0", result)


if __name__ == "__main__":
    unittest.main()
