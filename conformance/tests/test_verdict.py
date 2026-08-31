#!/usr/bin/env python3
"""Behavioral tests for conformance failure ownership."""

import subprocess
import tempfile
import unittest
from pathlib import Path


VERDICT = Path(__file__).resolve().parents[1] / "verdict.py"


class VerdictOwnershipTests(unittest.TestCase):
    def run_verdict(self, translation: str = "", driver: str = "", guest: str = "FAIL"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = root / "native.txt"
            guest_run = root / "guest.txt"
            translation_errors = root / "translation-errors.txt"
            driver_errors = root / "driver-errors.txt"
            native.write_text("CASE sample PASS oracle\nDEVICE native\n")
            guest_run.write_text(f"CASE sample {guest} guest\nDEVICE guest\n")
            translation_errors.write_text(translation)
            driver_errors.write_text(driver)
            return subprocess.run(
                [
                    "python3",
                    str(VERDICT),
                    "--native",
                    str(native),
                    "--guest",
                    str(guest_run),
                    "--translation-errors",
                    str(translation_errors),
                    "--driver-errors",
                    str(driver_errors),
                    "--quiet",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

    def test_translation_failure_is_named_and_accepted(self):
        result = self.run_verdict(translation="sample # package=sample-query\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("translation  sample", result.stdout)
        self.assertIn("active translation failures 1/1, active driver failures 0/0", result.stdout)

    def test_driver_failure_is_named_and_accepted(self):
        result = self.run_verdict(driver="sample # clear publication ordering\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("driver  sample", result.stdout)
        self.assertIn("active translation failures 0/0, active driver failures 1/1", result.stdout)

    def test_one_failure_cannot_have_two_owners(self):
        result = self.run_verdict(
            translation="sample # package=sample-query\n",
            driver="sample # publication ordering\n",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DUPLICATE-CLASSIFICATION  sample", result.stdout)

    def test_unclassified_guest_failure_remains_a_regression(self):
        result = self.run_verdict()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("REGRESSION  sample", result.stdout)

    def test_passing_known_failure_is_stale(self):
        result = self.run_verdict(driver="sample # publication ordering\n", guest="PASS")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FIXED-DRIVER  sample", result.stdout)


if __name__ == "__main__":
    unittest.main()
