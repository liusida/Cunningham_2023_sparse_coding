import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tinker_evaluator import interpret


class RunFeatureResilientTest(unittest.TestCase):
    def test_failure_is_recorded_without_a_completed_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with patch.object(interpret, "run_feature", side_effect=ValueError("bad labels")):
                completed = interpret.run_feature_resilient(
                    feature=7, output_dir=output_dir
                )

            self.assertFalse(completed)
            self.assertFalse((output_dir / "feature_7.json").exists())
            error = json.loads((output_dir / "feature_7.error.json").read_text())
            self.assertEqual(error["status"], "failed")
            self.assertEqual(error["error_type"], "ValueError")
            self.assertEqual(error["reason"], "bad labels")

    def test_success_removes_stale_error_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            error_path = output_dir / "feature_7.error.json"
            error_path.write_text("stale\n")
            with patch.object(interpret, "run_feature"):
                completed = interpret.run_feature_resilient(
                    feature=7, output_dir=output_dir
                )

            self.assertTrue(completed)
            self.assertFalse(error_path.exists())

    def test_fail_fast_preserves_original_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(interpret, "run_feature", side_effect=ValueError("bad labels")):
                with self.assertRaisesRegex(ValueError, "bad labels"):
                    interpret.run_feature_resilient(
                        feature=7,
                        output_dir=Path(directory),
                        fail_fast=True,
                    )


if __name__ == "__main__":
    unittest.main()
