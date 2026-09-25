"""Exercise the official example fixtures; no generated video or model run."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DevEvalTests(unittest.TestCase):
    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-B", "scripts/dev_eval.py", *map(str, args)],
                              cwd=ROOT, capture_output=True, text=True)

    def test_scores_official_examples_without_overwriting_experiments(self):
        with tempfile.TemporaryDirectory() as temp:
            for _ in range(2):
                result = self.run_tool("--pred", "examples/predictions.json", "--gt",
                                       "examples/ground_truth.json", "--experiments", temp)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            records = [json.loads(p.read_text()) for p in sorted(Path(temp).glob("EXP-*.json"))]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["mode"], "saved-predictions")
            self.assertEqual(records[0]["report"], records[1]["report"])
            self.assertIn("part_a", records[0]["report"])

    def test_missing_labels_fails_before_runner_or_experiment_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self.run_tool("--videos", "data/samples", "--gt", Path(temp) / "missing.json",
                                   "--experiments", Path(temp) / "experiments")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Required file missing", result.stderr)
            self.assertFalse((Path(temp) / "experiments").exists())

    def test_harness_error_is_not_silently_scored_as_a_successful_run(self):
        with tempfile.TemporaryDirectory() as temp:
            pred = json.loads((ROOT / "examples/predictions.json").read_text())
            pred["log"] = {"test": {"errors": ["over time budget"]}}
            path = Path(temp) / "failed.json"
            path.write_text(json.dumps(pred))
            result = self.run_tool("--pred", path, "--gt", "examples/ground_truth.json",
                                   "--experiments", Path(temp) / "experiments")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Harness recorded failures", result.stderr)
            self.assertEqual(list(Path(temp).glob("experiments/EXP-*.json")), [])


if __name__ == "__main__":
    unittest.main()
