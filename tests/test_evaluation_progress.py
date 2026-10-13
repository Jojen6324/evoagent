import json
import os
import tempfile
import unittest

from evoagent.evaluation_harness import EndToEndEvaluationHarness
from evoagent.evaluation_progress import EvaluationProgressReporter
from evoagent.reviewer import LocalRuleReviewer


class EvaluationProgressTests(unittest.TestCase):
    def test_harness_persists_case_progress_and_formats_terminal_status(self):
        case = {
            "id": "progress-1", "repository": "org/repo", "pull_request": 1,
            "split": "validation", "source": {"kind": "offline-fixture"},
            "diff": "--- a/a.py\n+++ b/a.py\n@@ -0,0 +1 @@\n+value = 1\n",
            "expected_findings": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            jsonl_path = os.path.join(directory, "progress.jsonl")
            reporter = EvaluationProgressReporter(jsonl_path, echo=False)

            EndToEndEvaluationHarness(progress_callback=reporter).run(
                LocalRuleReviewer(), [case], "progress-test",
            )

            with open(jsonl_path, encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(
                ["run_started", "case_started", "case_completed", "run_completed"],
                [item["event"] for item in events],
            )
            self.assertTrue(events[2]["result"]["execution_success"])
            line = reporter._console_line(events[2])
            self.assertIn("1/1", line)
            self.assertIn("PASS", line)
            self.assertIn("progress-1", line)


if __name__ == "__main__":
    unittest.main()
