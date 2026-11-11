"""Checks for scripts/compare_runs.py on toy runs; no models or GPU needed.

Run: .venv/bin/python -m unittest discover -s tests
"""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_runs import compare, compare_scores, compare_trials, differences  # noqa: E402


def row(n, label, score, tier="Level 1 (Top-K GMM) + ASV", error=""):
    scored = score is not None
    return {"trial": str(n), "speaker": "A", "file": f"{n}.wav", "source": "", "label": label,
            "status": "scored" if scored else "failed", "score": repr(score) if scored else "",
            "phoneme_score": repr(score) if scored else "", "tier": tier if scored else "",
            "error": error}


ROWS = [row(0, "bonafide", 0.9), row(1, "bonafide", 0.8), row(2, "spoof", 0.2),
        row(3, "spoof", None, error="missing_file: 3.wav")]


class CompareTrialsTest(unittest.TestCase):
    def test_identical_rows_pass(self):
        self.assertTrue(compare_trials(ROWS, [dict(r) for r in ROWS])["identical"])

    def test_tier_or_status_change_fails(self):
        changed = [dict(r) for r in ROWS]
        changed[0]["tier"] = "Level 2 (All GMM) + ASV"
        changed[3] = row(3, "spoof", 0.1)
        result = compare_trials(ROWS, changed)
        self.assertFalse(result["identical"])
        self.assertEqual(result["mismatches"], {"tier": 2, "status": 1, "error": 1})

    def test_missing_trial_fails(self):
        self.assertFalse(compare_trials(ROWS, ROWS[:-1])["identical"])

    def test_score_change_alone_passes_but_is_reported(self):
        changed = [dict(r) for r in ROWS]
        changed[1]["score"] = repr(0.8 + 1e-12)
        self.assertTrue(compare_trials(ROWS, changed)["identical"])
        scores = compare_scores(ROWS, changed)
        self.assertEqual(scores["compared"], 3)  # failed trial is not compared
        self.assertEqual((scores["score"]["identical"], scores["score"]["different"]), (2, 1))
        self.assertAlmostEqual(scores["score"]["max_abs_difference"], 1e-12)
        self.assertEqual(scores["score"]["largest"][0]["trial"], "1")
        self.assertEqual(scores["phoneme_score"]["different"], 0)


class CompareRunsTest(unittest.TestCase):
    def write_run(self, folder, rows, commit, eer):
        (folder / "profiles").mkdir(parents=True)
        (folder / "profiles/A.joblib").write_bytes(b"profile")
        record = {"run_id": folder.name, "git": {"commit": commit, "dirty": False, "branch": "x"},
                  "environment": {"packages": {"torch": "2.6.0"}}}
        metrics = {"pooled": {"eer": eer, "auc": 1.0},
                   "per_speaker": {"A": {"eer": eer, "auc": 1.0}}}
        (folder / "run.json").write_text(json.dumps(record))
        (folder / "config.resolved.json").write_text(json.dumps({"alpha": 0.8}))
        (folder / "metrics.json").write_text(json.dumps(metrics))
        with (folder / "scores.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(ROWS[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_full_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            self.write_run(a, ROWS, "c1", 0.10)
            self.write_run(b, ROWS, "c2", 0.11)
            report = compare(a, b)
        self.assertTrue(report["replay_pass"])
        self.assertEqual(report["provenance"]["record"], {"git.commit": ["c1", "c2"]})
        self.assertTrue(report["provenance"]["profiles_identical"])
        self.assertAlmostEqual(report["metrics"]["eer"]["difference_points"], 1.0)

    def test_differences_handles_nesting_and_missing_keys(self):
        self.assertEqual(differences({"a": {"b": 1}}, {"a": {"b": 2}, "c": 3}),
                         {"a.b": [1, 2], "c": [None, 3]})


if __name__ == "__main__":
    unittest.main()
