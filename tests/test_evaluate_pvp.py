"""Checks for scripts/evaluate_pvp.py on toy scores; no models or GPU needed.

Run: .venv/bin/python -m unittest discover -s tests
"""

import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_pvp import compare_with_paper, compute_eer, evaluate  # noqa: E402


def row(speaker, label, score=None, error=""):
    status = "scored" if score is not None else "failed"
    return {"speaker": speaker, "file": f"{speaker}-{label}-{score}.wav", "label": label,
            "status": status, "score": "" if score is None else repr(score),
            "tier": "Level 1 (Top-K GMM)" if score is not None else "", "error": error}


class EvaluateTest(unittest.TestCase):
    def test_matches_upstream_functions_and_skips_failed(self):
        rng = np.random.default_rng(0)
        rows = [row("A", "bonafide", float(s)) for s in rng.normal(1, 1, 50)]
        rows += [row("A", "spoof", float(s)) for s in rng.normal(0, 1, 80)]
        rows += [row("A", "spoof", error="missing_file: x.wav")]
        result = evaluate(rows)

        labels = [1] * 50 + [0] * 80
        scores = [float(r["score"]) for r in rows[:130]]
        eer, _ = compute_eer(np.array(scores), np.array(labels))
        self.assertAlmostEqual(result["pooled"]["eer"], eer)
        self.assertAlmostEqual(result["pooled"]["auc"], roc_auc_score(labels, scores))
        self.assertEqual((result["trials"], result["scored"], result["failed"]), (131, 130, 1))
        self.assertEqual(result["failed_by_reason"], {"missing_file": 1})

    def test_score_direction(self):
        # Higher score = bona fide: perfect separation gives EER 0 and AUC 1.
        rows = [row("A", "bonafide", 0.9), row("A", "bonafide", 0.8),
                row("A", "spoof", 0.2), row("A", "spoof", 0.1)]
        pooled = evaluate(rows)["pooled"]
        self.assertEqual((pooled["eer"], pooled["auc"]), (0.0, 1.0))
        flipped = [dict(r, label="spoof" if r["label"] == "bonafide" else "bonafide") for r in rows]
        self.assertEqual(evaluate(flipped)["pooled"]["auc"], 0.0)

    def test_per_speaker_needs_both_labels(self):
        rows = [row("A", "bonafide", 0.9), row("A", "spoof", 0.1), row("B", "spoof", 0.3)]
        result = evaluate(rows)
        self.assertEqual(result["per_speaker"]["A"]["auc"], 1.0)
        self.assertIsNone(result["per_speaker"]["B"]["eer"])

    def test_paper_comparison_in_percent(self):
        result = {"pooled": {"eer": 0.12, "auc": 0.95}}
        paper = {"eer_percent": 11.37, "auc_percent": 94.58}
        compared = compare_with_paper(result, paper)
        self.assertEqual(compared["eer_difference_points"], 0.63)
        self.assertEqual(compared["auc_difference_points"], 0.42)


if __name__ == "__main__":
    unittest.main()
