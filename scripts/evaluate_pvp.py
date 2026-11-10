"""Compute EER and ROC-AUC from a per-trial PVP scores file, as upstream 2_PVP.py does.

Metrics use upstream utils.compute_eer and sklearn roc_auc_score (2_PVP.py:160-176):
pooled over all scored trials, then per speaker. Label 1 = bonafide, 0 = spoof;
a higher score means more consistent with the genuine speaker profile.
Failed trials have no score: they are counted and listed, never silently dropped.
"""

import argparse
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_UTILS = ROOT / "original/PVP/utils.py"


def load_compute_eer():
    # Load by path: upstream's module name "utils" is too generic to put on sys.path.
    spec = importlib.util.spec_from_file_location("pvp_utils", UPSTREAM_UTILS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_eer


compute_eer = load_compute_eer()


def read_scores(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def metrics(labels, scores):
    labels, scores = np.array(labels, dtype=int), np.array(scores, dtype=float)
    counts = {"trials": len(labels), "bonafide": int(labels.sum()),
              "spoof": int(len(labels) - labels.sum())}
    if counts["bonafide"] == 0 or counts["spoof"] == 0:
        return {**counts, "eer": None, "auc": None, "note": "needs both labels"}
    eer, threshold = compute_eer(scores, labels)
    return {**counts, "eer": float(eer), "eer_threshold": float(threshold),
            "auc": float(roc_auc_score(labels, scores))}


def evaluate(rows):
    scored = [r for r in rows if r["status"] == "scored"]
    failed = [r for r in rows if r["status"] != "scored"]
    labels = [1 if r["label"] == "bonafide" else 0 for r in scored]
    scores = [float(r["score"]) for r in scored]
    result = {
        "metric": "upstream utils.compute_eer (FPR at ROC point nearest FNR) and sklearn "
                  "roc_auc_score; fractions, not percent",
        "score_direction": "higher = bona fide; label 1 = bonafide",
        "trials": len(rows),
        "scored": len(scored),
        "failed": len(failed),
        "failed_by_reason": dict(Counter(r["error"].split(":")[0] for r in failed)),
        "failed_files": [r["file"] for r in failed],
        "tiers": dict(Counter(r["tier"] for r in scored)),
        "pooled": metrics(labels, scores),
        "per_speaker": {},
    }
    for speaker in sorted({r["speaker"] for r in scored}):
        pairs = [(lab, s) for r, lab, s in zip(scored, labels, scores) if r["speaker"] == speaker]
        result["per_speaker"][speaker] = metrics([p[0] for p in pairs], [p[1] for p in pairs])
    return result


def compare_with_paper(result, paper):
    """Paper values are percentages (Table 2); ours are fractions."""
    eer, auc = result["pooled"]["eer"], result["pooled"]["auc"]
    return {
        **paper,
        "ours_eer_percent": None if eer is None else round(100 * eer, 2),
        "ours_auc_percent": None if auc is None else round(100 * auc, 2),
        "eer_difference_points": None if eer is None else round(100 * eer - paper["eer_percent"], 2),
        "auc_difference_points": None if auc is None else round(100 * auc - paper["auc_percent"], 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path, help="per-trial scores CSV from run_pvp.py")
    parser.add_argument("--output", type=Path, help="metrics JSON (default: print only)")
    parser.add_argument("--config", type=Path, help="run config with paper values to compare")
    args = parser.parse_args()
    result = evaluate(read_scores(args.scores))
    if args.config:
        result["paper_comparison"] = compare_with_paper(
            result, json.loads(args.config.read_text())["paper"])
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
