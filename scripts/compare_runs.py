"""Compare two PVP runs from run_pvp.py: provenance, per-trial results and metrics.

Replay rule (Phase 5, accepted 2026-11-11): a replay passes when the trial set, labels,
statuses, tiers and failure reasons are identical. Score differences do not fail it, but
are always reported, because upstream leaves cuDNN non-deterministic.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRIAL_FIELDS = ["trial", "speaker", "file", "source", "label", "status", "tier", "error"]
SCORE_FIELDS = ["score", "phoneme_score"]
RECORD_FIELDS = ["scientific_result", "git", "upstream_commit", "config_sha256", "protocols",
                 "audio_root", "weights_sha256", "environment", "seeds"]
EXAMPLES = 10


def run_dir(name):
    path = Path(name)
    return path if path.is_dir() else ROOT / "results/runs" / name


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(folder):
    with (folder / "scores.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {"record": json.loads((folder / "run.json").read_text()),
            "config": json.loads((folder / "config.resolved.json").read_text()),
            "metrics": json.loads((folder / "metrics.json").read_text()),
            "profiles": {p.name: sha256(p) for p in sorted((folder / "profiles").glob("*.joblib"))},
            "rows": rows}


def flatten(value, prefix=""):
    if not isinstance(value, dict):
        return {prefix: value}
    flat = {}
    for key, item in value.items():
        flat.update(flatten(item, f"{prefix}.{key}" if prefix else key))
    return flat


def differences(a, b):
    """Flattened keys whose values differ, as {key: [a, b]}; missing keys are None."""
    a, b = flatten(a), flatten(b)
    return {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def compare_provenance(a, b):
    records = [{k: run["record"].get(k) for k in RECORD_FIELDS} for run in (a, b)]
    for record in records:
        record["git"] = {k: record["git"][k] for k in ("commit", "dirty")}  # branch/status: noise
    return {"record": differences(*records),
            "config": differences(a["config"], b["config"]),
            "profiles_identical": a["profiles"] == b["profiles"],
            "profiles_different": sorted(k for k in set(a["profiles"]) | set(b["profiles"])
                                         if a["profiles"].get(k) != b["profiles"].get(k))}


def compare_trials(rows_a, rows_b):
    result = {"trials": [len(rows_a), len(rows_b)], "mismatches": {}, "mismatch_examples": []}
    for n, (x, y) in enumerate(zip(rows_a, rows_b)):
        fields = [f for f in TRIAL_FIELDS if x[f] != y[f]]
        for field in fields:
            result["mismatches"][field] = result["mismatches"].get(field, 0) + 1
        if fields and len(result["mismatch_examples"]) < EXAMPLES:
            result["mismatch_examples"].append({"row": n, "fields": fields,
                                                "a": {f: x[f] for f in fields},
                                                "b": {f: y[f] for f in fields}})
    result["identical"] = len(rows_a) == len(rows_b) and not result["mismatches"]
    return result


def compare_scores(rows_a, rows_b):
    """Only trials scored in both runs; exact = same printed float (repr)."""
    pairs = [(x, y) for x, y in zip(rows_a, rows_b)
             if x["status"] == "scored" and y["status"] == "scored"]
    result = {"compared": len(pairs)}
    for field in SCORE_FIELDS:
        diffs = [(abs(float(x[field]) - float(y[field])), x["trial"]) for x, y in pairs]
        changed = sorted((d for d in diffs if d[0] > 0), reverse=True)
        result[field] = {
            "identical": sum(x[field] == y[field] for x, y in pairs),
            "different": len(changed),
            "max_abs_difference": changed[0][0] if changed else 0.0,
            "largest": [{"trial": t, "abs_difference": d} for d, t in changed[:EXAMPLES]]}
    return result


def compare_metrics(a, b):
    result = {}
    for metric in ("eer", "auc"):
        x, y = a["pooled"][metric], b["pooled"][metric]
        speaker = [abs(a["per_speaker"][s][metric] - b["per_speaker"][s][metric])
                   for s in a["per_speaker"]
                   if s in b["per_speaker"] and a["per_speaker"][s][metric] is not None]
        # None: metric undefined (genuine-only test lists, e.g. EMIME Finnish talkers)
        difference = None if x is None or y is None else 100 * (y - x)
        result[metric] = {"a": x, "b": y, "difference_points": difference,
                          "max_per_speaker_difference_points": 100 * max(speaker, default=0.0)}
    return result


def compare(folder_a, folder_b):
    a, b = load_run(folder_a), load_run(folder_b)
    trials = compare_trials(a["rows"], b["rows"])
    return {"rule": __doc__.split("\n\n")[1].replace("\n", " "),
            "runs": [a["record"]["run_id"], b["record"]["run_id"]],
            "replay_pass": trials["identical"],
            "trials": trials,
            "scores": compare_scores(a["rows"], b["rows"]),
            "metrics": compare_metrics(a["metrics"], b["metrics"]),
            "provenance": compare_provenance(a, b)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", help="run directory or run ID under results/runs (original)")
    parser.add_argument("run_b", help="run directory or run ID under results/runs (replay)")
    parser.add_argument("--output", type=Path, help="report JSON (default: print only)")
    args = parser.parse_args()
    report = compare(run_dir(args.run_a), run_dir(args.run_b))
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    sys.exit(0 if report["replay_pass"] else 1)


if __name__ == "__main__":
    main()
