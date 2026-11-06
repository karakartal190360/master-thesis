"""Compare a Parquet dataset release with PVP enrollment/test protocols, using metadata only.

Reads audio path, label, speaker and source columns (names per release in
parquet_datasets.py); audio bytes are not decoded. Each protocol row is matched to the
release row with the same path, speaker and source. Reports missing, extra, ambiguous and
duplicate files, enrollment/test overlap, and protocol vs dataset label pairs.
Writes a map from protocol file name to release shard/row, used by extract_parquet_audio.py.
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from parquet_datasets import DATASETS, read_rows


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wav_name(key):
    # Same rule as upstream 2_PVP.py: append ".wav" unless already present.
    return key if key.endswith(".wav") else key + ".wav"


def read_protocol(path, split):
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        next(reader)  # upstream skips the header and assigns columns by position
        return [{"split": split, "speaker": r[0], "file": wav_name(r[1]), "source": r[2], "label": r[3]}
                for r in reader]


def read_dataset(parquet_dir, dataset):
    rows = []
    for shard in sorted(parquet_dir.glob("*.parquet")):
        rows.extend(read_rows(shard, dataset))
    return rows


def candidate_paths(p, strip_source_suffix):
    # Famous Figures: some protocol keys are "<release stem>_<Source>.wav" (e.g. "_-" for
    # bona fide), while the release path is "<release stem>.wav". Our inferred rule.
    names = [("direct", p["file"])]
    suffix = "_" + p["source"] + ".wav"
    if strip_source_suffix and p["file"].endswith(suffix):
        names.append(("source_suffix_stripped", p["file"][:-len(suffix)] + ".wav"))
    return names


def match(protocol, by_file, strip_source_suffix):
    """Find release rows with the same path, speaker and source for each protocol row."""
    for p in protocol:
        p["hits"] = [(rule, d) for rule, name in candidate_paths(p, strip_source_suffix)
                     for d in by_file.get(name, [])
                     if d["speaker"] == p["speaker"] and (d["source"] or "") == p["source"]]


def describe(p):
    return {k: p[k] for k in ("split", "speaker", "file", "source", "label")}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--enroll", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True, help="protocol file -> shard/row CSV")
    args = parser.parse_args()
    columns = DATASETS[args.dataset]

    protocol = read_protocol(args.enroll, "enroll") + read_protocol(args.test, "test")
    dataset = read_dataset(args.parquet_dir, args.dataset)

    by_file = {}
    for d in dataset:
        by_file.setdefault(d["file"], []).append(d)
    protocol_files = Counter(p["file"] for p in protocol)
    enroll_files = {p["file"] for p in protocol if p["split"] == "enroll"}
    test_files = {p["file"] for p in protocol if p["split"] == "test"}

    match(protocol, by_file, columns["strip_source_suffix"])
    matched = [p for p in protocol if len(p["hits"]) == 1]
    unmatched = [p for p in protocol if not p["hits"]]
    ambiguous = [p for p in protocol if len(p["hits"]) > 1]
    used = Counter((p["hits"][0][1]["shard"], p["hits"][0][1]["row"]) for p in matched)
    extra = [d for d in dataset if (d["shard"], d["row"]) not in used]

    rules, label_pairs, source_counts = Counter(), Counter(), Counter()
    for p in matched:
        rule, d = p["hits"][0]
        rules[f"{rule}: {p['source'] or '(empty)'}"] += 1
        label_pairs[f"{p['label']} -> {d['label']}"] += 1
        source_counts[p["source"] or "(empty)"] += 1

    # Release AudioName should equal the audio path without ".wav".
    name_mismatch = [{"file": d["file"], "name": d["name"]} for d in dataset
                     if d["name"] is not None and wav_name(d["name"]) != d["file"]]
    # Speaker splits: shard prefix "<Speaker>-000NN-of-000MM" should equal the row's speaker.
    split_mismatch = []
    if columns["speaker_splits"]:
        split_mismatch = [{"file": d["file"], "shard": d["shard"], "speaker": d["speaker"]}
                          for d in dataset if d["shard"].split("-")[0] != d["speaker"]]

    args.map.parent.mkdir(parents=True, exist_ok=True)
    with args.map.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file", "split", "speaker", "source", "label", "shard", "row",
                         "release_path", "rule"])
        for p in matched:
            rule, d = p["hits"][0]
            writer.writerow([p["file"], p["split"], p["speaker"], p["source"], p["label"],
                             d["shard"], d["row"], d["file"], rule])

    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "dataset_protocol_check",
        "scientific_result": False,
        "dataset": args.dataset,
        "columns": columns,
        "command": [sys.executable, *sys.argv],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True).strip()),
        "script_sha256": {p.name: sha256(p) for p in
                          (Path(__file__), Path(__file__).with_name("parquet_datasets.py"))},
        "protocol_sha256": {str(args.enroll): sha256(args.enroll), str(args.test): sha256(args.test)},
        "map": str(args.map),
        "map_sha256": sha256(args.map),
        "counts": {
            "dataset_rows": len(dataset),
            "dataset_unique_paths": len(by_file),
            "protocol_rows": len(protocol),
            "protocol_enroll_rows": sum(p["split"] == "enroll" for p in protocol),
            "protocol_test_rows": sum(p["split"] == "test" for p in protocol),
            "protocol_unique_files": len(protocol_files),
            "protocol_paths_absent_from_release": sum(p["file"] not in by_file for p in protocol),
            "matched_protocol_rows": len(matched),
            "unmatched_protocol_rows": len(unmatched),
            "ambiguous_protocol_rows": len(ambiguous),
            "release_rows_matched_twice": sum(n > 1 for n in used.values()),
            "extra_dataset_rows": len(extra),
            "name_path_mismatches": len(name_mismatch),
            "shard_speaker_mismatches": len(split_mismatch),
        },
        "match_rules": dict(rules),
        "label_pairs": dict(label_pairs),
        "matched_source_counts": dict(source_counts),
        "enroll_test_overlap": sorted(enroll_files & test_files),
        "protocol_duplicate_files": {f: n for f, n in protocol_files.items() if n > 1},
        "dataset_duplicate_paths": {f: [(d["shard"], d["row"], d["source"]) for d in rs]
                                    for f, rs in by_file.items() if len(rs) > 1},
        # Unmatched rows with any release rows of the same path, to show why they failed.
        "unmatched_protocol_rows": [
            {**describe(p), "same_path_release_rows": [
                {"speaker": d["speaker"], "source": d["source"]} for d in by_file.get(p["file"], [])]}
            for p in unmatched],
        "ambiguous_protocol_rows": [
            {**describe(p), "hits": [(rule, d["shard"], d["row"]) for rule, d in p["hits"]]}
            for p in ambiguous],
        "extra_dataset_rows": [{k: d[k] for k in ("shard", "row", "file", "speaker", "source")}
                               for d in extra],
        "name_path_mismatches": name_mismatch,
        "shard_speaker_mismatches": split_mismatch,
        "dataset_label_counts": dict(Counter(d["label"] for d in dataset)),
        "dataset_speaker_counts": dict(Counter(d["speaker"] for d in dataset)),
        "dataset_source_counts": dict(Counter(d["source"] or "(empty)" for d in dataset)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["counts"], indent=2))
    print(json.dumps(report["match_rules"], indent=2))
    print(f"report {args.report}")


if __name__ == "__main__":
    main()
