"""Compare a Parquet dataset release with PVP enrollment/test protocols, using metadata only.

Reads path, label, speaker and method columns; audio bytes are not decoded.
Reports missing, extra and duplicate files, enrollment/test overlap, and
cross-tabulations of protocol vs dataset speaker, label and source.
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

import pyarrow.parquet as pq


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


def read_dataset(parquet_dir):
    rows = []
    for shard in sorted(parquet_dir.glob("*.parquet")):
        # Per row group: pq.read_table loads the whole audio struct (13 GB RAM for one shard).
        parquet = pq.ParquetFile(shard)
        i = 0
        for group in range(parquet.num_row_groups):
            table = parquet.read_row_group(group, columns=["file_name.path", "label", "person", "method"])
            for r in table.to_pylist():
                rows.append({"shard": shard.name, "row": i, "file": r["file_name"]["path"],
                             "label": r["label"], "person": r["person"], "method": r["method"] or ""})
                i += 1
    return rows


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--enroll", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    protocol = read_protocol(args.enroll, "enroll") + read_protocol(args.test, "test")
    dataset = read_dataset(args.parquet_dir)

    by_file = {}
    for r in dataset:
        by_file.setdefault(r["file"], []).append(r)
    protocol_files = Counter(p["file"] for p in protocol)
    enroll_files = {p["file"] for p in protocol if p["split"] == "enroll"}
    test_files = {p["file"] for p in protocol if p["split"] == "test"}

    missing = [p for p in protocol if p["file"] not in by_file]
    speaker_mismatch, pairs_label, pairs_source = [], Counter(), Counter()
    for p in protocol:
        for d in by_file.get(p["file"], []):
            if d["person"] != p["speaker"]:
                speaker_mismatch.append({**p, "dataset_person": d["person"]})
            pairs_label[f"{p['label']} -> {d['label']}"] += 1
            pairs_source[f"{p['source'] or '(empty)'} -> {d['method'] or '(empty)'}"] += 1

    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "dataset_protocol_check",
        "scientific_result": False,
        "command": [sys.executable, *sys.argv],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "script_sha256": sha256(Path(__file__)),
        "protocol_sha256": {str(args.enroll): sha256(args.enroll), str(args.test): sha256(args.test)},
        "counts": {
            "dataset_rows": len(dataset),
            "dataset_unique_files": len(by_file),
            "protocol_rows": len(protocol),
            "protocol_enroll_rows": sum(p["split"] == "enroll" for p in protocol),
            "protocol_test_rows": sum(p["split"] == "test" for p in protocol),
            "protocol_unique_files": len(protocol_files),
            "missing_protocol_rows": len(missing),
            "extra_dataset_files": len(set(by_file) - set(protocol_files)),
            "speaker_mismatches": len(speaker_mismatch),
        },
        "dataset_duplicate_files": {f: [(d["shard"], d["row"]) for d in rs]
                                    for f, rs in by_file.items() if len(rs) > 1},
        "protocol_duplicate_files": {f: n for f, n in protocol_files.items() if n > 1},
        "enroll_test_overlap": sorted(enroll_files & test_files),
        "label_pairs": dict(pairs_label),
        "source_method_pairs": dict(pairs_source),
        "missing_protocol_rows": missing,
        "extra_dataset_files": sorted(set(by_file) - set(protocol_files)),
        "speaker_mismatches": speaker_mismatch,
        "dataset_label_counts": dict(Counter(d["label"] for d in dataset)),
        "dataset_person_counts": dict(Counter(d["person"] for d in dataset)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["counts"], indent=2))
    print(f"report {args.report}")


if __name__ == "__main__":
    main()
