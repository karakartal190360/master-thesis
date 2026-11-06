"""Write audio stored in Hugging Face Parquet shards to one flat folder of WAV files.

Bytes are written unchanged (no re-encoding, no resampling). Each file is decoded once
to check channels, frames and, where the release has it, duration; sample rates are
recorded per source system, and enforced only if --sample-rate is given. Upstream
resamples on load (librosa.load(sr=16000)). File names are the release paths, or the
protocol names from the --map CSV of check_dataset_protocols.py (Famous Figures, where
release paths repeat across systems). A per-file manifest with SHA-256 is written as CSV,
and a summary JSON report lists every failed file with its reason.
"""

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

from parquet_datasets import DATASETS, read_rows

# ZH-Famous duration_sec is float32; allow 1 ms of rounding.
DURATION_TOLERANCE = 0.001


def check_audio(data, expected_rate, expected_duration):
    info = sf.info(io.BytesIO(data))
    audio, rate = sf.read(io.BytesIO(data), dtype="int16", always_2d=True)
    duration = audio.shape[0] / rate
    problems = []
    if expected_rate and rate != expected_rate:
        problems.append(f"sample_rate={rate}")
    if audio.shape[1] != 1:
        problems.append(f"channels={audio.shape[1]}")
    if audio.shape[0] == 0:
        problems.append("empty")
    if expected_duration is not None and abs(duration - expected_duration) > DURATION_TOLERANCE:
        problems.append(f"duration={duration:.4f}!={expected_duration:.4f}")
    return info.format, info.subtype, rate, audio.shape[1], audio.shape[0], duration, problems


def read_map(path):
    with path.open(newline="") as stream:
        return {(r["shard"], int(r["row"])): r for r in csv.DictReader(stream)}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="flat WAV folder")
    parser.add_argument("--manifest", type=Path, required=True, help="per-file CSV")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--map", type=Path, help="protocol file -> shard/row CSV")
    parser.add_argument("--sample-rate", type=int, help="fail files with another rate")
    parser.add_argument("--shards", nargs="*", help="limit to these shard names (for tests)")
    args = parser.parse_args()

    shards = sorted(args.parquet_dir.glob("*.parquet"))
    if args.shards:
        shards = [s for s in shards if s.name in args.shards]
    names = read_map(args.map) if args.map else None
    args.output.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "dataset_audio_extraction",
        "scientific_result": False,
        "dataset": args.dataset,
        "columns": DATASETS[args.dataset],
        "command": [sys.executable, *sys.argv],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True).strip()),
        "script_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          (Path(__file__), Path(__file__).with_name("parquet_datasets.py"))},
        "soundfile": sf.__version__,
        "libsndfile": sf.__libsndfile_version__,
        "map": str(args.map) if args.map else None,
        "map_sha256": hashlib.sha256(args.map.read_bytes()).hexdigest() if args.map else None,
        "expected_sample_rate": args.sample_rate,
        "duration_tolerance_s": DURATION_TOLERANCE,
        "shards": [s.name for s in shards],
        "failed_files": [],
    }
    written, total_bytes, total_seconds = 0, 0, 0.0
    rates, formats, rates_by_source = Counter(), Counter(), defaultdict(Counter)
    seen = set()
    with args.manifest.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file", "shard", "row", "release_path", "source", "bytes", "sha256",
                         "format", "subtype", "sample_rate", "channels", "frames", "duration_s"])
        for shard in shards:
            print(f"Extracting {shard.name}", flush=True)
            for record in read_rows(shard, args.dataset, audio=True):
                where = {"shard": shard.name, "row": record["row"], "release_path": record["file"]}
                name = record["file"]
                if names is not None:
                    entry = names.get((shard.name, record["row"]))
                    if entry is None or entry["release_path"] != record["file"]:
                        report["failed_files"].append({**where, "problems": ["not_in_map"]})
                        continue
                    name = entry["file"]
                # Flat folder: a repeated name or a sub-path would overwrite or escape it.
                if name in seen or Path(name).name != name:
                    report["failed_files"].append({**where, "file": name, "problems": ["name_conflict"]})
                    continue
                seen.add(name)
                data = record["bytes"]
                try:
                    fmt, subtype, rate, channels, frames, duration, problems = check_audio(
                        data, args.sample_rate, record["duration"])
                except Exception as error:  # undecodable bytes; record, do not skip silently
                    fmt = subtype = rate = channels = frames = duration = None
                    problems = [f"decode_error: {error}"]
                if problems:
                    report["failed_files"].append({**where, "file": name, "problems": problems})
                # Written even if a check failed, so the runner sees and logs the trial.
                (args.output / name).write_bytes(data)
                source = record["source"] or "(empty)"
                writer.writerow([name, shard.name, record["row"], record["file"], source, len(data),
                                 hashlib.sha256(data).hexdigest(), fmt, subtype, rate, channels,
                                 frames, duration])
                written += 1
                total_bytes += len(data)
                total_seconds += duration or 0.0
                rates[str(rate)] += 1
                formats[f"{fmt}/{subtype}"] += 1
                rates_by_source[source][str(rate)] += 1

    if names is not None:
        report["map_rows_not_written"] = len(set(e["file"] for e in names.values()) - seen)
    report.update({
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not report["failed_files"] and not report.get("map_rows_not_written")
                  else "failed",
        "files_written": written,
        "total_bytes": total_bytes,
        "total_hours": total_seconds / 3600,
        "sample_rate_counts": dict(rates),
        "sample_rate_by_source": {s: dict(c) for s, c in sorted(rates_by_source.items())},
        "format_counts": dict(formats),
        "manifest": str(args.manifest),
    })
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['status']}: {written} files, {total_bytes / 1e9:.2f} GB, "
          f"{report['total_hours']:.1f} h, {len(report['failed_files'])} failed")
    sys.exit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
