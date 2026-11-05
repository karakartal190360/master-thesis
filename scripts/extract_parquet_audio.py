"""Write audio stored in Hugging Face Parquet shards to one flat folder of WAV files.

Bytes are written unchanged (no re-encoding, no resampling). Each file is decoded once
to check channels, frame count and duration; sample rates are recorded, and enforced
only if --sample-rate is given. Upstream resamples on load (librosa.load(sr=16000)).
A per-file manifest with SHA-256 is written as CSV, and a summary JSON report lists
every failed file with its reason.
"""

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

# Release's own duration_sec is float32; allow 1 ms of rounding.
DURATION_TOLERANCE = 0.001


def check_audio(data, expected_rate, expected_duration):
    audio, rate = sf.read(io.BytesIO(data), dtype="int16", always_2d=True)
    duration = audio.shape[0] / rate
    problems = []
    if expected_rate and rate != expected_rate:
        problems.append(f"sample_rate={rate}")
    if audio.shape[1] != 1:
        problems.append(f"channels={audio.shape[1]}")
    if audio.shape[0] == 0:
        problems.append("empty")
    if abs(duration - expected_duration) > DURATION_TOLERANCE:
        problems.append(f"duration={duration:.4f}!={expected_duration:.4f}")
    return rate, audio.shape[1], audio.shape[0], duration, problems


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="flat WAV folder")
    parser.add_argument("--manifest", type=Path, required=True, help="per-file CSV")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, help="fail files with another rate")
    parser.add_argument("--shards", nargs="*", help="limit to these shard names (for tests)")
    args = parser.parse_args()

    shards = sorted(args.parquet_dir.glob("*.parquet"))
    if args.shards:
        shards = [s for s in shards if s.name in args.shards]
    args.output.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "dataset_audio_extraction",
        "scientific_result": False,
        "command": [sys.executable, *sys.argv],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True).strip()),
        "soundfile": sf.__version__,
        "libsndfile": sf.__libsndfile_version__,
        "expected_sample_rate": args.sample_rate,
        "duration_tolerance_s": DURATION_TOLERANCE,
        "shards": [s.name for s in shards],
        "failed_files": [],
    }
    written, total_bytes, total_seconds, rates = 0, 0, 0.0, Counter()
    with args.manifest.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file", "shard", "row", "bytes", "sha256", "sample_rate",
                         "channels", "frames", "duration_s"])
        for shard in shards:
            print(f"Extracting {shard.name}", flush=True)
            parquet = pq.ParquetFile(shard)
            row = 0
            for group in range(parquet.num_row_groups):
                table = parquet.read_row_group(group, columns=["file_name", "duration_sec"])
                for record in table.to_pylist():
                    name = record["file_name"]["path"]
                    data = record["file_name"]["bytes"]
                    try:
                        rate, channels, frames, duration, problems = check_audio(
                            data, args.sample_rate, record["duration_sec"])
                    except Exception as error:  # undecodable bytes; record, do not skip silently
                        rate = channels = frames = duration = None
                        problems = [f"decode_error: {error}"]
                    if problems:
                        report["failed_files"].append(
                            {"file": name, "shard": shard.name, "row": row, "problems": problems})
                    # Written even if a check failed, so the runner sees and logs the trial.
                    (args.output / name).write_bytes(data)
                    writer.writerow([name, shard.name, row, len(data),
                                     hashlib.sha256(data).hexdigest(), rate, channels, frames,
                                     duration])
                    written += 1
                    total_bytes += len(data)
                    total_seconds += duration or 0.0
                    rates[str(rate)] += 1
                    row += 1

    report.update({
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not report["failed_files"] else "failed",
        "files_written": written,
        "total_bytes": total_bytes,
        "total_hours": total_seconds / 3600,
        "sample_rate_counts": dict(rates),
        "manifest": str(args.manifest),
    })
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['status']}: {written} files, {total_bytes / 1e9:.2f} GB, "
          f"{report['total_hours']:.1f} h, {len(report['failed_files'])} failed")
    sys.exit(0 if not report["failed_files"] else 1)


if __name__ == "__main__":
    main()
