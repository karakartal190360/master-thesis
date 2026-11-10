"""Decode every protocol audio file once and record integrity measures per file.

Reads the files exactly as upstream finds them (protocol key, ".wav" rule, flat audio root).
Flags are descriptive: nothing is removed or changed, and protocols stay as published.
The per-file CSV includes a hash of the decoded samples, used by check_dataset_leakage.py.
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

from check_dataset_protocols import read_protocol, sha256

# Descriptive thresholds (our choice; upstream defines none).
SHORT_S = 0.5      # 25 CTC/MMS frames of 20 ms; few phonemes can be aligned
LONG_S = 30.0      # longest input in the GPU memory check (docs/environment.md)
LOW_LEVEL_DBFS = -60.0
CLIP_LEVEL = 0.999
CLIP_FRACTION = 0.001
COLUMNS = ["file", "split", "speaker", "source", "label", "sample_rate", "channels", "frames",
           "duration_s", "peak", "rms_dbfs", "clip_fraction", "sample_sha256", "flags", "error"]


def measure(path):
    try:
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception as error:  # recorded per file, never skipped silently
        return {"error": f"{type(error).__name__}: {error}"}
    frames, channels = audio.shape
    # Same samples in another container still give the same hash.
    digest = hashlib.sha256(f"{rate},{frames},{channels};".encode())
    digest.update(np.ascontiguousarray(audio).tobytes())
    out = {"sample_rate": rate, "channels": channels, "frames": frames,
           "duration_s": frames / rate, "sample_sha256": digest.hexdigest()}
    flags = []
    if frames == 0:
        flags.append("empty")
    elif not np.isfinite(audio).all():
        flags.append("non_finite")
    else:
        # upstream librosa.load(mono=True) averages channels before analysis
        mono = audio.mean(axis=1, dtype=np.float64)
        peak = float(np.abs(audio).max())
        rms = float(np.sqrt(np.mean(mono ** 2)))
        out["peak"] = round(peak, 6)
        out["rms_dbfs"] = round(20 * np.log10(rms), 2) if rms > 0 else float("-inf")
        out["clip_fraction"] = round(float((np.abs(audio) >= CLIP_LEVEL).mean()), 6)
        if peak == 0:
            flags.append("digital_silence")
        elif out["rms_dbfs"] < LOW_LEVEL_DBFS:
            flags.append("low_level")
        if peak > 1.0:
            flags.append("above_full_scale")  # float WAV only; librosa does not clip
        if out["clip_fraction"] > CLIP_FRACTION:
            flags.append("clipping")
    if 0 < out["duration_s"] < SHORT_S:
        flags.append("short")
    if out["duration_s"] > LONG_S:
        flags.append("long")
    out["flags"] = flags
    return out


def check_file(args):
    root, row = args
    path = root / row["file"]
    if not path.is_file():
        return {**row, "error": "missing_file", "flags": []}
    return {**row, **measure(path)}


def duration_summary(values):
    values = np.array(values)
    return {"files": len(values), "hours": round(values.sum() / 3600, 2),
            "min": round(values.min(), 3), "median": round(float(np.median(values)), 3),
            "p99": round(float(np.percentile(values, 99)), 3), "max": round(values.max(), 3)}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--enroll", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="per-file CSV")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    protocol = read_protocol(args.enroll, "enroll") + read_protocol(args.test, "test")
    started = datetime.now(timezone.utc).isoformat()
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool, args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        jobs = ((args.audio_root, row) for row in protocol)
        for n, r in enumerate(pool.imap(check_file, jobs, chunksize=64), 1):
            writer.writerow({**r, "flags": ";".join(r["flags"])})
            results.append(r)
            if n % 20000 == 0:
                print(f"{n}/{len(protocol)}", flush=True)

    flag_counts = defaultdict(Counter)
    errors = Counter()
    durations = defaultdict(list)
    for r in results:
        if r.get("error"):
            errors[r["error"].split(":")[0]] += 1
            continue
        durations[r["source"] or "(bona fide)"].append(r["duration_s"])
        for flag in r["flags"]:
            flag_counts[flag][f"{r['split']} {r['label']} {r['source'] or '-'}"] += 1

    report = {
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "dataset_audio_integrity_check",
        "scientific_result": False,
        "command": [sys.executable, *sys.argv],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True).strip()),
        "script_sha256": {p.name: sha256(p) for p in
                          (Path(__file__), Path(__file__).with_name("check_dataset_protocols.py"))},
        "protocol_sha256": {str(args.enroll): sha256(args.enroll), str(args.test): sha256(args.test)},
        "audio_root": str(args.audio_root.resolve()),
        "soundfile": sf.__version__,
        "libsndfile": sf.__libsndfile_version__,
        "numpy": np.__version__,
        "decode": "soundfile float32, all channels; level measures on channel mean",
        "thresholds": {"short_s": SHORT_S, "long_s": LONG_S, "low_level_dbfs": LOW_LEVEL_DBFS,
                       "clip_level": CLIP_LEVEL, "clip_fraction": CLIP_FRACTION},
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "protocol_rows": len(protocol),
        "decoded_files": sum(not r.get("error") for r in results),
        "errors": dict(errors),
        "error_files": [r["file"] for r in results if r.get("error")],
        "flag_totals": {f: sum(c.values()) for f, c in sorted(flag_counts.items())},
        "flags_by_split_label_source": {f: dict(sorted(c.items())) for f, c in sorted(flag_counts.items())},
        "duration_by_source": {s: duration_summary(v) for s, v in sorted(durations.items())},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("decoded_files", "errors", "flag_totals")}, indent=2))


if __name__ == "__main__":
    main()
