"""Look for enrollment/test leakage in a benchmark, using hashes and metadata only.

1. Identical audio: files with the same decoded samples (hash from check_dataset_audio.py),
   grouped by split, speaker and label. Byte identity is reported from the WAV manifest.
2. ZH-Famous: test utterances whose transcript equals an enrollment transcript of the speaker.
3. Famous Figures: spoof files whose name gives the bona fide clip used as voice prompt
   (inferred naming rule, see docs/reproduction.md); is that clip an enrollment file?
Protocols are not changed; the results describe the benchmark as published.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from check_dataset_protocols import sha256


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def key(r):
    return f"{r['split']} {r['label']}"


def identical_audio(rows, byte_hash):
    groups = defaultdict(list)
    for r in rows:
        if r["sample_sha256"]:
            groups[r["sample_sha256"]].append(r)
    groups = {h: g for h, g in groups.items() if len(g) > 1}

    kinds = Counter()
    test_same_speaker, test_other_speaker = Counter(), Counter()
    enroll_files, per_speaker = set(), defaultdict(Counter)
    for g in groups.values():
        splits = "+".join(sorted({r["split"] for r in g}))
        labels = "+".join(sorted({r["label"] for r in g}))
        kinds[f"{splits} | {labels} | speakers={len({r['speaker'] for r in g})}"] += 1
        enrolled = {r["speaker"] for r in g if r["split"] == "enroll"}
        for r in g:
            if r["split"] == "enroll":
                if any(t["split"] == "test" and t["speaker"] == r["speaker"] for t in g):
                    enroll_files.add(r["file"])
                    per_speaker[r["speaker"]]["enroll_files"] += 1
            elif r["speaker"] in enrolled:
                test_same_speaker[r["label"]] += 1
                per_speaker[r["speaker"]][f"test_{r['label']}"] += 1
            elif enrolled:
                test_other_speaker[r["label"]] += 1

    summary = {
        "groups": len(groups),
        "files": sum(len(g) for g in groups.values()),
        "groups_not_byte_identical": sum(len({byte_hash[r["file"]] for r in g}) > 1
                                         for g in groups.values()),
        "group_size_counts": dict(sorted(Counter(len(g) for g in groups.values()).items())),
        "groups_by_splits_labels_speakers": dict(sorted(kinds.items())),
        "groups_with_mixed_labels": sum(len({r["label"] for r in g}) > 1 for g in groups.values()),
        "groups_with_several_speakers": sum(len({r["speaker"] for r in g}) > 1
                                            for g in groups.values()),
        "enroll_files_also_in_test_same_speaker": len(enroll_files),
        "test_trials_equal_to_enroll_same_speaker": dict(test_same_speaker),
        "test_trials_equal_to_enroll_other_speaker": dict(test_other_speaker),
        "per_speaker_enroll_test": {s: dict(c) for s, c in sorted(per_speaker.items())},
    }
    return groups, summary


def write_groups(path, groups, byte_hash):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "sample_sha256", "file_sha256", "file", "split", "speaker",
                         "source", "label"])
        for n, (h, g) in enumerate(sorted(groups.items(), key=lambda x: x[1][0]["file"])):
            for r in g:
                writer.writerow([n, h, byte_hash[r["file"]], r["file"], r["split"], r["speaker"],
                                 r["source"], r["label"]])


def same_transcript(rows, parquet_dir):
    text = {}
    for shard in sorted(parquet_dir.glob("*.parquet")):
        table = pq.ParquetFile(shard).read(columns=["file_name.path", "transcription"])
        for r in table.to_pylist():
            text[r["file_name"]["path"]] = (r["transcription"] or "").strip()
    enrolled = defaultdict(set)
    for r in rows:
        if r["split"] == "enroll" and text.get(r["file"]):
            enrolled[r["speaker"]].add(text[r["file"]])
    matches = Counter()
    for r in rows:
        if r["split"] == "test" and text.get(r["file"]) in enrolled[r["speaker"]]:
            matches[f"{r['label']} {r['source'] or '-'}"] += 1
    return {
        "rule": "exact string equality after stripping outer whitespace; same speaker",
        "empty_or_missing_transcripts": dict(Counter(key(r) for r in rows if not text.get(r["file"]))),
        "enroll_transcripts_unique": sum(len(s) for s in enrolled.values()),
        "test_trials_with_enroll_transcript": dict(sorted(matches.items())),
    }


def prompt_clip(r, release_path):
    # e.g. "X_00014_FISHSPEECH__Anthony_Blinken_02097.wav", LLASA: "X_01037_LLASA_06562.wav"
    pattern = rf"_{re.escape(r['source'])}_{{1,2}}(?:{re.escape(r['speaker'])}_)?(\d{{5}})\.wav$"
    found = re.search(pattern, release_path)
    return f"{r['speaker']}_{found.group(1)}" if found else None


def voice_prompts(rows, release_path):
    stem = {r["file"]: release_path[r["file"]].removesuffix(".wav")
            for r in rows if r["label"] == "bonafide"}
    enroll = {stem[r["file"]] for r in rows if r["split"] == "enroll"}
    test_bonafide = {stem[r["file"]] for r in rows if r["split"] == "test" and r["file"] in stem}
    counts, sources, prompts = Counter(), defaultdict(Counter), defaultdict(Counter)
    for r in rows:
        if r["label"] != "spoof":
            continue
        clip = prompt_clip(r, release_path[r["file"]])
        where = ("no_prompt_in_name" if clip is None else "enroll" if clip in enroll
                 else "test_bonafide" if clip in test_bonafide else "not_in_protocols")
        counts[where] += 1
        sources[where][r["source"]] += 1
        if clip:
            prompts[r["speaker"]][f"{clip} ({where})"] += 1
    return {
        "rule": "name ends with _<Source>_ or _<Source>__, optional <Speaker>_, five digits",
        "test_spoof_prompt_location": dict(counts),
        "by_source": {w: dict(sorted(c.items())) for w, c in sorted(sources.items())},
        "prompt_clips_per_speaker": {s: dict(c.most_common()) for s, c in sorted(prompts.items())},
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["zh-famous", "famous-figures"], required=True)
    parser.add_argument("--audio-check", type=Path, required=True, help="check_dataset_audio.py CSV")
    parser.add_argument("--manifest", type=Path, required=True, help="WAV manifest with file SHA-256")
    parser.add_argument("--parquet-dir", type=Path, help="release shards (ZH-Famous transcripts)")
    parser.add_argument("--groups", type=Path, required=True, help="identical-audio groups CSV")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows = read_csv(args.audio_check)
    manifest = read_csv(args.manifest)
    byte_hash = {m["file"]: m["sha256"] for m in manifest}
    groups, identical = identical_audio(rows, byte_hash)
    write_groups(args.groups, groups, byte_hash)

    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "dataset_leakage_check",
        "scientific_result": False,
        "dataset": args.dataset,
        "command": [sys.executable, *sys.argv],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True).strip()),
        "script_sha256": {Path(__file__).name: sha256(Path(__file__))},
        "inputs_sha256": {str(args.audio_check): sha256(args.audio_check),
                          str(args.manifest): sha256(args.manifest)},
        "groups": str(args.groups),
        "groups_sha256": sha256(args.groups),
        "rows": len(rows),
        "rows_without_audio": sum(not r["sample_sha256"] for r in rows),
        "identical_audio": identical,
    }
    if args.dataset == "zh-famous":
        report["same_transcript"] = same_transcript(rows, args.parquet_dir)
    else:
        release_path = {m["file"]: m["release_path"] for m in manifest}
        report["voice_prompts"] = voice_prompts(rows, release_path)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k in
                      ("identical_audio", "same_transcript", "voice_prompts")},
                     indent=2, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
