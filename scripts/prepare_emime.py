"""Prepare EMIME talkers for a cross-language PVP evaluation (our protocol, not Xue's).

Groups: German talkers (German/English, with clones) and Finnish talkers (Finnish/English,
genuine only; confirmation set of the analysis plan, Section 7).

1. Manifest: decode every German-talker WAV (both microphones, both languages) once:
   byte SHA-256, sample hash, rate, channels, duration, level flags (check_dataset_audio.py).
2. Counts against the README (14 talkers x 2 languages x 145 sentences x 2 microphones)
   and groups of identical audio.
3. Sentence split, fixed seed, stratified by prompt type: enrollment 60, clone
   reference 10, test 75 sentences. Same sentence numbers for all talkers and both
   languages: Europarl sentences 1-25 are translations of each other in the German
   and English prompts, so a translation never crosses enrollment and test.
4. PVP protocols (speaker,path,source,label), microphone 0 only (both microphones
   recorded the same takes): one enrollment list per language and one shared test
   list. Paths are relative to data/emime. For German talkers the test list names the
   clone files that scripts/generate_clones.py writes. Files missing from the release
   are left out and listed (missing.csv).
"""

import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/emime"
RELEASE = "v1/UEDIN_bilingual_data_2010"
CODES = {"de": "Ger", "en": "Eng", "fi": "Fin"}
PROMPT_FILES = {"de": "Prompts/German/german_prompts.txt", "en": "Prompts/English/english_prompts.txt",
                "fi": "Prompts/Finnish/finnish_prompts.txt"}
SEED = 20261115
# README section 3: 25 Europarl, 100 news, 20 SUS sentences, in this order (checked in
# main against the prompt count). Counts per part: enrollment, reference, test.
TYPES = [("europarl", range(1, 26), (10, 2, 13)),
         ("news", range(26, 126), (41, 7, 52)),
         ("sus", range(126, 146), (9, 1, 10))]
MIC = "0"  # close-talking DPA 4035
# Defective genuine files found by this check (2026-11-15), excluded before any scoring.
EXCLUDE = {
    ("GM3", "de", 1): "audio identical to GM1_GER_0001_0; mic-1 envelope correlation with "
                      "GM1 0.91, with GM3 0.35: release error, recording is GM1",
    ("GF3", "en", 69): "near-silent on both microphones (mic 0 -69.4 dBFS, 0.73 s for a full sentence)",
    ("GF3", "en", 94): "near-silent on both microphones (mic 0 -71.3 dBFS, mic 1 -59.5 dBFS)",
}
# Finnish talkers: the check (2026-11-16) found no defective file (no identical audio,
# no near-silent file); FM4 and FM7 lack microphone-0 Finnish sentences 5-54 (missing.csv).
EXCLUDE_FINNISH = {}
GROUPS = {
    "german": {"folder": "German_talkers", "prefix": "G", "langs": ["de", "en"], "clones": True,
               "protocols": "protocols/emime", "exclude": EXCLUDE},
    "finnish": {"folder": "Finnish_talkers", "prefix": "F", "langs": ["fi", "en"], "clones": False,
                "protocols": "protocols/emime/finnish", "exclude": EXCLUDE_FINNISH},
}
FOLDERS = {g["prefix"]: g["folder"] for g in GROUPS.values()}
CLONE_SYSTEM = "qwen3tts"
NAME = re.compile(r"([GF][FM]\d)_(GER|ENG|FIN)_(\d{4})_([01])\.wav$")


def talkers(group):
    return [f"{GROUPS[group]['prefix']}{g}{i}" for g in "FM" for i in range(1, 8)]


def wav_path(talker, lang, sentence, mic=MIC):
    gender = "Female" if talker[1] == "F" else "Male"
    code = CODES[lang]
    name = f"{talker}_{code.upper()}_{sentence:04d}_{mic}.wav"
    return f"{RELEASE}/downsampled_22kHz/{FOLDERS[talker[0]]}/{gender}/{talker}/{code}/{name}"


def clone_path(talker, lang, sentence, ref_lang):
    # "synthetic" in every clone file name (docs/additional-datasets.md)
    name = f"{talker}_{lang.upper()}_{sentence:04d}_{CLONE_SYSTEM}_ref{ref_lang}_synthetic.wav"
    return f"clones/{CLONE_SYSTEM}/{talker}/{name}"


def read_prompts(lang):
    """Sentence number -> text. Lines: '<number> <text>' (tab or spaces)."""
    prompts = {}
    path = DATA / RELEASE / PROMPT_FILES[lang]
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        match = re.match(r"\s*(\d+)[\s.:]+(.+?)\s*$", line)
        if match:
            prompts[int(match.group(1))] = match.group(2)
    return prompts


def split_sentences():
    rng = random.Random(SEED)
    parts = {}
    for kind, numbers, (n_enroll, n_ref, _) in TYPES:
        order = list(numbers)
        rng.shuffle(order)
        for i, sentence in enumerate(order):
            part = "enroll" if i < n_enroll else "reference" if i < n_enroll + n_ref else "test"
            parts[sentence] = (kind, part)
    return parts


def check_file(path):
    from check_dataset_audio import measure  # local: generate_clones.py imports read_prompts only

    rel = str(path.relative_to(DATA))
    talker, code, sentence, mic = NAME.search(path.name).groups()
    language = {c.upper(): lang for lang, c in CODES.items()}[code]
    row = {"file": rel, "speaker": talker, "language": language,
           "sentence": int(sentence), "mic": mic,
           "byte_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {**row, **measure(path)}


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def enroll_rows(group, enroll_lang, parts, present):
    return [[talker, wav_path(talker, enroll_lang, sentence), "-", "bonafide"]
            for talker in talkers(group) for sentence, (_, part) in sorted(parts[enroll_lang].items())
            if part == "enroll" and wav_path(talker, enroll_lang, sentence) in present]


def test_rows(group, parts, present):
    """One test list for both directions; source names language and condition."""
    test, langs, exclude = [], GROUPS[group]["langs"], GROUPS[group]["exclude"]
    for talker in talkers(group):
        for lang in langs:  # genuine tests in both languages
            for sentence, (_, part) in sorted(parts[lang].items()):
                path = wav_path(talker, lang, sentence)
                if part == "test" and (talker, lang, sentence) not in exclude and path in present:
                    test.append([talker, path, f"{lang}-genuine", "bonafide"])
        if not GROUPS[group]["clones"]:
            continue
        for lang in langs:  # clones in both languages, from references in both languages
            for ref_lang in langs:
                for sentence, (_, part) in sorted(parts[lang].items()):
                    if part == "test":
                        test.append([talker, clone_path(talker, lang, sentence, ref_lang),
                                     f"{lang}-{CLONE_SYSTEM}-ref{ref_lang}", "spoof"])
    return test


def leakage(tables, rows, langs):
    """Identical audio or file names shared between enrollment and genuine test files."""
    by_file = {r["file"]: r.get("sample_sha256") for r in rows}
    test = {r[1] for r in tables["emime_test.csv"] if r[3] == "bonafide"}
    result = {}
    for lang in langs:
        enroll = {r[1] for r in tables[f"emime_{lang}_enroll.csv"]}
        shared_hash = {by_file[f] for f in enroll} & {by_file[f] for f in test}
        result[lang] = {"shared_files": len(enroll & test), "shared_audio": len(shared_hash)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--group", choices=GROUPS, default="german")
    args = parser.parse_args()
    from check_dataset_audio import duration_summary

    group = GROUPS[args.group]
    langs, exclude = group["langs"], group["exclude"]
    out = ROOT / "results/datasets"
    protocols = ROOT / group["protocols"]
    stem = f"emime-{args.group}-talkers"

    files = sorted((DATA / RELEASE / "downsampled_22kHz" / group["folder"]).rglob("*.wav"))
    with Pool(args.workers) as pool:
        rows = pool.map(check_file, files, chunksize=32)
    columns = ["file", "speaker", "language", "sentence", "mic", "byte_sha256", "sample_rate",
               "channels", "frames", "duration_s", "peak", "rms_dbfs", "clip_fraction",
               "sample_sha256", "flags", "error"]
    write_csv(out / f"{stem}-manifest.csv",
              [{**r, "flags": ";".join(r.get("flags", []))} for r in rows], columns)

    present = {(r["speaker"], r["language"], r["sentence"], r["mic"]) for r in rows}
    expected = {(t, l, s, m) for t in talkers(args.group) for l in langs for s in range(1, 146) for m in "01"}
    present_files = {r["file"] for r in rows if r["mic"] == MIC and not r.get("error")}
    groups = defaultdict(list)
    for r in rows:
        if "sample_sha256" in r:
            groups[r["sample_sha256"]].append(r["file"])
    identical = [sorted(g) for g in groups.values() if len(g) > 1]

    prompts = {lang: read_prompts(lang) for lang in langs}
    for lang, texts in prompts.items():
        if sorted(texts) != list(range(1, 146)):
            sys.exit(f"{lang} prompts: expected sentences 1-145, found {len(texts)}")
    split = split_sentences()
    parts = {lang: split for lang in langs}
    # No prompt texts in Git (English news sentences come from LDC's WSJ1); read them locally.
    split_rows = [{"language": lang, "sentence": s, "type": kind, "part": part}
                  for lang in langs for s, (kind, part) in sorted(parts[lang].items())]
    write_csv(protocols / "sentence_split.csv", split_rows, ["language", "sentence", "type", "part"])
    header = ["Speaker", "AudioName", "Source", "Label"]  # same header as upstream protocols
    tables = {f"emime_{lang}_enroll.csv": enroll_rows(args.group, lang, parts, present_files) for lang in langs}
    tables["emime_test.csv"] = test_rows(args.group, parts, present_files)
    excluded = [{"speaker": t, "language": l, "sentence": s, "file": wav_path(t, l, s),
                 "part": parts[l][s][1], "reason": why} for (t, l, s), why in exclude.items()]
    for row in excluded:  # rule: only test files may be excluded; anything else needs a new decision
        if row["part"] != "test":
            sys.exit(f"excluded file is not a test file: {row}")
    write_csv(protocols / "excluded.csv", excluded, ["speaker", "language", "sentence", "file", "part", "reason"])
    missing = [{"speaker": t, "language": l, "sentence": s, "file": wav_path(t, l, s), "part": parts[l][s][1]}
               for t, l, s, m in sorted(expected - present) if m == MIC]
    if missing:  # absent from the release; never in a protocol
        write_csv(protocols / "missing.csv", missing, ["speaker", "language", "sentence", "file", "part"])
    for name, table in tables.items():
        with (protocols / name).open("w", newline="") as stream:
            csv.writer(stream).writerows([header, *table])

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "emime_dataset_check_and_protocol", "scientific_result": False,
        "command": [sys.executable, *sys.argv],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        "seed": SEED, "microphone": MIC, "split_counts": {k: c for k, _, c in TYPES},
        "group": args.group, "files": len(rows), "expected": len(expected),
        "missing": sorted(map(list, expected - present)), "extra": sorted(map(list, present - expected)),
        "decode_errors": [r["file"] for r in rows if r.get("error")],
        "flags": Counter(f"{f} {r['speaker']} mic{r['mic']}" for r in rows for f in r.get("flags", [])),
        "sample_rates": Counter(r.get("sample_rate") for r in rows),
        "channels": Counter(r.get("channels") for r in rows),
        "duration_s": {f"{lang} mic{m}": duration_summary([r["duration_s"] for r in rows if
                       r["language"] == lang and r["mic"] == m and "duration_s" in r])
                       for lang in langs for m in "01"},
        "identical_audio_groups": identical,
        "excluded": excluded, "leakage": leakage(tables, rows, langs),
        "protocol_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(protocols.glob("*.csv"))},
    }
    (out / f"{stem}-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("files", "expected", "decode_errors", "flags",
                                             "sample_rates", "channels", "duration_s")}, indent=2))
    print("leakage:", report["leakage"])
    print(f"missing {len(report['missing'])}, extra {len(report['extra'])}, "
          f"identical groups {len(identical)}")


if __name__ == "__main__":
    main()
