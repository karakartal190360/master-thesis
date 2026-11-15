"""Count same-speaker German/English material in downloaded dataset metadata.

Planning check only (no audio). Expected layout under --root:
voxpopuli/asr_{de,en}.tsv.gz, voxceleb1-b/{vox1_lang_label.csv,list_test_bilingual.txt},
scdf/metadata.zip. Sources and hashes: docs/additional-datasets.md.
"""
import argparse
import csv
import gzip
import io
import struct
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

VCC2020 = "https://raw.githubusercontent.com/nii-yamagishilab/VCC2020-database/master/"
VCC2020_ZIPS = ["vcc2020_database_groundtruth.zip", "vcc2020_database_training_target_task2.zip"]

csv.field_size_limit(sys.maxsize)


def voxpopuli_minutes(root, lang):
    """Minutes per speaker; 'invalid' split rows and unknown speakers skipped."""
    minutes = defaultdict(float)
    path = root / f"voxpopuli/asr_{lang}.tsv.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="|", quoting=csv.QUOTE_NONE):
            if row["speaker_id"] == "None" or row["split"] == "invalid":
                continue
            minutes[row["speaker_id"]] += (float(row["end_time"]) - float(row["start_time"])) / 60
    return minutes


def report_voxpopuli(de, en):
    both = sorted(set(de) & set(en), key=lambda s: -min(de[s], en[s]))
    print(f"VoxPopuli speakers: de {len(de)}, en {len(en)}, both {len(both)}")
    for limit in (1, 2, 4, 10, 30):
        count = sum(min(de[s], en[s]) >= limit for s in both)
        print(f"  >= {limit} min in each language: {count}")
    for s in both[:15]:
        print(f"  {s}: de {de[s]:.1f} min, en {en[s]:.1f} min")


def report_voxceleb1b(root):
    file_lang = {}
    speaker_langs = defaultdict(Counter)
    with open(root / "voxceleb1-b/vox1_lang_label.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            file_lang[row["filename"]] = row["label"]
            speaker_langs[row["id"]][row["label"]] += 1
    de_en = sorted(s for s, c in speaker_langs.items() if c["german"] and c["english"])
    print(f"VoxCeleb1-B: {len(speaker_langs)} speakers, {len(de_en)} with German and English files")
    for s in de_en:
        print(f"  {s}: german {speaker_langs[s]['german']}, english {speaker_langs[s]['english']} files")
    trials = Counter()
    with open(root / "voxceleb1-b/list_test_bilingual.txt", encoding="utf-8") as f:
        for line in f:
            label, a, b = line.split()
            pair = tuple(sorted((file_lang.get(a, "?"), file_lang.get(b, "?"))))
            if "german" in pair:
                trials[(label, pair)] += 1
    print(f"  trials with German (label, languages): {dict(trials)}")


def report_scdf(root, vp_de, vp_en):
    """SCDF genuine minutes per speaker, linked to VoxPopuli by MEP ID."""
    with zipfile.ZipFile(root / "scdf/metadata.zip") as z:
        for lang in ("de", "en"):
            genuine = defaultdict(float)
            fakes = Counter()
            with z.open(f"metadata_{lang}.csv") as raw:
                for row in csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8")):
                    if row["tool"] == "real":
                        genuine[row["speaker_id"]] += float(row["duration"]) / 60
                    else:
                        fakes[row["speaker_id"]] += 1
            print(f"SCDF {lang}: {len(genuine)} speakers")
            for s in sorted(genuine):
                print(f"  {s}: SCDF genuine {genuine[s]:.1f} min, fakes {fakes[s]}; "
                      f"VoxPopuli de {vp_de.get(s, 0):.1f} min, en {vp_en.get(s, 0):.1f} min")


def fetch(url, start, end):
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request) as response:
        return response.read()


def remote_zip_entries(url):
    """Read only the zip central directory via HTTP range requests."""
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as response:
        size = int(response.headers["Content-Length"])
    tail = fetch(url, max(0, size - 65536), size - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    count, cd_size, cd_offset = struct.unpack("<HII", tail[eocd + 10:eocd + 20])
    directory = fetch(url, cd_offset, cd_offset + cd_size - 1)
    pos = 0
    for _ in range(count):
        file_size = struct.unpack("<I", directory[pos + 24:pos + 28])[0]
        name_len, extra_len, comment_len = struct.unpack("<HHH", directory[pos + 28:pos + 34])
        yield directory[pos + 46:pos + 46 + name_len].decode("utf-8", "replace"), file_size
        pos += 46 + name_len + extra_len + comment_len


def report_vcc2020():
    for name in VCC2020_ZIPS:
        files = Counter()
        minutes = Counter()
        for path, size in remote_zip_entries(VCC2020 + name):
            if path.endswith(".wav") and "__MACOSX" not in path:
                speaker = path.rsplit("/", 2)[-2]
                files[speaker] += 1
                minutes[speaker] += (size - 44) / (24000 * 2) / 60  # 24 kHz, 16-bit mono (estimate)
        print(f"VCC2020 {name}:")
        for speaker in sorted(files):
            print(f"  {speaker}: {files[speaker]} wav, about {minutes[speaker]:.1f} min")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/dataset-metadata"))
    parser.add_argument("--vcc2020", action="store_true", help="list remote VCC2020 zips")
    args = parser.parse_args()
    vp_de = voxpopuli_minutes(args.root, "de")
    vp_en = voxpopuli_minutes(args.root, "en")
    report_voxpopuli(vp_de, vp_en)
    report_voxceleb1b(args.root)
    report_scdf(args.root, vp_de, vp_en)
    if args.vcc2020:
        report_vcc2020()


if __name__ == "__main__":
    main()
