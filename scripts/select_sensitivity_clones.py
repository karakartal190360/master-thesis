"""List the clones left out in the EMIME sensitivity analysis (analysis plan, Section 2).

The primary analysis keeps every clone. The sensitivity analysis leaves out a clone if
Whisper detects another language than the target language, or if its character error
rate is more than 0.10 above that of the matched genuine file (same talker, language,
sentence; 0 if the genuine file is excluded). Relative to genuine speech, because an
absolute word error limit flags genuine files as often as clones (numbers, abbreviations).
"""

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CER_MARGIN = 0.10


def select(rows):
    genuine = {(r["speaker"], r["language"], r["sentence"]): float(r["cer"])
               for r in rows if r["kind"] == "genuine"}
    selected = []
    for r in rows:
        if r["kind"] != "clone":
            continue
        excess = float(r["cer"]) - genuine.get((r["speaker"], r["language"], r["sentence"]), 0.0)
        reasons = []
        if r["detected_language"] != r["language"]:
            reasons.append(f"detected language {r['detected_language']}")
        if excess > CER_MARGIN:
            reasons.append(f"CER {excess:.3f} above genuine")
        if reasons:
            selected.append({"file": r["file"], "speaker": r["speaker"], "language": r["language"],
                             "ref_language": r["ref_language"], "reason": "; ".join(reasons)})
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("quality", type=Path, help="results/clone-quality/<run id>/quality.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "protocols/emime/sensitivity_clones.csv")
    args = parser.parse_args()
    selected = select(list(csv.DictReader(args.quality.open(encoding="utf-8"))))
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["file", "speaker", "language", "ref_language", "reason"])
        writer.writeheader()
        writer.writerows(selected)
    print(f"{len(selected)} clones listed in {args.output}")


if __name__ == "__main__":
    main()
