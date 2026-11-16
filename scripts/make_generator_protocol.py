"""Derive the EMIME test list of another clone generator (analysis plan, Section 7).

Same rows as emime_test.csv: genuine rows unchanged; in clone rows the system name
(path, file name, Source) is replaced, e.g. qwen3tts -> voxcpm2.
"""

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ROOT / "protocols/emime"
BASE = "qwen3tts"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", choices=["voxcpm2", "xtts"])
    args = parser.parse_args()
    with (PROTOCOLS / "emime_test.csv").open(newline="") as stream:
        header, *rows = list(csv.reader(stream))
    for row in rows:
        if row[3] == "spoof":
            assert f"/{BASE}/" in row[1] and f"-{BASE}-" in row[2]
            row[1] = row[1].replace(f"/{BASE}/", f"/{args.system}/").replace(f"_{BASE}_", f"_{args.system}_")
            row[2] = row[2].replace(f"-{BASE}-", f"-{args.system}-")
    out = PROTOCOLS / f"emime_test_{args.system}.csv"
    with out.open("w", newline="") as stream:
        csv.writer(stream).writerows([header, *rows])
    print(f"{len(rows)} rows written to {out}")


if __name__ == "__main__":
    main()
