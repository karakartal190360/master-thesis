"""Derive the wrong-speaker control test list for EMIME (analysis plan, Section 5).

Same rows as emime_test.csv; only the Speaker column (the profile that scores the trial)
is replaced by the next talker of the same sex: GF1 -> GF2, ..., GF7 -> GF1, the same
for GM. The true talker stays visible in the file name.
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ROOT / "protocols/emime"


def rotation():
    return {f"G{sex}{i}": f"G{sex}{i % 7 + 1}" for sex in "FM" for i in range(1, 8)}


def main():
    profile = rotation()
    with (PROTOCOLS / "emime_test.csv").open(newline="") as stream:
        header, *rows = list(csv.reader(stream))
    assert header == ["Speaker", "AudioName", "Source", "Label"]
    for row in rows:
        assert f"/{row[0]}/" in row[1] or f"{row[0]}_" in Path(row[1]).name  # talker in path
        row[0] = profile[row[0]]
    with (PROTOCOLS / "emime_test_wrong_speaker.csv").open("w", newline="") as stream:
        csv.writer(stream).writerows([header, *rows])
    print(f"{len(rows)} rows, rotation {profile}")


if __name__ == "__main__":
    main()
