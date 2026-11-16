"""Prepare the VCC2020 Task 2 confirmation set (analysis plan, Section 7, addendum C).

1. Map the two German VCC2020 targets to EMIME talkers: waveform correlation (16 kHz)
   of their English ground-truth files with every EMIME German talker's recording of
   the same sentence.
2. Map the 25 English evaluation sentences to EMIME sentence numbers and our split.
3. Protocols (Speaker,AudioName,Source,Label; paths relative to data/): our EMIME
   enrollment lists limited to the two talkers; test list = their genuine EMIME test
   files (both languages) and all submitted English conversions into them.
No transcript text is written (some EMIME English prompts come from LDC's WSJ1).
"""

import csv
import json
import re
import unicodedata
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import correlate

from prepare_emime import DATA as EMIME, read_prompts, wav_path

ROOT = Path(__file__).resolve().parents[1]
VCC = ROOT / "data/vcc2020"
EXTRACTED = VCC / "database/extracted"
SUBMITTED = VCC / "submitted/VCC2020_submitted_wavs"
OUT = ROOT / "protocols/vcc2020"
TARGETS = ["TGF1", "TGM1"]
TALKERS = [f"G{g}{i}" for g in "FM" for i in range(1, 8)]
CHECK_SENTENCES = ["30001", "30002", "30003"]
NAME = re.compile(r"(TG[FM]1)_(SE[FM]\d)_E(\d{5})\.wav$")


def norm(text):
    return re.sub(r"[^\w]+", " ", unicodedata.normalize("NFKC", text).lower()).strip()


def evaluation_sentences():
    """VCC evaluation ID (e.g. 30001) -> EMIME English sentence number."""
    path = EXTRACTED / "vcc2020_database_transcriptions/transcriptions_evaluation/vcc2020_database_evaluation_transcriptions.txt"
    prompts = {norm(t): n for n, t in read_prompts("en").items()}
    mapping = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*(\d{5})\s+(.+?)\s*$", line)
        if match:
            mapping[match.group(1)] = prompts[norm(match.group(2))]  # KeyError: no EMIME match
    return mapping


def load16(path):
    audio, rate = sf.read(path)
    audio = audio if audio.ndim == 1 else audio.mean(axis=1)
    return librosa.resample(audio, orig_sr=rate, target_sr=16000)


def correlation(x, y):
    """Peak normalized cross-correlation (1 = same recording, any time shift)."""
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    return float(np.max(np.abs(correlate(y, x, mode="full", method="fft"))) / min(len(x), len(y)))


def map_targets(sentences):
    result = {}
    for target in TARGETS:
        scores = {}
        for uid in CHECK_SENTENCES:
            vcc = load16(EXTRACTED / f"vcc2020_database_groundtruth/{target}/E{uid}.wav")
            for talker in TALKERS:
                emime = load16(EMIME / wav_path(talker, "en", sentences[uid]))
                scores.setdefault(talker, []).append(round(correlation(vcc, emime), 3))
        ranked = sorted(scores, key=lambda t: -np.mean(scores[t]))
        best, second = ranked[0], ranked[1]
        assert min(scores[best]) > 0.8 > max(scores[second]), (target, scores)  # unambiguous
        result[target] = {"talker": best, "correlations": scores[best],
                          "next_talker": second, "next_correlations": scores[second]}
    return result


def main():
    sentences = evaluation_sentences()
    targets = map_targets(sentences)
    talker_of = {t: v["talker"] for t, v in targets.items()}
    split = {(r["language"], int(r["sentence"])): r["part"]
             for r in csv.DictReader((ROOT / "protocols/emime/sentence_split.csv").open())}
    OUT.mkdir(parents=True, exist_ok=True)
    header = ["Speaker", "AudioName", "Source", "Label"]
    talkers = set(talker_of.values())
    for lang in ["de", "en"]:
        rows = [r for r in csv.reader((ROOT / f"protocols/emime/emime_{lang}_enroll.csv").open())][1:]
        rows = [[r[0], f"emime/{r[1]}", r[2], r[3]] for r in rows if r[0] in talkers]
        with (OUT / f"vcc2020_{lang}_enroll.csv").open("w", newline="") as stream:
            csv.writer(stream).writerows([header, *rows])
    test = [[r[0], f"emime/{r[1]}", r[2], r[3]]
            for r in list(csv.reader((ROOT / "protocols/emime/emime_test.csv").open()))[1:]
            if r[0] in talkers and r[3] == "bonafide"]
    conversions = []
    for path in sorted(SUBMITTED.glob("team*/wavs/TG*1_*.wav")):
        target, source, uid = NAME.search(path.name).groups()
        team = path.parts[-3]
        test.append([talker_of[target], str(path.relative_to(ROOT / "data")), f"en-vcc2020-{team}", "spoof"])
        conversions.append({"file": path.name, "team": team, "target": target, "talker": talker_of[target],
                            "source_speaker": source, "vcc_sentence": uid,
                            "emime_sentence": sentences[uid], "emime_part": split[("en", sentences[uid])]})
    with (OUT / "vcc2020_test.csv").open("w", newline="") as stream:
        csv.writer(stream).writerows([header, *test])
    with (OUT / "conversions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(conversions[0]))
        writer.writeheader()
        writer.writerows(conversions)
    report = {"targets": targets,
              "evaluation_sentences": {uid: {"emime_sentence": n, "part": split[("en", n)]}
                                       for uid, n in sorted(sentences.items())},
              "test_rows": len(test), "conversions": len(conversions),
              "teams": len({c["team"] for c in conversions})}
    (OUT / "mapping.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "evaluation_sentences"}, indent=1))


if __name__ == "__main__":
    main()
