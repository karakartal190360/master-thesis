"""Count clones that continue after the sentence (analysis plan, addendum B; descriptive).

Run with .venv-qwen3tts. Whisper large-v3-turbo with word timestamps and the target
language forced; tail = audio after the end of the last recognized word. A clone has
trailing speech if its tail is at least 1.0 s long and at least -40 dBFS (RMS). Fixed on
2026-11-17 from the XTTS pilot (tails of 2.3-10.7 s at -17 to -31 dBFS), before any score
of generators 2 and 3. No clone is excluded; the count is reported per generator.
"""

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/emime"
MIN_TAIL_S = 1.0
MIN_TAIL_DBFS = -40.0
LANGUAGE = {"de": "german", "en": "english"}


def tail(audio, last_end):
    rest = audio[int(last_end * 16000):]
    seconds = len(rest) / 16000
    level = 20 * np.log10(np.sqrt(np.mean(rest ** 2)) + 1e-12) if len(rest) else -120.0
    return round(seconds, 3), round(float(level), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clone_run", type=Path, help="results/clones/<run id>")
    args = parser.parse_args()
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    import librosa
    import soundfile as sf
    import torch
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model=str(ROOT / "models/whisper-large-v3-turbo"),
                   device=0, dtype=torch.float16)
    clones = [r for r in csv.DictReader((args.clone_run / "clones.csv").open(encoding="utf-8"))
              if r["status"] in ("generated", "existing")]
    rows = []
    for n, c in enumerate(clones, 1):
        audio, rate = sf.read(DATA / c["file"])
        audio = librosa.resample(audio.astype("float32"), orig_sr=rate, target_sr=16000)
        out = asr(audio, return_timestamps="word", generate_kwargs={"language": LANGUAGE[c["language"]]})
        ends = [w["timestamp"][1] for w in out["chunks"] if w["timestamp"][1] is not None]
        last = max(ends) if ends else 0.0
        seconds, level = tail(audio, last)
        rows.append({"file": c["file"], "speaker": c["speaker"], "language": c["language"],
                     "ref_language": c["ref_language"], "duration_s": round(len(audio) / 16000, 3),
                     "last_word_end_s": last, "tail_s": seconds, "tail_dbfs": level,
                     "trailing_speech": seconds >= MIN_TAIL_S and level >= MIN_TAIL_DBFS})
        if n % 200 == 0:
            print(f"{n}/{len(clones)}", flush=True)
    out_dir = ROOT / "results/clone-quality" / args.clone_run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "trailing.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    flagged = [r for r in rows if r["trailing_speech"]]
    summary = {"clone_run": args.clone_run.name, "rule": {"min_tail_s": MIN_TAIL_S, "min_tail_dbfs": MIN_TAIL_DBFS},
               "clones": len(rows), "trailing_speech": len(flagged),
               "by_condition": dict(Counter(f"{r['language']}-ref{r['ref_language']}" for r in flagged)),
               "by_speaker": dict(Counter(r["speaker"] for r in flagged))}
    (out_dir / "trailing.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
