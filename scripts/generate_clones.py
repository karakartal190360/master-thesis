"""Generate the EMIME voice clones named in the test protocol with Qwen3-TTS (local, offline).

Run with .venv-qwen3tts. Each spoof row of protocols/emime/emime_test.csv names its talker,
target language, test sentence and reference language. The clone speaks the test-sentence
text, so genuine and spoof trials say the same words. Its voice prompt is one genuine
microphone-0 recording of the same talker from the "reference" sentences of the reference
language (never an enrollment or test sentence), chosen with a fixed seed, with its
transcript. Every clone has its own seed. Clones and their manifest stay on the server.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import random
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from prepare_emime import read_prompts

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/emime"
MODEL_DIR = ROOT / "models/qwen3-tts-12hz-1.7b-base"
MODEL_REVISION = "fd4b254389122332181a7c3db7f27e918eec64e3"  # configs/qwen3tts-models.json
SEED = 20261115
LANGUAGE = {"de": "German", "en": "English"}
CLONE = re.compile(r"(G[FM]\d)_(DE|EN)_(\d{4})_qwen3tts_ref(de|en)_synthetic\.wav$")
RELEASE = "v1/UEDIN_bilingual_data_2010/downsampled_22kHz/German_talkers"
COLUMNS = ["file", "speaker", "language", "sentence", "text", "ref_language", "ref_file",
           "ref_sentence", "seed", "status", "sha256", "sample_rate", "duration_s",
           "generate_s", "error"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def genuine_path(talker, lang, sentence):
    gender = "Female" if talker[1] == "F" else "Male"
    code = {"de": "Ger", "en": "Eng"}[lang]
    return f"{RELEASE}/{gender}/{talker}/{code}/{talker}_{code.upper()}_{sentence:04d}_0.wav"


def read_jobs(test_path, split_path, speakers, per_condition):
    split = list(csv.DictReader(split_path.open(encoding="utf-8")))
    texts = {(lang, n): text for lang in LANGUAGE for n, text in read_prompts(lang).items()}
    references = {lang: sorted(int(r["sentence"]) for r in split
                               if r["language"] == lang and r["part"] == "reference")
                  for lang in LANGUAGE}
    jobs, taken = [], {}
    for row in csv.DictReader(test_path.open(encoding="utf-8")):
        match = CLONE.search(row["AudioName"])
        if row["Label"] != "spoof" or not match:
            continue
        talker, lang, sentence, ref_lang = match.groups()
        lang, sentence = lang.lower(), int(sentence)
        if speakers and talker not in speakers:
            continue
        key = (talker, lang, ref_lang)
        taken[key] = taken.get(key, 0) + 1
        if per_condition and taken[key] > per_condition:
            continue
        name = row["AudioName"]
        ref_sentence = random.Random(f"{SEED}-ref-{name}").choice(references[ref_lang])
        seed = int.from_bytes(hashlib.sha256(f"{SEED}-{name}".encode()).digest()[:4], "big")
        jobs.append({"file": name, "speaker": talker, "language": lang, "sentence": sentence,
                     "text": texts[(lang, sentence)], "ref_language": ref_lang,
                     "ref_file": genuine_path(talker, ref_lang, ref_sentence),
                     "ref_sentence": ref_sentence, "ref_text": texts[(ref_lang, ref_sentence)],
                     "seed": seed})
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=ROOT / "protocols/emime/emime_test.csv")
    parser.add_argument("--split", type=Path, default=ROOT / "protocols/emime/sentence_split.csv")
    parser.add_argument("--speakers", nargs="*", help="subset of talkers (pilot)")
    parser.add_argument("--per-condition", type=int, help="first N clones per talker, language, reference")
    parser.add_argument("--resume", action="store_true", help="keep existing clone files (hash recorded)")
    args = parser.parse_args()

    jobs = read_jobs(args.test, args.split, args.speakers, args.per_condition)
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    run_dir = ROOT / "results/clones" / run_id
    run_dir.mkdir(parents=True)
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    report = {
        "run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "emime_voice_clones", "command": [sys.executable, *sys.argv],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        "script_sha256": sha256(__file__),
        "protocol_sha256": {str(p): sha256(p) for p in (args.test, args.split)},
        "model_dir": str(MODEL_DIR), "model_revision": MODEL_REVISION,
        "weights_sha256": {p.name: sha256(p) for p in sorted(MODEL_DIR.glob("**/*.safetensors"))},
        "versions": {n: importlib.metadata.version(n)
                     for n in ["qwen-tts", "torch", "transformers", "accelerate", "numpy", "soundfile"]},
        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        "dtype": "bfloat16", "attn_implementation": "sdpa",
        "generation": "checkpoint sampling defaults, max_new_tokens 2048; voice prompt with transcript",
        "seed": SEED, "speakers": args.speakers, "per_condition": args.per_condition,
        "jobs": len(jobs), "status": "running",
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    model = Qwen3TTSModel.from_pretrained(str(MODEL_DIR), device_map="cuda:0",
                                          dtype=torch.bfloat16, attn_implementation="sdpa")
    prompts, counts = {}, {}
    with (run_dir / "clones.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for n, job in enumerate(jobs, 1):
            out = DATA / job["file"]
            row = {**job, "status": "failed"}
            try:
                if out.exists():
                    if not args.resume:
                        raise FileExistsError("clone exists; use --resume to keep it")
                    row.update(status="existing", sha256=sha256(out))
                else:
                    if job["ref_file"] not in prompts:
                        prompts[job["ref_file"]] = model.create_voice_clone_prompt(
                            ref_audio=str(DATA / job["ref_file"]), ref_text=job["ref_text"],
                            x_vector_only_mode=False)
                    torch.manual_seed(job["seed"])
                    start = time.perf_counter()
                    wavs, rate = model.generate_voice_clone(
                        text=job["text"], language=LANGUAGE[job["language"]],
                        voice_clone_prompt=prompts[job["ref_file"]], max_new_tokens=2048)
                    seconds = time.perf_counter() - start
                    out.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(out, wavs[0], rate, subtype="PCM_16")
                    row.update(status="generated", sha256=sha256(out), sample_rate=rate,
                               duration_s=round(len(wavs[0]) / rate, 3), generate_s=round(seconds, 2))
            except Exception as error:  # recorded, never dropped
                row["error"] = f"{type(error).__name__}: {error}"
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            writer.writerow(row)
            stream.flush()
            if n % 50 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} {counts}", flush=True)

    report.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat(),
                  counts=counts, clones_sha256=sha256(run_dir / "clones.csv"),
                  peak_vram_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"run {run_id}: {counts}; record in {run_dir}")


if __name__ == "__main__":
    main()
