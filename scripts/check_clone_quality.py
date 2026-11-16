"""Quality checks for EMIME voice clones, independent of PVP, plus a listening folder.

Run with .venv-qwen3tts. For each clone of a clone run and for its matched genuine test
file (same talker, language and sentence; the baseline):
- intelligibility: Whisper large-v3-turbo transcript with the target language forced,
  word and character error rate against the prompt text (lower case, no punctuation);
  spoken language detected by Whisper without forcing;
- speaker similarity: cosine of WavLM-base-plus-sv x-vectors to the talker's mean
  enrollment x-vector per language, and the highest similarity to another talker;
  WavLM, not ECAPA, because PVP's speaker branch uses ECAPA;
- naturalness: UTMOS22 strong predicted MOS (1-5; trained on English data only);
- signal: duration, level, clipping, duration ratio to the genuine file.
No threshold is applied here. Listening folder: results/deepfake-checks/<clone run id>/.
"""

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from prepare_emime import read_prompts

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/emime"
MODELS = ROOT / "models"
SPEECHMOS = MODELS / "speechmos-ed25eac"
LANGUAGE = {"de": "german", "en": "english"}
COLUMNS = ["kind", "file", "speaker", "language", "sentence", "ref_language", "duration_s",
           "rms_dbfs", "peak", "duration_ratio", "detected_language", "p_target_language",
           "transcript", "wer", "cer", "sim_de", "sim_en", "sim_impostor_max", "utmos"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(text):
    text = unicodedata.normalize("NFC", text.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def error_rates(reference, hypothesis):
    ref, hyp = normalize(reference), normalize(hypothesis)
    wer = edit_distance(ref.split(), hyp.split()) / max(len(ref.split()), 1)
    cer = edit_distance(ref.replace(" ", ""), hyp.replace(" ", "")) / max(len(ref.replace(" ", "")), 1)
    return round(wer, 4), round(cer, 4)


def load_16k(path):
    import librosa

    audio, _ = librosa.load(path, sr=16000)  # mono, same resampling as upstream PVP
    return audio


class Checks:
    def __init__(self, device):
        import torch
        from transformers import (AutoFeatureExtractor, WavLMForXVector,
                                  WhisperForConditionalGeneration, WhisperProcessor)

        sys.path.insert(0, str(SPEECHMOS))
        from speechmos.utmos22.strong.model import UTMOS22Strong

        self.torch, self.device = torch, device
        whisper = MODELS / "whisper-large-v3-turbo"
        self.asr_processor = WhisperProcessor.from_pretrained(whisper)
        self.asr = WhisperForConditionalGeneration.from_pretrained(
            whisper, torch_dtype=torch.float16).to(device).eval()
        # all Whisper language tokens, e.g. <|de|>; detection = softmax over these only
        tokenizer = self.asr_processor.tokenizer
        self.lang_ids = {t[2:-2]: tokenizer.convert_tokens_to_ids(t) for t in tokenizer.additional_special_tokens
                         if re.fullmatch(r"<\|[a-z]{2,3}\|>", t) and t not in ("<|nocaptions|>",)}
        wavlm = MODELS / "wavlm-base-plus-sv"
        self.sv_extractor = AutoFeatureExtractor.from_pretrained(wavlm)
        self.sv = WavLMForXVector.from_pretrained(wavlm).to(device).eval()
        self.mos = UTMOS22Strong()
        self.mos.load_state_dict(torch.load(SPEECHMOS / "utmos22_strong_step7459_v1.pt",
                                            map_location="cpu", weights_only=True))
        self.mos.to(device).eval()

    def whisper(self, audio, lang):
        torch = self.torch
        features = self.asr_processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        features = features.to(self.device, torch.float16)
        with torch.no_grad():
            # language detection: first decoder step after <|startoftranscript|>
            start = torch.tensor([[self.asr.generation_config.decoder_start_token_id]], device=self.device)
            logits = self.asr(input_features=features, decoder_input_ids=start).logits[0, -1]
            codes = list(self.lang_ids)
            probs = torch.softmax(logits[[self.lang_ids[c] for c in codes]].float(), -1)
            detected = codes[int(probs.argmax())]
            p_target = probs[codes.index(lang)].item()
            ids = self.asr.generate(features, language=LANGUAGE[lang], task="transcribe")
        text = self.asr_processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return detected, round(p_target, 4), text

    def xvector(self, audio):
        torch = self.torch
        inputs = self.sv_extractor(audio, sampling_rate=16000, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vector = self.sv(**inputs).embeddings[0]
        return torch.nn.functional.normalize(vector, dim=-1).cpu().numpy()

    def utmos(self, audio):
        torch = self.torch
        with torch.no_grad():
            wave = torch.from_numpy(audio).unsqueeze(0).to(self.device)
            return round(self.mos(wave, 16000).item(), 3)


def centroids(checks, speakers):
    """Mean enrollment x-vector per talker and language (unit length)."""
    result = {}
    for lang in ["de", "en"]:
        rows = list(csv.DictReader((ROOT / f"protocols/emime/emime_{lang}_enroll.csv").open()))
        for speaker in speakers:
            vectors = [checks.xvector(load_16k(DATA / r["AudioName"])) for r in rows
                       if r["Speaker"] == speaker]
            mean = np.mean(vectors, axis=0)
            result[(speaker, lang)] = mean / np.linalg.norm(mean)
    return result


def measure(checks, path, speaker, lang, text, cents):
    audio = load_16k(path)
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    detected, p_target, transcript = checks.whisper(audio, lang)
    wer, cer = error_rates(text, transcript)
    vector = checks.xvector(audio)
    sims = {key: float(vector @ c) for key, c in cents.items()}
    return {"duration_s": round(len(audio) / 16000, 3),
            "rms_dbfs": round(20 * np.log10(rms), 2) if rms > 0 else float("-inf"),
            "peak": round(float(np.abs(audio).max()), 4),
            "detected_language": detected, "p_target_language": p_target,
            "transcript": transcript, "wer": wer, "cer": cer,
            "sim_de": round(sims[(speaker, "de")], 4), "sim_en": round(sims[(speaker, "en")], 4),
            "sim_impostor_max": round(max(v for (s, _), v in sims.items() if s != speaker), 4),
            "utmos": checks.utmos(audio)}


def listening_folder(clones, run_id):
    out = ROOT / "results/deepfake-checks" / run_id
    out.mkdir(parents=True, exist_ok=True)
    sheet, seen = [], set()
    for c in clones:
        folder = out / c["speaker"]
        folder.mkdir(exist_ok=True)
        stem = f"{c['language'].upper()}_{int(c['sentence']):04d}"
        pairs = [(f"{stem}_clone_ref{c['ref_language']}.wav", DATA / c["file"], "clone"),
                 (f"{stem}_genuine.wav", DATA / genuine(c), "genuine"),
                 (f"reference_{c['ref_language']}_{int(c['ref_sentence']):04d}.wav",
                  DATA / c["ref_file"], "voice prompt (genuine)")]
        for name, source, kind in pairs:
            if not source.exists() or f"{c['speaker']}/{name}" in seen:
                continue
            seen.add(f"{c['speaker']}/{name}")
            shutil.copyfile(source, folder / name)
            sheet.append({"file": f"{c['speaker']}/{name}", "kind": kind, "speaker": c["speaker"],
                          "language": c["language"] if kind != "voice prompt (genuine)" else c["ref_language"],
                          "your_rating (good/bad)": "", "notes": ""})
    with (out / "ratings.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sheet[0]))
        writer.writeheader()
        writer.writerows(sorted(sheet, key=lambda r: r["file"]))
    return out


def genuine(clone):
    from generate_clones import genuine_path

    return genuine_path(clone["speaker"], clone["language"], int(clone["sentence"]))


def median(rows, key):
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return round(float(np.median(values)), 4) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clone_run", type=Path, help="results/clones/<run id>")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-listening", action="store_true",
                        help="skip the listening folder (full runs: avoids copying every clone)")
    args = parser.parse_args()
    clones = [r for r in csv.DictReader((args.clone_run / "clones.csv").open(encoding="utf-8"))
              if r["status"] in ("generated", "existing")]
    texts = {lang: read_prompts(lang) for lang in ["de", "en"]}
    excluded = {r["file"] for r in csv.DictReader((ROOT / "protocols/emime/excluded.csv").open())}

    checks = Checks(args.device)
    speakers = sorted({c["speaker"] for c in clones})
    # all 14 talkers: impostor similarity needs the other talkers too
    cents = centroids(checks, sorted({r["Speaker"] for r in csv.DictReader(
        (ROOT / "protocols/emime/emime_de_enroll.csv").open())}))
    rows, genuine_done = [], {}
    for c in clones:
        lang, sentence = c["language"], int(c["sentence"])
        text = texts[lang][sentence]
        g = genuine(c)
        if g not in genuine_done and g not in excluded:
            genuine_done[g] = {"kind": "genuine", "file": g, "speaker": c["speaker"], "language": lang,
                               "sentence": sentence, "ref_language": "",
                               **measure(checks, DATA / g, c["speaker"], lang, text, cents)}
            rows.append(genuine_done[g])
        row = {"kind": "clone", "file": c["file"], "speaker": c["speaker"], "language": lang,
               "sentence": sentence, "ref_language": c["ref_language"],
               **measure(checks, DATA / c["file"], c["speaker"], lang, text, cents)}
        if g in genuine_done:
            row["duration_ratio"] = round(row["duration_s"] / genuine_done[g]["duration_s"], 3)
        rows.append(row)
        if len(rows) % 50 == 0:
            print(f"{len(rows)} rows", flush=True)

    run_id = args.clone_run.name
    out = ROOT / "results/clone-quality" / run_id
    out.mkdir(parents=True, exist_ok=True)
    with (out / "quality.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for r in rows:
        key = f"{r['language']}-genuine" if r["kind"] == "genuine" else f"{r['language']}-clone-ref{r['ref_language']}"
        groups.setdefault(key, []).append(r)
    summary = {key: {"files": len(g), **{f"median_{k}": median(g, k) for k in
                     ["wer", "cer", "p_target_language", "sim_de", "sim_en", "sim_impostor_max",
                      "utmos", "duration_s", "duration_ratio"]},
                     "wrong_language": sum(r["detected_language"] != r["language"] for r in g)}
               for key, g in sorted(groups.items())}
    folder = None if args.no_listening else listening_folder(clones, run_id)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "purpose": "clone_quality_check",
        "clone_run": run_id, "command": [sys.executable, *sys.argv],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        "script_sha256": sha256(__file__), "models": "configs/quality-models.json",
        "utmos_weights_sha256": sha256(SPEECHMOS / "utmos22_strong_step7459_v1.pt"),
        "speakers": speakers, "summary": summary, "listening_folder": folder and str(folder),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"quality: {out}\nlistening folder: {folder}")


if __name__ == "__main__":
    main()
