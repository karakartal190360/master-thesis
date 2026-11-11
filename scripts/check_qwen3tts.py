"""Setup check for Qwen3-TTS voice cloning (scientific_result=false).

Clones Qwen's own published example voice, never a corpus speaker. Checks offline loading,
same-language and cross-language generation, seeded repeatability, speed and peak VRAM.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/qwen3-tts-12hz-1.7b-base"
# Example reference and its transcript from the Qwen3-TTS README (voice clone section).
REF_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
REF_TEXT = ("Okay. Yeah. I resent you. I love you. I respect you. "
            "But you know what? You blew it! And thanks to you.")
TARGETS = [
    ("en", "English", "The meeting has been moved to Thursday afternoon at three o'clock."),
    ("de", "German", "Die Besprechung wurde auf Donnerstagnachmittag um drei Uhr verschoben."),
]
SEED = 1234


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(model, text, language, prompt, seed):
    import torch

    torch.manual_seed(seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    wavs, sr = model.generate_voice_clone(text=text, language=language,
                                          voice_clone_prompt=prompt, max_new_tokens=2048)
    torch.cuda.synchronize()
    return wavs[0], sr, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/checks/qwen3tts-smoke")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    ref_path = args.output / "reference_qwen_example.wav"
    if not ref_path.exists():
        subprocess.run(["curl", "-sS", "--fail", "-o", str(ref_path), REF_URL], check=True)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "qwen3tts_setup_check",
        "scientific_result": False,
        "command": [sys.executable, *sys.argv],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        "script_sha256": sha256(__file__),
        "model_dir": str(MODEL_DIR),
        "model_revision": "fd4b254389122332181a7c3db7f27e918eec64e3",
        "versions": {name: importlib.metadata.version(name)
                     for name in ["qwen-tts", "torch", "transformers", "accelerate", "numpy"]},
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "generation": "checkpoint defaults (do_sample, temperature 0.9, top_k 50), max_new_tokens 2048",
        "seed": SEED,
        "reference": {"url": REF_URL, "sha256": sha256(ref_path), "text": REF_TEXT},
        "outputs": [],
    }

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(str(MODEL_DIR), device_map="cuda:0",
                                          dtype=torch.bfloat16, attn_implementation="sdpa")
    report["load_seconds"] = round(time.perf_counter() - start, 2)
    prompt = model.create_voice_clone_prompt(ref_audio=str(ref_path), ref_text=REF_TEXT,
                                             x_vector_only_mode=False)

    for tag, language, text in TARGETS:
        runs = []
        for repeat in range(2):  # same seed twice: repeatability check
            wav, sr, seconds = generate(model, text, language, prompt, SEED)
            path = args.output / f"spoof_qwen3tts_example_{tag}_r{repeat}.wav"
            sf.write(path, wav, sr, subtype="PCM_16")
            runs.append({"file": path.name, "sha256": sha256(path), "sample_rate": sr,
                         "duration_s": round(len(wav) / sr, 3), "generate_s": round(seconds, 2),
                         "real_time_factor": round(seconds / (len(wav) / sr), 3)})
        report["outputs"].append({"language": language, "text": text, "runs": runs,
                                  "identical_repeat": runs[0]["sha256"] == runs[1]["sha256"]})

    report["peak_vram_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    report["peak_vram_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ["load_seconds", "peak_vram_allocated_gb", "peak_vram_reserved_gb", "outputs"]},
                     indent=2))


if __name__ == "__main__":
    main()
