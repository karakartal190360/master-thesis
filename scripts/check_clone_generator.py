"""Setup check for the voice-cloning generators of generate_clones.py (scientific_result=false).

Clones Qwen's published example voice (saved by check_qwen3tts.py), never a corpus
speaker, into one English and one German sentence, twice with the same seed. Checks offline
loading through the same loader as the clone runs, seeded repeatability, speed, peak VRAM.
"""

import argparse
import importlib.metadata
import json
import os
import time
from datetime import datetime, timezone

from check_qwen3tts import REF_TEXT, SEED, TARGETS, sha256
from generate_clones import ROOT, SYSTEMS, load_voxcpm2, load_xtts

REFERENCE = ROOT / "results/checks/qwen3tts-smoke/reference_qwen_example.wav"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", choices=["voxcpm2", "xtts"])
    args = parser.parse_args()
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    import soundfile as sf
    import torch

    out = ROOT / f"results/checks/{args.system}-smoke"
    out.mkdir(parents=True, exist_ok=True)
    system = SYSTEMS[args.system]
    start = time.perf_counter()
    synthesize = {"voxcpm2": load_voxcpm2, "xtts": load_xtts}[args.system](ROOT / system["model_dir"])
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "purpose": "clone_generator_setup_check",
              "scientific_result": False, "system": args.system, "revision": system["revision"],
              "generation": system["generation"], "reference": str(REFERENCE),
              "reference_sha256": sha256(REFERENCE), "seed": SEED,
              "versions": {n: importlib.metadata.version(n) for n in system["packages"]},
              "load_s": round(time.perf_counter() - start, 1), "outputs": []}
    for lang, _, text in TARGETS:
        hashes = []
        for repeat in range(2):
            torch.manual_seed(SEED)
            begin = time.perf_counter()
            wav, rate = synthesize({"text": text, "language": lang, "ref_file": str(REFERENCE),
                                    "ref_text": REF_TEXT})
            seconds = time.perf_counter() - begin
            wav = wav.cpu().numpy() if hasattr(wav, "cpu") else wav
            path = out / f"spoof_{args.system}_example_{lang}_r{repeat}.wav"
            sf.write(path, wav, rate, subtype="PCM_16")
            hashes.append(sha256(path))
            report["outputs"].append({"file": path.name, "sample_rate": rate,
                                      "duration_s": round(len(wav) / rate, 3),
                                      "generate_s": round(seconds, 2), "sha256": hashes[-1]})
        report[f"repeat_identical_{lang}"] = hashes[0] == hashes[1]
    report["peak_vram_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "versions"}, indent=1))


if __name__ == "__main__":
    main()
