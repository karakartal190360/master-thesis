"""Load the three pinned PVP models through unchanged upstream code and test GPU inference.

Setup check only (scientific_result=false): synthetic noise input, no dataset audio.
Hub access is forced offline, so a missing local file fails instead of downloading.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from link_pretrained import MODELS, ROOT, UPSTREAM, link_pretrained


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True)
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    return {"commit": commit.strip(), "dirty": bool(status.strip())}


def gpu_info():
    # WSL ships nvidia-smi outside PATH; same fallback as setup_pvp_linux.sh.
    command = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    query = "name,driver_version,memory.total"
    output = subprocess.check_output(
        [command, f"--query-gpu={query}", "--format=csv,noheader"], text=True)
    return output.strip()


def compare_weights(model_state, file_state, rename=lambda name: name):
    """Compare loaded module weights with the weight file, tensor by tensor."""
    import torch

    file_state = {rename(name): value for name, value in file_state.items()}
    common = sorted(set(model_state) & set(file_state))
    different = [name for name in common
                 if not torch.equal(model_state[name].cpu(), file_state[name])]
    return {
        "compared": len(common),
        "different": different,
        "only_in_model": sorted(set(model_state) - set(file_state)),
        "only_in_file": sorted(set(file_state) - set(model_state)),
    }


def check_weights(pvp, analyzer):
    import torch

    mms_file = torch.load("pretrained_models/mms_300m.ckpt", map_location="cpu")
    ctc_file = torch.load(MODELS / "wav2vec2-xlsr-53-espeak-cv-ft/pytorch_model.bin",
                          map_location="cpu")
    ecapa_file = torch.load("pretrained_models/spkrec-ecapa-voxceleb/embedding_model.ckpt",
                            map_location="cpu")
    mms_backbone = {name[12:]: value for name, value in mms_file.items()
                    if name.startswith("m_ssl.model.")}
    return {
        "mms_backbone": compare_weights(analyzer.feat_model.model.state_dict(), mms_backbone),
        # Transformers maps old weight-norm names to torch parametrization names on load.
        "ctc": compare_weights(
            analyzer.asr_model.state_dict(), ctc_file,
            lambda name: name.replace("weight_g", "parametrizations.weight.original0")
                             .replace("weight_v", "parametrizations.weight.original1")),
        "ecapa_embedding": compare_weights(
            pvp.asv_model.mods.embedding_model.state_dict(), ecapa_file),
    }


def run_inference(pvp, analyzer, seconds, device):
    """Same steps as analyzer.get_phoneme_features_from_file and _extract_embedding,
    run directly so errors are raised instead of printed and skipped."""
    import numpy as np
    import soundfile as sf
    import torch

    rng = np.random.default_rng(0)
    audio = (0.1 * rng.standard_normal(16000 * seconds)).astype("float32")
    path = Path(f"synthetic_noise_{seconds}s.wav")
    sf.write(path, audio, 16000)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    inputs = analyzer.asr_processor(audio, return_tensors="pt", sampling_rate=16000)
    input_values = inputs.input_values.to(analyzer.device)
    with torch.no_grad():
        logits = analyzer.asr_model(input_values).logits
        ids = torch.argmax(logits, dim=-1)
        transcription = analyzer.asr_processor.decode(ids[0], output_char_offsets=True)
        features = analyzer.feat_model(input_values).squeeze(0)
    embedding = pvp.asv_model.encode_batch(torch.from_numpy(audio).float()).squeeze()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    # Upstream path on the same file; it returns [] on internal errors.
    segments = analyzer.get_phoneme_features_from_file(str(path))
    return {
        "seconds": seconds,
        "input_values_shape": list(input_values.shape),
        "ctc_logits_shape": list(logits.shape),
        "mms_features_shape": list(features.shape),
        # Upstream indexes MMS frames with CTC offsets; frame counts must agree.
        "ctc_and_mms_frames_equal": logits.shape[1] == features.shape[0],
        "decoded_phoneme_tokens": len(transcription.char_offsets),
        "upstream_phoneme_segments": len(segments),
        "ecapa_embedding_shape": list(embedding.shape),
        "inference_seconds": round(elapsed, 3),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path,
                        default=ROOT / "results/environment/model-check")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "results/environment/model-check.json")
    parser.add_argument("--seconds", type=int, nargs="+", default=[4, 15, 30])
    args = parser.parse_args()
    work_dir = args.work_dir.resolve()
    report_path = args.report.resolve()

    # Offline, with an empty private cache: only the linked local files can be used.
    hf_home = work_dir / "hf-home"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(hf_home)

    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "in_progress",
        "purpose": "pretrained_model_loading_and_inference_check",
        "scientific_result": False,
        "command": [sys.executable, *sys.argv],
        "git": git_state(),
        # Dirty checkouts: hashes pin the exact code that ran.
        "script_sha256": {path.name: sha256(path) for path in
                          [Path(__file__).resolve(), ROOT / "scripts/link_pretrained.py"]},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["torch", "fairseq", "transformers", "speechbrain",
                      "huggingface-hub", "safetensors"]},
        "gpu": gpu_info(),
        "work_dir": str(work_dir),
        "input": "seeded Gaussian noise (seed 0), 16 kHz mono; not speech",
    }
    try:
        link_pretrained(work_dir)
        os.chdir(work_dir)
        sys.path.insert(0, str(UPSTREAM))
        import torch

        start = time.perf_counter()
        pvp = importlib.import_module("2_PVP")  # loads ECAPA at import (2_PVP.py:24-28)
        report["upstream_settings"] = {"backbone": pvp.BACKBONE, "device": pvp.DEVICE}
        report["memory_after_ecapa_mib"] = round(torch.cuda.memory_allocated() / 2**20)
        # Same call as 2_PVP.py:112; loads CTC model and MMS backbone.
        analyzer = pvp.UniversalPhonemeAnalyzer(device=pvp.DEVICE, feat_model_name=pvp.BACKBONE)
        report["load_seconds"] = round(time.perf_counter() - start, 1)
        report["memory_after_all_models_mib"] = round(torch.cuda.memory_allocated() / 2**20)
        report["ctc_source"] = analyzer.asr_model.config._name_or_path

        report["weights"] = check_weights(pvp, analyzer)
        report["inference"] = [run_inference(pvp, analyzer, seconds, pvp.DEVICE)
                               for seconds in args.seconds]
        report["hf_home_files"] = sorted(
            str(path.relative_to(hf_home)) for path in hf_home.rglob("*") if path.is_file())
        report["status"] = "completed"
    except Exception as error:
        report.update({"status": "failed", "error_type": type(error).__name__,
                       "error": str(error), "traceback": traceback.format_exc()})
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Status: {report['status']}. Report: {report_path}")
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
