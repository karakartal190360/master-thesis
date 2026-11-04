"""Check software and synthetic inputs without downloading pretrained weights."""

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import runpy
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def check_environment(pvp=False, cuda=False):
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("HF_HOME", str(root / ".cache/huggingface"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import librosa
    import numpy as np
    import pandas as pd
    import soundfile as sf
    import torch
    from sklearn.metrics import roc_auc_score
    from sklearn.mixture import GaussianMixture
    from transformers import (
        Wav2Vec2ForCTC, Wav2Vec2PhonemeCTCTokenizer, Wav2Vec2Processor,
    )

    packages = [
        "numpy", "scipy", "pandas", "scikit-learn", "librosa", "numba",
        "soundfile", "torch", "transformers", "huggingface-hub",
    ]
    if pvp:
        for module in ["fairseq", "torchaudio", "speechbrain.inference.speaker"]:
            importlib.import_module(module)
        packages.extend(["fairseq", "torchaudio", "speechbrain"])

    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "system": platform.system(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "checks": {},
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_build": torch.version.cuda,
        "thread_environment": {
            name: os.environ.get(name)
            for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]
        },
        "pvp_imports_checked": pvp,
        "pretrained_models_checked": False,
        "ctc_classes": [Wav2Vec2ForCTC.__name__, Wav2Vec2Processor.__name__],
    }
    if platform.system() == "Linux":
        report["os_release"] = Path("/etc/os-release").read_text()

    # Numeric stack and statistical model
    rng = np.random.default_rng(42)
    enrollment = rng.normal(size=(40, 8))
    gmm = GaussianMixture(
        n_components=5, covariance_type="diag", reg_covar=1e-3, random_state=42,
    ).fit(enrollment)
    likelihoods = gmm.score_samples(enrollment)
    assert gmm.converged_ and np.isfinite(likelihoods).all()
    assert pd.DataFrame({"score": likelihoods})["score"].notna().all()
    report["checks"]["numeric_stack_gmm"] = "passed"

    # Check the pinned metric function without importing experiment entry points
    compute_eer = runpy.run_path(str(root / "original/PVP/utils.py"))["compute_eer"]
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_eer(scores, labels)[0] == 0
    assert compute_eer(-scores, labels)[0] == 1
    assert roc_auc_score(labels, scores) == 1
    report["checks"]["metric_direction"] = "passed"

    with tempfile.TemporaryDirectory(prefix="pvp-environment-") as temp:
        temp = Path(temp)

        # Audio decoding and the 16 kHz resampling path
        signal = 0.1 * np.sin(2 * np.pi * 440 * np.arange(1600) / 8000)
        sf.write(temp / "tone.wav", signal, 8000, subtype="FLOAT")
        audio, sample_rate = librosa.load(temp / "tone.wav", sr=16000)
        assert sample_rate == 16000 and audio.shape == (3200,)
        assert np.isfinite(audio).all()
        report["checks"]["audio_decode_resample"] = "passed"

        # Minimal vocabulary: no hub files or text phonemizer needed
        vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "a": 4, "b": 5}
        (temp / "vocab.json").write_text(json.dumps(vocab))
        tokenizer = Wav2Vec2PhonemeCTCTokenizer(
            str(temp / "vocab.json"), do_phonemize=False,
        )
        decoded = tokenizer.decode([4, 4, 0, 5, 5, 0], output_char_offsets=True)
        offsets = [
            (item["char"], int(item["start_offset"]), int(item["end_offset"]))
            for item in decoded.char_offsets
        ]
        assert offsets == [("a", 0, 2), ("b", 3, 5)]
        report["checks"]["ctc_offsets"] = "passed"
        report["ctc_example_offsets"] = offsets
        report["ctc_end_offset"] = "exclusive"

    device = "cuda" if cuda else "cpu"
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this environment.")
    matrix = torch.eye(3, device=device)
    assert torch.equal(matrix @ matrix, matrix)
    vector = enrollment[0].astype(np.float32)
    assert np.array_equal(torch.from_numpy(vector).to(device).cpu().numpy(), vector)
    report["checks"][f"torch_{device}_operation"] = "passed"
    if cuda:
        torch.cuda.synchronize()
        report["gpu"] = torch.cuda.get_device_name(0)

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pvp", action="store_true", help="Also import the Linux PVP stack.")
    parser.add_argument("--cuda", action="store_true", help="Require and test an NVIDIA GPU.")
    parser.add_argument("--output", type=Path, help="Save the verification report as JSON.")
    args = parser.parse_args()
    report = check_environment(pvp=args.pvp, cuda=args.cuda)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
