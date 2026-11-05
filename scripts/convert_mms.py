"""Convert the MMS safetensors file to the PyTorch checkpoint that upstream PVP loads.

Lossless container change only: same tensor names, shapes, dtypes and values.
Upstream `backbones/mms.py` reads `pretrained_models/mms_300m.ckpt` with `torch.load`.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file

EXPECTED_SHA256 = "9bd5d9785bf72f91dfca8508651354946c4b307921cb358505e1b94035102224"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    model_dir = root / "models/mms-300m-anti-deepfake"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=model_dir / "model.safetensors")
    parser.add_argument("--target", type=Path, default=model_dir / "mms_300m.ckpt")
    parser.add_argument("--report", type=Path,
                        default=root / "results/environment/mms-conversion.json")
    args = parser.parse_args()

    source_hash = sha256(args.source)
    if source_hash != EXPECTED_SHA256:
        raise SystemExit("Source hash differs from the pinned download; not converting.")

    original = load_file(str(args.source))
    torch.save(original, args.target)

    # Reload exactly as upstream does and compare every tensor bit for bit.
    converted = torch.load(args.target, map_location="cpu")
    if converted.keys() != original.keys():
        raise SystemExit("Tensor names differ after conversion.")
    for name, tensor in original.items():
        other = converted[name]
        if other.dtype != tensor.dtype or other.shape != tensor.shape:
            raise SystemExit(f"Dtype or shape differs for {name}.")
        if not torch.equal(other, tensor):
            raise SystemExit(f"Values differ for {name}.")

    report = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "purpose": "mms_format_conversion",
        "scientific_result": False,
        "command": [sys.executable, *sys.argv],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "script_sha256": sha256(Path(__file__)),
        "torch": torch.__version__,
        "source": {"path": str(args.source.resolve()), "sha256": source_hash,
                   "size_bytes": args.source.stat().st_size},
        "target": {"path": str(args.target.resolve()), "sha256": sha256(args.target),
                   "size_bytes": args.target.stat().st_size},
        "tensors": len(original),
        "backbone_tensors": sum(name.startswith("m_ssl.model.") for name in original),
        "head_tensors": sum(name.startswith("proj_fc.") for name in original),
        "all_tensors_equal": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Converted {len(original)} tensors; all equal. Report: {args.report}")


if __name__ == "__main__":
    main()
