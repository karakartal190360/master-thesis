"""Download the pinned model files and verify their sizes and SHA-256 hashes."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "models")
    parser.add_argument("--report", type=Path,
                        default=root / "results/environment/pretrained-download.json")
    args = parser.parse_args()
    plan_path = root / "configs/pretrained-models.json"
    plan = json.loads(plan_path.read_text())

    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ.setdefault("HF_HOME", str(root / ".cache/huggingface"))
    from huggingface_hub import hf_hub_download

    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "in_progress",
        "purpose": "pretrained_download",
        "scientific_result": False,
        "command": [sys.executable, *sys.argv],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "script_sha256": sha256(Path(__file__)),
        "plan_sha256": sha256(plan_path),
        "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        "output_directory": str(args.output.resolve()),
        "files": [],
        "models_loaded": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    try:
        for model in plan["models"]:
            for entry in model["files"]:
                report["current_file"] = f"{model['repository']}/{entry['path']}"
                args.report.write_text(json.dumps(report, indent=2) + "\n")
                print(f"Downloading {report['current_file']}", flush=True)
                path = Path(hf_hub_download(
                    repo_id=model["repository"], revision=model["revision"],
                    filename=entry["path"], local_dir=args.output / model["directory"],
                ))
                size = path.stat().st_size
                digest = sha256(path)
                if size != entry["size_bytes"]:
                    raise ValueError("Downloaded file size differs from the pinned plan")
                if entry["sha256"] and digest != entry["sha256"]:
                    raise ValueError("Downloaded weight hash differs from the pinned plan")
                report["files"].append({
                    "repository": model["repository"], "revision": model["revision"],
                    "filename": entry["path"], "local_path": str(path.resolve()),
                    "size_bytes": size, "sha256": digest,
                    "expected_sha256": entry["sha256"], "status": "verified",
                })
                args.report.write_text(json.dumps(report, indent=2) + "\n")
                print(f"Verified {entry['path']} ({size} bytes)", flush=True)
        report.pop("current_file")
        report["status"] = "completed"
        report["total_bytes"] = sum(item["size_bytes"] for item in report["files"])
    except Exception as error:
        report.update({"status": "failed", "error_type": type(error).__name__})
        print(f"Download failed: {type(error).__name__}. See {args.report}", flush=True)
        raise SystemExit(1) from None
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {len(report['files'])} files; {report['total_bytes']} bytes in total.")


if __name__ == "__main__":
    main()
