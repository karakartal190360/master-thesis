"""Verify a downloaded Hugging Face dataset snapshot against the Hub's file sizes and SHA-256 hashes."""

import argparse
import hashlib
import importlib.metadata
import json
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
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True, help="full commit hash")
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    from huggingface_hub import HfApi
    info = HfApi().dataset_info(args.repository, revision=args.revision, files_metadata=True)

    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "dataset_download_verification",
        "scientific_result": False,
        "command": [sys.executable, *sys.argv],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "script_sha256": sha256(Path(__file__)),
        "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        "repository": args.repository,
        "revision": info.sha,
        "local_dir": str(args.local_dir.resolve()),
        "files": [],
    }
    for entry in info.siblings:
        path = args.local_dir / entry.rfilename
        # Hub publishes SHA-256 only for LFS files; small Git files are checked by size.
        expected = entry.lfs["sha256"] if entry.lfs else None
        record = {"path": entry.rfilename, "expected_size": entry.size,
                  "expected_sha256": expected}
        if not path.is_file():
            record["status"] = "missing"
        else:
            record["size"] = path.stat().st_size
            if expected:
                print(f"Hashing {entry.rfilename}", flush=True)
                record["sha256"] = sha256(path)
            size_ok = record["size"] == entry.size
            hash_ok = expected is None or record["sha256"] == expected
            record["status"] = "ok" if size_ok and hash_ok else "mismatch"
        report["files"].append(record)

    failed = [f["path"] for f in report["files"] if f["status"] != "ok"]
    report["status"] = "passed" if not failed else "failed"
    report["failed_files"] = failed
    report["total_bytes"] = sum(f.get("size", 0) for f in report["files"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['status']}: {len(report['files']) - len(failed)}/{len(report['files'])} files ok, "
          f"report {args.report}")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
