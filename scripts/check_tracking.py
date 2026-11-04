"""Check W&B logging with environment evidence, without a research experiment."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


def git_output(root, *arguments):
    return subprocess.check_output(["git", *arguments], cwd=root, text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Keep the W&B run local.")
    parser.add_argument("--report", type=Path, help="Use a different environment JSON report.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/tracking.json").read_text())
    report_path = args.report or root / "results/environment/linux.json"
    environment = json.loads(report_path.read_text())
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / "results/tracking" / f"{timestamp}-{uuid.uuid4().hex[:8]}"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)

    # Record the executing checkout, including its untracked-file status
    status = git_output(root, "status", "--porcelain=v1", "--untracked-files=all")
    provenance = {
        "purpose": "tracking_setup",
        "scientific_result": False,
        "git_commit": git_output(root, "rev-parse", "HEAD").strip(),
        "git_branch": git_output(root, "rev-parse", "--abbrev-ref", "HEAD").strip(),
        "git_remote": git_output(root, "remote", "get-url", "origin").strip(),
        "git_dirty": bool(status),
        "git_status": status.splitlines(),
        "upstream_commit": json.loads((root / "original/manifest.json").read_text())["commit"],
        "source_sha256": {},
    }
    source_files = [
        "scripts/check_tracking.py",
        "scripts/check_environment.py",
        "configs/tracking.json",
        "requirements/tracking.in",
        "requirements/tracking-linux-py39.lock",
        "requirements/pvp-linux-py39-candidate.lock",
        "original/manifest.json",
    ]
    for name in source_files:
        path = root / name
        provenance["source_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        destination = artifacts / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

    packages = {
        item.metadata["Name"]: item.version
        for item in importlib.metadata.distributions()
    }
    (artifacts / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (artifacts / "config.resolved.json").write_text(json.dumps(config, indent=2) + "\n")
    (artifacts / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    (artifacts / "installed-packages.json").write_text(json.dumps(packages, indent=2) + "\n")
    (artifacts / "git-diff.patch").write_text(git_output(root, "diff", "--binary", "HEAD", "--"))

    # This setup run may be dirty; official scientific runs must require a clean tree
    record = {"status": "prepared", "purpose": "tracking_setup", "scientific_result": False}
    record_path = run_dir / "run.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    import wandb

    mode = "offline" if args.offline else "online"
    run_config = {
        **config,
        "purpose": "tracking_setup",
        "scientific_result": False,
        "git_commit": provenance["git_commit"],
        "git_branch": provenance["git_branch"],
        "git_dirty": provenance["git_dirty"],
        "upstream_commit": provenance["upstream_commit"],
        "environment": environment,
    }
    settings = wandb.Settings(disable_git=True, save_code=False, console="off", x_disable_stats=True)
    try:
        with wandb.init(
            entity=config["entity"], project=config["project"],
            name=f"tracking-setup-{timestamp}", job_type="tracking_setup",
            tags=["setup", "non-scientific"], config=run_config,
            dir=str(run_dir), mode=mode, settings=settings,
        ) as run:
            run.log({
                "setup/checks_passed": sum(value == "passed" for value in environment["checks"].values()),
                "setup/cuda_available": int(environment["cuda_available"]),
            })
            artifact = wandb.Artifact(
                f"tracking-setup-{run.id}", type="reproducibility",
                metadata={"purpose": "tracking_setup", "git_commit": provenance["git_commit"]},
            )
            artifact.add_dir(str(artifacts))
            logged = run.log_artifact(artifact)
            if not args.offline:
                logged.wait()
            run.summary["setup_status"] = "passed"
            record.update({"run_id": run.id, "mode": mode, "url": run.url if not args.offline else None})
        record["status"] = "completed"
    except Exception as error:
        record.update({"status": "failed", "error_type": type(error).__name__})
        raise
    finally:
        record_path.write_text(json.dumps(record, indent=2) + "\n")

    print(json.dumps({**record, "local_directory": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
