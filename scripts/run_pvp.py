"""Run PVP on one benchmark with unchanged upstream code and save every trial.

Repeats the loop of upstream run_poi_specificity_experiment (2_PVP.py:103-153) with the
same upstream objects and settings, but (1) keeps a score or failure reason for every
test trial, (2) records each trial's scoring tier, including upstream's silent 0.5
fallback, and (3) saves profiles, provenance and metrics. Upstream source is not changed.

Only --scientific runs count as results: they need a clean, committed checkout and the
config's full protocols. All other runs are setup checks (scientific_result=false).
"""

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from evaluate_pvp import compare_with_paper, evaluate, read_scores
from link_pretrained import MODELS, ROOT, UPSTREAM, link_data, link_pretrained

WORK_DIR = ROOT / "results/runtime"  # upstream working directory with runtime links
SCORE_COLUMNS = ["trial", "speaker", "file", "source", "label", "status", "score",
                 "phoneme_score", "tier", "error"]
WEIGHT_FILES = ["mms-300m-anti-deepfake/mms_300m.ckpt",
                "wav2vec2-xlsr-53-espeak-cv-ft/pytorch_model.bin",
                "spkrec-ecapa-voxceleb/embedding_model.ckpt"]
PACKAGES = ["torch", "fairseq", "transformers", "speechbrain", "librosa", "soundfile",
            "numpy", "scikit-learn", "pandas", "wandb"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def git_state():
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status), "status": status.splitlines()}


def environment():
    import torch

    command = "/usr/lib/wsl/lib/nvidia-smi"  # WSL ships nvidia-smi outside PATH
    gpu = subprocess.check_output(
        [command, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        text=True).strip()
    return {"python": platform.python_version(), "platform": platform.platform(), "gpu": gpu,
            "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "packages": {p: importlib.metadata.version(p) for p in PACKAGES}}


def read_protocol(path):
    import pandas as pd

    # Same call as 2_PVP.py:110-111: header skipped, columns assigned by position.
    return pd.read_csv(path, sep=",", names=["speaker", "path", "source", "label"], header=0)


def enroll_paths(group, audio_root):
    # 2_PVP.py:118-122: the first path decides whether ".wav" is appended to all.
    paths = group["path"].map(str).tolist()
    if not paths[0].endswith(".wav"):
        return [os.path.join(audio_root, p + ".wav") for p in paths]
    return [os.path.join(audio_root, p) for p in paths]


def test_path(path, audio_root):
    # 2_PVP.py:139-141
    path = str(path)
    return os.path.join(audio_root, path if path.endswith(".wav") else path + ".wav")


def watch_empty_features(analyzer):
    """Observe files without phoneme features; upstream prints and continues (analyzer.py:90-92)."""
    empty = []
    original = analyzer.get_phoneme_features_from_file

    def observed(path):
        features = original(path)
        if not features:
            empty.append(path)
        return features

    analyzer.get_phoneme_features_from_file = observed
    return empty


def save_profile(detector, path):
    import joblib

    joblib.dump({"phoneme_gmms": detector.phoneme_gmms,
                 "phoneme_weights": detector.phoneme_weights,
                 "top_k_set": sorted(detector.top_k_set),
                 "broad_class_gmms": detector.broad_class_gmms,
                 "broad_class_weights": detector.broad_class_weights,
                 "speaker_gmm": detector.target_gmm}, path)


def enroll(pvp, analyzer, enroll_df, config, profile_dir, empty):
    audio_root = str(ROOT / config["audio_root"])
    profiles, records = {}, {}
    for speaker, group in enroll_df.groupby("speaker"):
        paths = enroll_paths(group, audio_root)
        record = {"files": len(paths)}
        if len(paths) < config["min_enroll_files"]:
            records[speaker] = {**record, "enrolled": False, "reason": "too_few_enroll_files"}
            continue
        start, empty_before = time.time(), len(empty)
        detector = pvp.HybridPOIDetector(analyzer)
        detector.fit(paths, top_k=config["top_k"])
        save_profile(detector, profile_dir / f"{speaker}.joblib")
        profiles[speaker] = detector
        records[speaker] = {
            **record, "enrolled": True, "seconds": round(time.time() - start, 1),
            "files_without_phoneme_features": [Path(p).name for p in empty[empty_before:]],
            "phoneme_gmms": len(detector.phoneme_gmms), "top_k": sorted(detector.top_k_set),
            "broad_class_gmms": sorted(detector.broad_class_gmms),
            "speaker_gmm_components": detector.target_gmm.n_components}
        print(f"enrolled {speaker}: {len(paths)} files, {records[speaker]['seconds']} s", flush=True)
    return profiles, records


def score_trial(profiles, row, audio_root, alpha):
    out = {"status": "failed", "score": "", "phoneme_score": "", "tier": "", "error": ""}
    if row.speaker not in profiles:  # upstream raises KeyError here (2_PVP.py:136)
        return {**out, "error": "speaker_not_enrolled"}
    path = test_path(row.path, audio_root)
    try:
        fused, phoneme, tier = profiles[row.speaker].predict(path, alpha=alpha)
    except Exception as error:  # upstream prints and skips (2_PVP.py:143-147); we keep it
        reason = "missing_file" if not os.path.isfile(path) else type(error).__name__
        return {**out, "error": f"{reason}: {error}"}
    return {**out, "status": "scored", "score": repr(float(fused)),
            "phoneme_score": repr(float(phoneme)), "tier": tier}


def score(profiles, test_df, config, scores_path, log):
    audio_root = str(ROOT / config["audio_root"])
    start = time.time()
    with scores_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SCORE_COLUMNS)
        writer.writeheader()
        for n, row in enumerate(test_df.itertuples(index=False)):
            result = score_trial(profiles, row, audio_root, config["alpha"])
            source = "" if row.source != row.source else row.source  # NaN for bona fide ZH rows
            writer.writerow({"trial": n, "speaker": row.speaker,
                             "file": os.path.basename(test_path(row.path, audio_root)),
                             "source": source, "label": row.label, **result})
            if (n + 1) % 1000 == 0 or n + 1 == len(test_df):
                stream.flush()
                rate = (n + 1) / (time.time() - start)
                print(f"scored {n + 1}/{len(test_df)} ({rate:.1f} trials/s, "
                      f"{(len(test_df) - n - 1) / rate / 3600:.2f} h left)", flush=True)
                log({"progress/trials": n + 1, "progress/trials_per_second": rate})


def start_wandb(mode, run_id, run_dir, run_config, tags):
    if mode == "disabled":
        return None
    import wandb

    tracking = json.loads((ROOT / "configs/tracking.json").read_text())
    settings = wandb.Settings(disable_git=True, save_code=False, console="off", x_disable_stats=True)
    return wandb.init(entity=tracking["entity"], project=tracking["project"], id=run_id,
                      name=run_id, job_type="pvp_run", tags=tags, config=run_config,
                      dir=str(run_dir), mode=mode, settings=settings)


def upload(run, run_dir, record):
    import wandb

    summary = record["metrics"]["pooled"]
    run.summary.update({"eer": summary["eer"], "auc": summary["auc"],
                        "trials": record["metrics"]["trials"], "failed": record["metrics"]["failed"]})
    artifact = wandb.Artifact(f"pvp-run-{run.id}", type="reproducibility",
                              metadata={"scientific_result": record["scientific_result"],
                                        "git_commit": record["git"]["commit"]})
    for name in ["run.json", "config.resolved.json", "enrollment.json", "scores.csv", "metrics.json"]:
        artifact.add_file(str(run_dir / name))
    artifact.add_dir(str(run_dir / "profiles"), name="profiles")
    logged = run.log_artifact(artifact)
    if run.settings.mode == "online":
        logged.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--test", type=Path, help="subset of the test protocol (setup checks only)")
    parser.add_argument("--scientific", action="store_true",
                        help="official run: refuse a dirty checkout or a protocol override")
    parser.add_argument("--wandb", choices=["online", "offline", "disabled"], default="online")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    test_protocol = (args.test or ROOT / config["test"]).resolve()
    state = git_state()
    if args.scientific and (state["dirty"] or args.test):
        sys.exit("--scientific needs a clean, committed checkout and the config's full protocols")

    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    run_dir = ROOT / "results/runs" / run_id
    (run_dir / "profiles").mkdir(parents=True)
    record_path = run_dir / "run.json"
    record = {
        "run_id": run_id, "status": "prepared", "started_at": datetime.now(timezone.utc).isoformat(),
        "scientific_result": args.scientific, "command": [sys.executable, *sys.argv],
        "git": state, "upstream_commit": json.loads((ROOT / "original/manifest.json").read_text())["commit"],
        "config": str(args.config), "config_sha256": sha256(args.config),
        "protocols": {str(p): sha256(p) for p in (ROOT / config["enroll"], test_protocol)},
        "audio_root": str((ROOT / config["audio_root"]).resolve()),
        "weights_sha256": {name: sha256(MODELS / name) for name in WEIGHT_FILES},
        "environment": environment(),
        "seeds": "no random sampling at inference; GMMs use random_state=42 (analyzer.py:137, 2_PVP.py:68-74)",
    }
    (run_dir / "config.resolved.json").write_text(json.dumps(config, indent=2) + "\n")
    record_path.write_text(json.dumps(record, indent=2) + "\n")

    tags = ["pvp", config["benchmark"], config["backbone"],
            "scientific" if args.scientific else "non-scientific"]
    run = start_wandb(args.wandb, run_id, run_dir, {**config, "run_id": run_id,
                      "scientific_result": args.scientific, "git_commit": state["commit"],
                      "git_dirty": state["dirty"], "test_protocol": str(test_protocol)}, tags)
    log = run.log if run else (lambda values: None)
    try:
        # Offline with a private cache: only the linked local model files can be used.
        os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                           "HF_HOME": str(WORK_DIR / "hf-home")})
        link_pretrained(WORK_DIR)
        link_data(WORK_DIR)
        os.chdir(WORK_DIR)
        sys.path.insert(0, str(UPSTREAM))
        import torch

        pvp = importlib.import_module("2_PVP")  # loads ECAPA on pvp.DEVICE at import
        if config["device"] != pvp.DEVICE:
            raise ValueError(f"upstream fixes the ECAPA device to {pvp.DEVICE}")
        analyzer = pvp.UniversalPhonemeAnalyzer(device=pvp.DEVICE, feat_model_name=config["backbone"])
        empty = watch_empty_features(analyzer)
        record["status"] = "running"
        record_path.write_text(json.dumps(record, indent=2) + "\n")

        profiles, enrollment = enroll(pvp, analyzer, read_protocol(ROOT / config["enroll"]),
                                      config, run_dir / "profiles", empty)
        (run_dir / "enrollment.json").write_text(json.dumps(enrollment, indent=2) + "\n")
        enroll_empty = len(empty)
        test_df = read_protocol(test_protocol)
        score(profiles, test_df, config, run_dir / "scores.csv", log)

        metrics = evaluate(read_scores(run_dir / "scores.csv"))
        metrics["paper_comparison"] = compare_with_paper(metrics, config["paper"])
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        record.update({"status": "completed", "finished_at": datetime.now(timezone.utc).isoformat(),
                       "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                       "test_files_without_phoneme_features": len(empty) - enroll_empty,
                       "metrics": {k: metrics[k] for k in ("trials", "scored", "failed", "pooled",
                                                           "failed_by_reason", "tiers")},
                       "paper_comparison": metrics["paper_comparison"]})
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        if run:
            upload(run, run_dir, record)
            record["wandb_url"] = run.url if args.wandb == "online" else None
    except BaseException as error:
        record.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        if run:
            run.finish(exit_code=0 if record["status"] == "completed" else 1)
    print(json.dumps({k: record[k] for k in ("run_id", "status", "metrics", "paper_comparison")}, indent=2))
    print(f"local record: {run_dir}")


if __name__ == "__main__":
    main()
