"""EMIME cross-language analysis as fixed in protocols/emime/analysis-plan.md (commit 7721b7d).

Inputs: the scores of the official runs (German and English enrollment, and the
wrong-speaker controls). Unit of inference: the talker (n = 14).
- Primary (Section 3/4): P1-P4, pooled values, talker bootstrap (10,000 draws, seed
  20261115, percentile 95% interval), exact sign-flip test on per-talker values (16,384
  patterns, two-sided), Holm over P1-P4, margin reading for P1/P2 (2.0 EER points).
- Control (Section 5): EER(wrong profile) - EER(right profile) per cell.
- Secondary (Section 6): S1-S7.
EER and AUC as in evaluate_pvp.py (upstream compute_eer; accept if score >= threshold).
Values in percent.
"""

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from evaluate_pvp import compute_eer

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261115
DRAWS = 10_000
MARGIN = 2.0  # D1, EER points
ALPHA = 0.8  # upstream fusion weight
LANGS = ["de", "en"]
TALKERS = [f"G{s}{i}" for s in "FM" for i in range(1, 8)]
OTHER = {"de": "en", "en": "de"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_run(run_id, leave_out):
    """Per true talker: arrays of label, test language, prompt language, scores."""
    rows = list(csv.DictReader((ROOT / "results/runs" / run_id / "scores.csv").open()))
    failed = [r["file"] for r in rows if r["status"] != "scored"]
    data = {}
    for t in TALKERS:
        own = [r for r in rows if r["status"] == "scored" and re.match(rf"{t}_", r["file"])]
        data[t] = {
            "label": np.array([r["label"] == "bonafide" for r in own]),
            "lang": np.array([r["source"][:2] for r in own]),
            "prompt": np.array([r["source"][-2:] if r["label"] == "spoof" else "" for r in own]),
            "score": np.array([float(r["score"]) for r in own]),
            "phoneme": np.array([float(r["phoneme_score"]) for r in own]),
            "tier": np.array([r["tier"] for r in own]),
            "file": np.array([r["file"] for r in own]),
            "sensitivity_out": np.array([r["file"] in leave_out for r in own]),
        }
        data[t]["speaker"] = (data[t]["score"] - ALPHA * data[t]["phoneme"]) / (1 - ALPHA)
    return data, failed


def gather(run, talkers, lang, field="score", kind=None, prompt=None, sensitivity=False):
    """Labels and values of one test language for a talker list (repeats count twice)."""
    labels, values = [], []
    for t in talkers:
        d = run[t]
        mask = d["lang"] == lang
        if kind == "genuine":
            mask &= d["label"]
        if kind == "clone":
            mask &= ~d["label"]
        if prompt:
            mask &= d["label"] | (d["prompt"] == prompt)
        if sensitivity:
            mask &= ~d["sensitivity_out"]
        labels.append(d["label"][mask])
        values.append(d[field][mask])
    return np.concatenate(labels), np.concatenate(values)


def eer(labels, scores):
    value, threshold = compute_eer(scores, labels.astype(int))
    return 100 * float(value), float(threshold)


def auc(labels, scores):
    return 100 * float(roc_auc_score(labels.astype(int), scores))


def rates(run, talkers, lang, threshold, sensitivity=False):
    """False rejection of genuine and false acceptance of clones, percent."""
    labels, scores = gather(run, talkers, lang, sensitivity=sensitivity)
    return (100 * float(np.mean(scores[labels] < threshold)),
            100 * float(np.mean(scores[~labels] >= threshold)))


# Contrasts: functions of a talker list; per-talker versions take the pooled threshold.

def p1(runs, talkers, sens=False):  # English trials: German profile minus English profile
    return (eer(*gather(runs["de"], talkers, "en", sensitivity=sens))[0]
            - eer(*gather(runs["en"], talkers, "en", sensitivity=sens))[0])


def p2(runs, talkers, sens=False):  # German trials: English profile minus German profile
    return (eer(*gather(runs["en"], talkers, "de", sensitivity=sens))[0]
            - eer(*gather(runs["de"], talkers, "de", sensitivity=sens))[0])


def shifts(run, talkers, same, cross, threshold=None, sens=False):
    """P3/P4 (and S2): threshold at the EER point of the same-language cell."""
    if threshold is None:
        threshold = eer(*gather(run, talkers, same, sensitivity=sens))[1]
    frr_same, far_same = rates(run, talkers, same, threshold, sens)
    frr_cross, far_cross = rates(run, talkers, cross, threshold, sens)
    return {"genuine_shift": frr_cross - frr_same, "clone_shift": far_cross - far_same,
            "threshold": threshold}


def interval(values):
    low, high = np.percentile(values, [2.5, 97.5])
    return [round(float(low), 3), round(float(high), 3)]


def sign_flip(values):
    """Exact two-sided sign-flip test on the mean of per-talker values."""
    values = np.asarray(values)
    signs = np.array(list(product([1, -1], repeat=len(values))))
    means = signs @ values / len(values)
    return float(np.mean(np.abs(means) >= abs(values.mean()) - 1e-12))


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def margin_reading(ci):
    low, high = ci
    if -MARGIN <= low and high <= MARGIN:
        return "no meaningful change; PVP transfers in this setting"
    if low > MARGIN:
        return "meaningful loss"
    if low > 0 or high < 0:
        return "loss, size uncertain" if low > 0 else "gain, size uncertain"
    return "inconclusive"


def summary(pooled, boot, per_talker=None):
    out = {"pooled": round(pooled, 3), "ci95": interval(boot)}
    if per_talker is not None:
        values = [per_talker[t] for t in TALKERS]
        out.update({"per_talker": {t: round(per_talker[t], 3) for t in TALKERS},
                    "per_talker_mean": round(float(np.mean(values)), 3),
                    "talkers_positive": int(sum(v > 0 for v in values)),
                    "p_sign_flip": sign_flip(values)})
    return out


def bootstrap(fn, draws):
    return [fn(list(d)) for d in draws]


def primary(runs, draws, sens=False):
    out = {}
    for name, fn in [("P1", p1), ("P2", p2)]:
        out[name] = summary(fn(runs, TALKERS, sens), bootstrap(lambda ts: fn(runs, ts, sens), draws),
                            {t: fn(runs, [t], sens) for t in TALKERS})
        out[name]["reading"] = margin_reading(out[name]["ci95"])
    pooled = shifts(runs["de"], TALKERS, "de", "en", sens=sens)
    boot = bootstrap(lambda ts: shifts(runs["de"], ts, "de", "en", sens=sens), draws)
    per = {t: shifts(runs["de"], [t], "de", "en", pooled["threshold"], sens) for t in TALKERS}
    for name, key in [("P3", "genuine_shift"), ("P4", "clone_shift")]:
        out[name] = summary(pooled[key], [b[key] for b in boot], {t: per[t][key] for t in TALKERS})
        out[name]["threshold"] = pooled["threshold"]
    difference = [b["genuine_shift"] - b["clone_shift"] for b in boot]
    out["P3_minus_P4"] = {"pooled": round(pooled["genuine_shift"] - pooled["clone_shift"], 3),
                          "ci95": interval(difference)}
    p3, d = out["P3"]["ci95"], out["P3_minus_P4"]["ci95"]
    p4 = out["P4"]["ci95"]
    if (p3[0] > 0 or p3[1] < 0) and (d[0] > 0 or d[1] < 0) and d[0] > 0:
        out["RQ2_reading"] = "the genuine shift dominates"
    elif (p4[0] > 0 or p4[1] < 0) and d[1] < 0:
        out["RQ2_reading"] = "the clone shift dominates"
    else:
        out["RQ2_reading"] = "both shifts reported, no dominance claimed"
    adjusted = holm([out[n]["p_sign_flip"] for n in ["P1", "P2", "P3", "P4"]])
    for name, p in zip(["P1", "P2", "P3", "P4"], adjusted):
        out[name]["p_holm"] = round(p, 5)
    return out


def control(runs, draws):
    out = {}
    for e, t in product(LANGS, LANGS):
        right, wrong = runs[e], runs[f"{e}_wrong"]
        fn = lambda ts: eer(*gather(wrong, ts, t))[0] - eer(*gather(right, ts, t))[0]
        result = summary(fn(TALKERS), bootstrap(fn, draws), {x: fn([x]) for x in TALKERS})
        result["wrong_eer"] = round(eer(*gather(wrong, TALKERS, t))[0], 3)
        result["right_eer"] = round(eer(*gather(right, TALKERS, t))[0], 3)
        result["speaker_specific_shown"] = result["ci95"][0] > MARGIN
        out[f"enroll_{e}_test_{t}"] = result
    return out


def cells(runs, draws):
    """S1: all cells, pooled and per talker, tiers, neutral phoneme scores."""
    out = {}
    for e, t in product(LANGS, LANGS):
        run = runs[e]
        labels, scores = gather(run, TALKERS, t)
        tiers = np.concatenate([run[x]["tier"][run[x]["lang"] == t] for x in TALKERS])
        phoneme = gather(run, TALKERS, t, "phoneme")[1]
        out[f"enroll_{e}_test_{t}"] = {
            "trials": len(labels), "genuine": int(labels.sum()), "clones": int((~labels).sum()),
            "eer": summary(eer(labels, scores)[0], bootstrap(lambda ts: eer(*gather(run, ts, t))[0], draws),
                           {x: eer(*gather(run, [x], t))[0] for x in TALKERS}),
            "auc": summary(auc(labels, scores), bootstrap(lambda ts: auc(*gather(run, ts, t)), draws),
                           {x: auc(*gather(run, [x], t)) for x in TALKERS}),
            "tiers": {k: int(v) for k, v in zip(*np.unique(tiers, return_counts=True))},
            "neutral_phoneme_score_0.5": int(np.sum(phoneme == 0.5)),
        }
    return out


def language_auc(run, talkers, e, kind):
    """S3: positive = trials in the profile's language, negative = the other language."""
    pos = gather(run, talkers, e, kind=kind)[1]
    neg = gather(run, talkers, OTHER[e], kind=kind)[1]
    labels = np.concatenate([np.ones(len(pos), bool), np.zeros(len(neg), bool)])
    return auc(labels, np.concatenate([pos, neg]))


def secondary(runs, draws):
    out = {"S1": cells(runs, draws)}
    pooled = shifts(runs["en"], TALKERS, "en", "de")
    boot = bootstrap(lambda ts: shifts(runs["en"], ts, "en", "de"), draws)
    out["S2"] = {k: summary(pooled[k], [b[k] for b in boot]) for k in ["genuine_shift", "clone_shift"]}
    out["S2"]["threshold"] = pooled["threshold"]
    out["S3"] = {f"enroll_{e}_{kind}": summary(language_auc(runs[e], TALKERS, e, kind),
                                                bootstrap(lambda ts: language_auc(runs[e], ts, e, kind), draws))
                 for e in LANGS for kind in ["genuine", "clone"]}
    out["S4"] = {}
    for e in LANGS:  # cross cells: test language t = other
        t, run = OTHER[e], runs[e]
        fn = lambda ts, p: auc(*gather(run, ts, t, prompt=p))
        out["S4"][f"enroll_{e}_test_{t}"] = {
            f"auc_prompt_{e}_enrollment_language": round(fn(TALKERS, e), 3),
            f"auc_prompt_{t}_test_language": round(fn(TALKERS, t), 3),
            "difference": summary(fn(TALKERS, e) - fn(TALKERS, t),
                                  bootstrap(lambda ts: fn(ts, e) - fn(ts, t), draws))}
    out["S5"] = {}
    for e, t in product(LANGS, LANGS):
        for branch in ["phoneme", "speaker"]:
            labels, values = gather(runs[e], TALKERS, t, branch)
            out["S5"][f"enroll_{e}_test_{t}_{branch}"] = {"eer": round(eer(labels, values)[0], 3),
                                                          "auc": round(auc(labels, values), 3)}
    fn = lambda ts: auc(*gather(runs["de"], ts, "en", "speaker", prompt="de"))
    out["S5"]["hypothesis_speaker_auc_enroll_de_test_en_prompt_de"] = summary(fn(TALKERS), bootstrap(fn, draws))
    out["S6"] = {}
    for t in LANGS:  # score under the cross profile minus score under the same-language profile
        for kind in ["genuine", "clone"]:
            same = gather(runs[t], TALKERS, t, kind=kind)[1]
            cross = gather(runs[OTHER[t]], TALKERS, t, kind=kind)[1]
            diff = cross - same
            out["S6"][f"test_{t}_{kind}"] = {
                "trials": len(diff), "median": float(np.median(diff)),
                "quartiles": [float(q) for q in np.percentile(diff, [25, 75])],
                "share_lower_under_cross": round(float(np.mean(diff < 0)), 4)}
    out["S7"] = primary(runs, draws, sens=True)
    out["S7"]["clones_left_out_per_run"] = int(sum(runs["de"][t]["sensitivity_out"].sum() for t in TALKERS))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--de", required=True, help="run ID, German enrollment")
    parser.add_argument("--en", required=True, help="run ID, English enrollment")
    parser.add_argument("--de-wrong", required=True, help="run ID, German enrollment, wrong speaker")
    parser.add_argument("--en-wrong", required=True, help="run ID, English enrollment, wrong speaker")
    args = parser.parse_args()
    run_ids = {"de": args.de, "en": args.en, "de_wrong": args.de_wrong, "en_wrong": args.en_wrong}
    sensitivity = ROOT / "protocols/emime/sensitivity_clones.csv"
    leave_out = {Path(r["file"]).name for r in csv.DictReader(sensitivity.open())}
    runs, failed = {}, {}
    for key, run_id in run_ids.items():
        runs[key], failed[key] = load_run(run_id, leave_out)
    # paired comparisons drop a trial that failed in only one run (Section 2)
    if any(failed.values()):
        sys.exit(f"failed trials present, pairing rule needed first: { {k: len(v) for k, v in failed.items()} }")
    draws = np.random.default_rng(SEED).integers(0, len(TALKERS), size=(DRAWS, len(TALKERS)))
    draws = [[TALKERS[i] for i in d] for d in draws]

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(), "purpose": "emime_analysis_plan",
        "analysis_plan": "protocols/emime/analysis-plan.md",
        "analysis_plan_sha256": sha256(ROOT / "protocols/emime/analysis-plan.md"),
        "command": [sys.executable, *sys.argv],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        "script_sha256": sha256(__file__), "runs": run_ids,
        "scores_sha256": {k: sha256(ROOT / "results/runs" / v / "scores.csv") for k, v in run_ids.items()},
        "sensitivity_list_sha256": sha256(sensitivity),
        "failed_trials": {k: len(v) for k, v in failed.items()},
        "settings": {"seed": SEED, "draws": DRAWS, "margin_eer_points": MARGIN, "units": "percent",
                     "accept_rule": "score >= threshold", "unit_of_inference": "talker (14)"},
        "note_per_talker": "per-talker EER moves in steps of about 1.3 points (75 genuine trials)",
    }
    result["primary"] = primary(runs, draws)
    result["control"] = control(runs, draws)
    result["secondary"] = secondary(runs, draws)
    out = ROOT / "results/analysis/emime" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    (out / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"primary": {k: {kk: v.get(kk) for kk in ["pooled", "ci95", "p_holm", "reading", "talkers_positive"]}
                                  if isinstance(v, dict) else v for k, v in result["primary"].items()},
                      "control": {k: {kk: v[kk] for kk in ["right_eer", "wrong_eer", "pooled", "ci95"]}
                                  for k, v in result["control"].items()}}, indent=1))
    print(f"analysis: {out}")


if __name__ == "__main__":
    main()
