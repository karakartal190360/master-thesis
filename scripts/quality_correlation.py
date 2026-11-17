"""Clone quality vs PVP score within each generator (exploratory, not in the analysis plan).

Fixed on 2026-11-17 before any value was computed; seen before: all P1-P4 and S1 results of
the three generators and their per-condition quality medians. Question: inside one generator
(same model, settings, voice prompts, sentences), do clones with better quality measures get
more genuine-like PVP scores? Exploratory: no confirmatory reading, p values uncorrected.

- Clones: every clone of each generator's two official runs (German and English profile).
- Quality measures (results/clone-quality/<clone run>/quality.csv), each with the direction
  "better quality" expected to give a higher PVP score:
  utmos (naturalness, +), sim_profile (WavLM similarity to the talker's enrollment centroid
  in the profile's language, +), cer (character error rate, -), length_error
  (|log duration ratio|, clone vs genuine of the same sentence, -).
- PVP values: final score (decision), and its phoneme and speaker (ECAPA) parts.
- Cells: profile language x clone language (4). Spearman rho per talker and cell
  (150 clones: 75 sentences x 2 voice-prompt languages), so talker differences do not
  create a correlation. Undefined rho (a constant measure) is left out and counted.
- Summary per cell: mean per-talker rho, talker bootstrap 95% interval (10,000 draws,
  seed 20261115), exact two-sided sign-flip test, talkers with rho > 0.
- Reference: the same rho for genuine speech (utmos, sim_profile, cer; no duration ratio),
  to see whether PVP tracks a measure for real speech as well.
- Between generators: medians of each measure next to the same-language EER (3 points,
  descriptive only).
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from analyze_emime import ALPHA, DRAWS, MAIN_RUNS, ROOT, SEED, TALKERS, interval, sha256, sign_flip

GENERATORS = {
    "qwen3tts": {**MAIN_RUNS, "quality": "20261115T121027Z-676c77ea",
                 "analysis": "results/analysis/emime/20261116T145229Z/analysis.json"},
    "voxcpm2": {"de": "20261117T141912Z-8a48f2c4", "en": "20261117T142422Z-4ffaafc4",
                "quality": "20261117T083513Z-b64c030a",
                "analysis": "results/analysis/emime-voxcpm2/20261117T144059Z/analysis.json"},
    "xtts": {"de": "20261117T144059Z-275bdd0d", "en": "20261117T144613Z-6271e05c",
             "quality": "20261117T114700Z-45ea6df0",
             "analysis": "results/analysis/emime-xtts/20261117T150247Z/analysis.json"},
}
CLONE_MEASURES = ["utmos", "sim_profile", "cer", "length_error"]
GENUINE_MEASURES = ["utmos", "sim_profile", "cer"]
EXPECTED = {"utmos": "+", "sim_profile": "+", "cer": "-", "length_error": "-"}
PARTS = ["score", "phoneme", "speaker"]
LANGS = ["de", "en"]


def load(system, profile):
    """Rows of one run joined with the quality record of the same file."""
    paths = GENERATORS[system]
    quality = {Path(r["file"]).name: r for r in
               csv.DictReader((ROOT / "results/clone-quality" / paths["quality"] / "quality.csv").open(encoding="utf-8"))}
    rows = []
    for r in csv.DictReader((ROOT / "results/runs" / paths[profile] / "scores.csv").open()):
        assert r["status"] == "scored", r  # all six runs have no failed trial
        q = quality[r["file"]]
        score, phoneme = float(r["score"]), float(r["phoneme_score"])
        ratio = q["duration_ratio"]
        rows.append({"talker": r["speaker"], "kind": "genuine" if r["label"] == "bonafide" else "clone",
                     "lang": q["language"], "score": score, "phoneme": phoneme,
                     "speaker": (score - ALPHA * phoneme) / (1 - ALPHA),
                     "utmos": float(q["utmos"]), "sim_profile": float(q[f"sim_{profile}"]),
                     "cer": float(q["cer"]),
                     "length_error": abs(np.log(float(ratio))) if ratio else None})
    return rows


def per_talker_rho(rows, kind, lang, measure, part):
    result = {}
    for t in TALKERS:
        # 6 clones lack a duration ratio (their genuine sentence is one of the 3 excluded files)
        own = [r for r in rows if r["talker"] == t and r["kind"] == kind and r["lang"] == lang
               and r[measure] is not None]
        x, y = [r[measure] for r in own], [r[part] for r in own]
        if len(set(x)) > 1 and len(set(y)) > 1:
            result[t] = float(spearmanr(x, y).correlation)
    return result


def summarize(rhos, draws):
    values = np.array(list(rhos.values()))
    boot = [np.mean([rhos[t] for t in d if t in rhos]) for d in draws]
    return {"mean_rho": round(float(values.mean()), 3), "ci95": interval(boot),
            "p_sign_flip": sign_flip(values), "talkers": len(values),
            "talkers_positive": int((values > 0).sum()),
            "per_talker": {t: round(v, 3) for t, v in rhos.items()}}


def correlations(rows, kind, measures, draws):
    out = {}
    for profile_lang, lang_rows in rows.items():
        for lang in LANGS:
            cell = f"profile_{profile_lang}_test_{lang}"
            out[cell] = {m: {p: summarize(per_talker_rho(lang_rows, kind, lang, m, p), draws)
                             for p in PARTS} for m in measures}
    return out


def between(system, rows):
    analysis = json.loads((ROOT / GENERATORS[system]["analysis"]).read_text())
    clones = [r for r in rows["de"] if r["kind"] == "clone"]
    medians = {m: round(float(np.median([r[m] for r in clones if r[m] is not None])), 4)
               for m in ["utmos", "cer", "length_error"]}
    medians["sim_profile_de"] = round(float(np.median([r["sim_profile"] for r in clones if r["lang"] == "de"])), 4)
    s1 = analysis.get("S1") or analysis["secondary"]["S1"]
    return {"clone_medians": medians,
            "same_language_eer": {k: s1[k]["eer"]["pooled"] for k in ["enroll_de_test_de", "enroll_en_test_en"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    draws = np.random.default_rng(SEED).integers(0, len(TALKERS), size=(DRAWS, len(TALKERS)))
    draws = [[TALKERS[i] for i in d] for d in draws]
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(), "purpose": "exploratory_quality_correlation",
        "scientific_status": "exploratory, defined after the generator results; not in the analysis plan",
        "command": [sys.executable, *sys.argv],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        "script_sha256": sha256(__file__), "inputs": GENERATORS,
        "inputs_sha256": {s: {"scores_de": sha256(ROOT / "results/runs" / g["de"] / "scores.csv"),
                              "scores_en": sha256(ROOT / "results/runs" / g["en"] / "scores.csv"),
                              "quality": sha256(ROOT / "results/clone-quality" / g["quality"] / "quality.csv")}
                          for s, g in GENERATORS.items()},
        "settings": {"seed": SEED, "draws": DRAWS, "correlation": "Spearman per talker and cell",
                     "expected_direction": EXPECTED, "unit_of_inference": "talker (14)"},
        "generators": {},
    }
    for system in GENERATORS:
        rows = {p: load(system, p) for p in LANGS}
        result["generators"][system] = {"clones": correlations(rows, "clone", CLONE_MEASURES, draws),
                                        "genuine_reference": correlations(rows, "genuine", GENUINE_MEASURES, draws),
                                        "between": between(system, rows)}
        print(system, flush=True)
    out = ROOT / "results/analysis/quality-correlation" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    (out / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    for system, g in result["generators"].items():
        print(f"\n{system} (final score; mean per-talker rho [95% CI], talkers > 0)")
        for cell, measures in g["clones"].items():
            print(" ", cell, "  ".join(f"{m} {v['score']['mean_rho']:+.2f} {v['score']['ci95']} {v['score']['talkers_positive']}/{v['score']['talkers']}"
                                        for m, v in measures.items()))
    print(f"analysis: {out}")


if __name__ == "__main__":
    main()
