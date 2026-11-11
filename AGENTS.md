# Personalized speech deepfake detection

## Objective and sources

Research at Johannes Kepler University Linz (JKU), Master's in Artificial Intelligence.
Build a scientifically reproducible baseline before investigating a new contribution.

- Primary scientific reference: `Papers/profiling-the-voice.pdf`, Xue et al. (2026),
  *Profiling the Voice: Speaker-Specific Phoneme Fingerprinting for Speech Deepfake Detection*.
- Primary code reference: <https://github.com/JunXue-tech/PVP>, pinned at
  `1eed3f43309e6728583a5e43595e68121472b2ca` in `original/PVP/`.
- Read relevant paper sections and source files before implementation decisions.
  Compare them; never assume they agree. Record differences in `docs/reproduction.md`.
- Never invent methods, findings, measurements, or successful reproduction claims.

## Stage A: Practical Work in AI

Priority: November 2026-February 2027. Initial PVP reproduction target: end of November 2026.
Understand PVP, prepare its environment and official benchmark protocols, reproduce selected
experiments, compare EER and ROC-AUC with the paper, then evaluate additional datasets
and benchmarks selected later. German/English and cross-language evaluation are possible
directions, depending on the selected data. Write a report covering methods, results,
problems, and limitations. Success means an executed, reproducible baseline with measured results.
If time allows after the pipeline is complete, add an audio-upload interface with visual
explanations of phoneme scores and their contributions. This interface is optional.

`practical-work` is the main branch for Stage A and contains only practical work.
Keep its README concise. The `main` README preserves detailed project/setup information.
Future thesis work will use a separate `master-thesis` branch based on the practical work.

## Stage B: Master's Thesis

Planned: March-July 2027. Candidate direction: cross-language generalization and adaptation
of personalized phoneme-based speech deepfake detection. Open questions:

- Can German enrollment detect English deepfakes of the same speaker?
- Which speaker-specific phonetic characteristics transfer across languages?
- Can detection improve without target-language enrollment recordings?

Agree on the final contribution with the supervisor after Stage A and further literature work.
Phase 7 is not authorized for implementation. Do not add adaptation algorithms, architectures,
or thesis contributions during baseline reproduction. Keep new evaluations separate from reproduction.

## Roadmap and progress

This is the single project roadmap. Use only Not Started, In Progress, Completed, or Blocked.
Update it for completed milestones, changed research decisions, and important blockers;
do not rewrite it for trivial edits. Technical notes elsewhere are evidence, not a second roadmap.

| Phase | Status | Work and completion evidence |
| --- | --- | --- |
| 1. Research and repository setup | Completed | Paper read; upstream inspected and pinned; pipeline, files, dependencies, protocol counts, and discrepancies documented; instructions and initial structure created. No experiment executed. |
| 2. Environment setup | Completed | Linux PVP/CUDA checks pass. W&B 0.26.1 was added without changing the existing 95 packages; offline and online non-scientific tracking checks pass in the private academic project. Mac checks pass with CPU thread limits. User-approved downloads of the three pinned models completed on 2026-11-04: 14 files, 2.62 GB, sizes and available published weight hashes verified. On 2026-11-05: MMS keys matched upstream and were converted losslessly to `mms_300m.ckpt`; all three models load offline through unchanged upstream code with weights equal to their files; CTC/MMS frame counts agree; peak GPU memory 3.3 GB allocated at 30 s synthetic input (setup check, `scientific_result=false`). |
| 3. Dataset preparation | Completed | ZH-Famous downloaded (64.3 GB), hash-verified, mapped to the upstream protocols (9 LLaSA test files missing from the release), and extracted to WAV on 2026-11-05. Famous Figures downloaded (170.4 GB) and hash-verified on 2026-11-06; on the same day all 348,136 protocol rows were mapped one-to-one to release rows with an inferred `_<Source>` suffix rule (release paths repeat across systems) and extracted to WAV under protocol names (179.8 GB; 13,399 bona fide files are stereo, downmixed by upstream on load). On 2026-11-08 `scripts/link_pretrained.py` linked protocols and both audio roots into the upstream working directory; all protocol rows resolve except the 9 known missing LLaSA files. On 2026-11-08 every protocol file was decoded (no errors) and checked for identical audio, shared transcripts (ZH) and voice prompts (Famous Figures): 60 (ZH) and 34 (Famous Figures) bona fide test files are exact copies of enrollment files; many LLaSA spoofs are under 0.5 s; files reach 79.5 s. Protocols stay unchanged. |
| 4. PVP reproduction | Completed | Authorized 2026-11-10. Runner `scripts/run_pvp.py` (unchanged upstream code, every trial recorded with score, tier or failure reason, full provenance, W&B artifact), evaluator `scripts/evaluate_pvp.py` with unit tests, configs for both benchmarks (upstream code settings). Smoke checks on both datasets replayed bit-identically. Scientific MMS-300M runs: ZH-Famous `20261110T161146Z-a7e7dcab` EER 12.24% / AUC 93.62% (paper 11.37 / 94.58); EN-Famous `20261110T194052Z-89c7e58d` EER 7.34% / AUC 96.58% (paper 7.24 / 96.61). Failed trials recorded (9 missing files, 334 very short spoofs). On 2026-11-11 the user limited the scope to the default MMS backbone and accepted the results. Summary and limitations in `docs/reproduction.md`. |
| 5. Reproducible experiments | Not Started | Partly covered by the Phase 4 runner: configs, run records, W&B artifacts and smoke-check replay exist. Remaining scope (e.g. full-run replay, comparison tooling) to be defined. Basic provenance is required for every experiment. |
| 6. Additional evaluations and report | Not Started | Select suitable additional datasets and benchmarks later; document their protocols and keep these results separate from reproduction. German/English and cross-language evaluation remain possible directions. Write the practical work report. If time allows after the pipeline is complete, add a visual explanation interface. Target: February 2027. |
| 7. Master's Thesis | Not Started | With supervisor approval, define contribution; extend baseline; run experiments and ablations; evaluate and write thesis. March-July 2027; implementation not authorized. |

Phase 1 review was followed by authorization to continue on 2026-11-03. Phase 2 was
completed on 2026-11-05: software, model downloads, MMS conversion, offline model loading,
and GPU inference memory on synthetic input are verified. Phase 3 started with ZH-Famous on 2026-11-05;
Famous Figures was downloaded, verified, mapped, and extracted on 2026-11-06.
Phase 3 was completed on 2026-11-08 with runtime links and audio integrity/leakage checks.
Phase 4 was completed on 2026-11-11 with MMS-300M on both benchmarks; get user confirmation before starting the next phase.
The user made the home server reachable on 2026-11-04. Use the configured SSH alias
`homeserver`; it opens a Windows shell. Run Linux commands through `wsl.exe -d Ubuntu`.
The project copy is `/home/mut/projects/master-thesis` in WSL, on `practical-work`,
with the GitHub origin configured and current unpushed files preserved. It has its
own verified Linux `.venv`; never copy the Mac `.venv` or `.cache` there. Direct-desktop
setup remains supported; keep machine connection details out of reusable Python code.
Reproduced MMS results (2026-11-10/11): ZH-Famous close to the paper (+0.87 EER points), EN-Famous very close (+0.10); see `docs/reproduction.md`.

Known constraints: both benchmarks are gated; the Famous Figures mapping relies on an inferred file-name rule;
both benchmarks contain exact enrollment/test audio copies and sub-25 ms spoof files (see `docs/reproduction.md`); pinned model files
are in the server's ignored `models/` directory and reach upstream only through the
symlinked working directory from `scripts/link_pretrained.py`; upstream
has no full dependency lock; speaker-score settings and enrollment counts differ from the paper.
See `docs/reproduction.md` and `docs/environment.md` for evidence and unresolved details.
The first baseline must use Xue's official datasets and supplied enrollment/test protocols.
W&B Academic is selected: entity `umuttanriverdi`, private project
`personalized-speech-deepfake-detection`. See `docs/tracking.md` for setup evidence.

## Repository responsibilities

| Path | Responsibility |
| --- | --- |
| `AGENTS.md` / `CLAUDE.md` | Central instructions and roadmap / import only. |
| `README.md` | Short introduction and setup entry point. |
| `Papers/` | Primary paper PDFs; preserve existing files and add relevant literature here. |
| `original/PVP/` | Unchanged upstream snapshot; never mix our edits with it. |
| `original/UPSTREAM.md`, `original/manifest.json` | Source commit and file provenance. |
| `docs/` | Research, environment, reproduction, and discrepancy notes. |
| `requirements.txt`, `requirements/` | Direct macOS development requirements and separate Mac/Linux locks. Linux software and pretrained model checks pass. |
| `src/` | Future reusable Python code. |
| `scripts/` | Environment setup/checks, MMS conversion, and the upstream runtime-link layout; experiment entry points when reproduction begins. |
| `configs/` | Pinned model download plan and W&B destination; experiment settings when needed. |
| `notebooks/` | Future exploration, figures, and result analysis. |
| `data/` | Future local datasets and protocols; keep raw audio out of Git. |
| `models/` | Downloaded pretrained files on WSL and future fitted profiles; out of Git. |
| `results/` | Future scores, logs, plots, and run artifacts; out of Git by default. |
| `tests/` | Future small, meaningful verification tests. |

Create future folders when useful; do not add empty modules or placeholder frameworks.
Keep code in `.py` files. Do not duplicate notebook and module implementations.
Do not move or rename existing files without a clear reason. Keep datasets, weights,
caches, and generated outputs out of Git; retain small protocols/configs and evidence notes.

## Python and writing style

- Simplicity before cleverness: standard Python, clear execution flow, small functions
  with one responsibility, descriptive short names. Use libraries where appropriate.
- Avoid unnecessary classes/inheritance, frameworks, patterns, abstractions, helpers,
  dependencies, nested logic, one-liners, and defensive code for irrelevant cases.
- Use type hints only where they improve clarity. Do not simplify the scientific method
  merely to shorten the code. Preserve upstream behavior until a change is documented.
- Keep comments short and precise, mostly keywords. Explain non-obvious scientific
  decisions, algorithm steps, dataset/protocol assumptions, and reproduction differences.
  Do not comment on obvious operations or add long narrative comments.
- Write explanations, README files, and notes in clear B2 English. Explain technical
  terms when first introduced. Explain concepts when needed, not all at once.

## Reproducibility requirements

For every major result, save the dataset release and evaluation protocol; enrollment/test
split files and hashes; speaker IDs and labels; model/backbone names, revisions and weight
hashes; configuration and random seeds; our code revision and upstream commit; OS, hardware,
Python, CUDA/driver and relevant package versions; metric implementation and score direction;
per-trial predictions/scores; and all failed, skipped, or fallback trials with reasons.
Save the command and run identifier. Never silently drop failed trials or tune on the test set.
Separate paper-reported results, our reproduced results, and new German/English results.
Claim reproduction success only after execution and an explicit comparison with the paper.
Document all methodological changes and ambiguities before interpreting results.

Official scientific runs must require a clean, committed executing checkout. Record the
server's Git state, not just the Mac's; they are separate working copies. Save the resolved
configuration, schema/hash, seeds/deterministic settings, and local artifacts as well as
W&B records. Every rerun has a new ID and must support replay comparison. Setup checks
may be dirty only when labelled `scientific_result=false`; they are not reproduction results.
Bulk datasets, shared weights, and feature caches stay on the server. Never store credentials
in Git or configuration files. These rules are accepted requirements for the future runner.

## Development environment

- macOS MacBook Pro M5 Pro, 48 GB unified memory: local development and analysis.
  Do not assume CUDA or verified fairseq/Apple Silicon compatibility.
  Set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` for local checks/analysis;
  the legacy numeric stack crashes in the combined smoke check with default threads.
  The local SciPy 1.7.3 exception and runtime evidence are documented in `docs/environment.md`.
- Desktop RTX 4070 Ti SUPER, 16 GB VRAM: official experiments on Ubuntu under WSL2,
  acting as a private GPU server. SSH/WSL access, driver 610.88, PyTorch CUDA 12.4,
  the pinned fairseq build, and PVP dependency imports are verified. The three pinned
  models load offline and run together; peak memory was 3.3 GB at 30 s synthetic input.
- Keep reusable paths portable and configurable. Do not change the algorithm for Apple Silicon.
- Phase 1 installed no environment. Phase 2 is now authorized; upstream pins, tested
  environments, and compatibility findings are recorded in `docs/environment.md`.
- In Phase 2, prefer `uv` and a local `.venv` for our development code; isolate the upstream
  legacy environment if necessary. Check compatibility before version changes. Run code with
  the selected environment's Python and record how it was created. Add `ipykernel` when notebooks exist.

## Implementing and verifying changes

1. Inspect relevant existing files and paper sections; briefly explain the task's purpose.
2. Make the smallest useful change within the current phase. Keep original source unchanged;
   place our wrappers or explicitly documented patches outside `original/PVP/`.
3. Run appropriate checks. Prefer small meaningful tests for scientific behavior and metrics;
   do not run model-loading imports during read-only checks because they may download weights.
4. Report changes, actual checks, limitations, and the next concrete step. Update progress
   when a milestone, decision, or blocker changes.

Commit messages: one line of 5-10 words in clear B2 English. No em dashes.
No AI signature or trailer, such as `Co-Authored-By` or "Generated with" lines.

Work one major phase at a time. Phase 2 is complete; report its verification status to the
user and get confirmation before starting dataset preparation. Do not download large datasets
or model weights without approval. Do not push to GitHub without approval. If a required
resource is unavailable, report the blocker; do not replace it with a different methodology.
