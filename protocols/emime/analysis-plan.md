# EMIME cross-language evaluation: analysis plan

Status: **binding from its commit.** Written by the user on 2026-11-15, confirmed on
2026-11-16, before any PVP score on EMIME exists. Seen before the commit: the clone
generation record and the clone-quality results (no PVP score); they set only the
sensitivity list of Section 2. This commit is older than the first cross-language run ID.

Purpose: fix what is measured and how it is judged before results are seen, so that the
main results cannot be chosen after the fact. This is an evaluation of the unchanged PVP
method. It is separate from the reproduction and adds no method change.

## 0. Decisions to confirm before the commit

Confirmed by the user on 2026-11-16 (D4 explicitly: a real attacker does not filter
their fakes either). After the commit they change only through Section 9.

| # | Decision | Proposal |
| --- | --- | --- |
| D1 | Equivalence margin for EER differences | ±2.0 EER points (absolute, pooled) |
| D2 | Primary contrasts | P1-P4 in Section 3, no others |
| D3 | Wrong-speaker pairing | Rotation inside each sex (Section 5) |
| D4 | Clone exclusion | None for quality reasons in the primary analysis (Section 2) |
| D5 | Confirmation sets | Generators 2 and 3, VCC2020, Finnish talkers (Section 7) |
| D6 | Resampling | 10,000 draws over talkers, seed 20261115 |

## 1. Data and runs

Protocol files in this folder at commit `73818fa`: `emime_de_enroll.csv`,
`emime_en_enroll.csv` (60 sentences per talker each), `emime_test.csv`, `excluded.csv`.

Test list, 6,297 trials, 14 German talkers (GF1-GF7, GM1-GM7):

| `Source` | Meaning | Trials |
| --- | --- | --- |
| `de-genuine` | genuine German | 1,049 |
| `en-genuine` | genuine English (talkers' second language) | 1,048 |
| `de-qwen3tts-refde`, `de-qwen3tts-refen` | German clones, voice prompt German / English | 1,050 each |
| `en-qwen3tts-refde`, `en-qwen3tts-refen` | English clones, voice prompt German / English | 1,050 each |

Two scientific runs with upstream code settings (MMS backbone, alpha 0.8, top 12), each
over the full test list:

- **Run DE**: German enrollment.
- **Run EN**: English enrollment.

Every trial therefore gets two scores, one per profile. Notation: E = enrollment
language, T = test language, P = prompt language of a clone. A *cell* is one pair (E, T):
genuine trials of language T against all clones of language T (both prompt languages),
scored with the profile of language E. "Same" means E = T, "cross" means E ≠ T.

Each run needs a clean committed checkout, `--scientific`, and one replay compared with
`scripts/compare_runs.py`. Speaker profiles are not uploaded to W&B.

## 2. Trials that enter the analysis

- All trials with `status = scored` enter. This includes trials where upstream returns
  the neutral phoneme score 0.5 (no evidence); they are counted per cell by tier.
- Failed trials have no score. They are counted and listed per cell and run, never
  silently dropped. A trial that fails in only one run is left out of paired
  comparisons, and the number is reported.
- The three files in `excluded.csv` were removed before any scoring. Their clones stay
  in the test list (3 clones without a genuine partner).
- **No clone is excluded for quality reasons in the primary analysis.** A poor fake is
  still a fake. Sensitivity analysis (secondary): repeat P1-P4 without clones whose
  Whisper word error rate is above 0.2 or whose target-language probability is below
  0.5 (`scripts/check_clone_quality.py`), and report how many were removed.
  *Changed before the commit (2026-11-16), after the clone-quality results and before
  any PVP score:* an absolute word error limit flags genuine files as often as clones
  (WER > 0.2: genuine 107 of 2,097, clones 198 of 4,200; mostly news sentences with
  numbers and abbreviations). The sensitivity analysis therefore leaves out a clone if
  Whisper detects the wrong language, or if its character error rate is more than 0.10
  above that of the matched genuine file. The list is fixed in `sensitivity_clones.csv`
  (63 clones), made by `scripts/select_sensitivity_clones.py` from quality run
  `20261115T121027Z-676c77ea`.

## 3. Primary contrasts

Metrics: EER and ROC-AUC as in `scripts/evaluate_pvp.py` (upstream `compute_eer`,
scikit-learn AUC; label 1 = bona fide, higher score = more consistent with the profile),
reported in percent.

**RQ1, transfer cost.** Same test trials, two profiles.

- **P1 (main direction):** English test trials. EER(E = German) − EER(E = English).
- **P2 (reverse):** German test trials. EER(E = English) − EER(E = German).

The same differences are reported for AUC, but EER decides.

**RQ2, decomposition by threshold transfer.** Take the German profile (Run DE). Fix the
threshold at the pooled EER point of its same-language cell (T = German). Apply it to
the English trials.

- **P3 (genuine shift):** false rejection rate of genuine English at that threshold,
  minus the false rejection rate of genuine German at that threshold.
- **P4 (clone shift):** false acceptance rate of English clones at that threshold,
  minus the false acceptance rate of German clones at that threshold.

P3 measures how much language mismatch makes genuine speech look inconsistent. P4
measures how much clones gain or lose. Both are free of the score scale.

Hypotheses: P1 > 0 and P2 > 0 (cross is worse); P3 > 0 and P3 > P4 (the genuine shift
dominates).

## 4. Statistics and decision rules

The unit is the talker (n = 14). Trials of one talker are not independent.

- **Intervals:** resample talkers with replacement (all trials of a drawn talker, a
  talker drawn twice counts twice), recompute the pooled contrast in each draw;
  percentile 95% interval (D6).
- **Tests:** exact sign-flip permutation test (all 16,384 sign patterns) on the 14
  per-talker values of each contrast, two-sided. For P1 and P2 the per-talker value is
  the per-talker EER difference; for P3 and P4 the per-talker rate difference at the
  pooled threshold. Holm correction over P1-P4, level 0.05.
- **Also reported for each contrast:** the 14 per-talker values, their mean, and the
  number of talkers with a positive value.

Reading of P1 and P2 with margin m (D1):

| 95% interval | Statement |
| --- | --- |
| entirely inside [−m, +m] | no meaningful change; PVP transfers in this setting |
| entirely above +m | meaningful loss |
| excludes 0 but overlaps the margin | loss, size uncertain |
| anything else | inconclusive |

Reading of RQ2: "the genuine shift dominates" is stated only if the interval of P3
excludes 0 and the interval of P3 − P4 excludes 0. The opposite statement needs the
same for P4 − P3. Otherwise: both shifts reported, no dominance claimed.

Per-talker EER moves in steps of about 1.3 points (75 genuine trials per talker and
language). This limit is stated with every per-talker result.

## 5. Control: wrong-speaker profile (secondary, run with the first runs)

Question: does detection on this corpus use the speaker's profile, or would any profile
do? All genuine files come from one studio and all clones from one generator, so a
profile-independent shortcut is possible.

Each talker's test trials are scored against the profile of another talker of the same
sex: GF1 → GF2, GF2 → GF3, ..., GF7 → GF1, and the same for GM. This needs a derived
test list in which only the `Speaker` column is replaced; it is created by a script and
committed before the control runs. One control run per enrollment language.

Reported per cell: EER(wrong profile) − EER(right profile) with the interval of
Section 4. If this interval does not lie entirely above +m, the report states that
speaker-specific detection is not shown on this corpus, and all other results are read
with that limit.

## 6. Secondary analyses (planned, reported in full, not corrected as primary)

- **S1, all cells:** pooled and per-talker EER and AUC for the four cells of both runs,
  with intervals; tier counts and failed-trial counts per cell.
- **S2, reverse threshold transfer:** P3 and P4 with the English profile and German
  test trials.
- **S3, language effect on genuine speech:** with one profile, AUC of genuine speech in
  the profile's language against genuine speech in the other language (50% = none).
  The same for clones.
- **S4, prompt language:** in each cross cell, AUC against clones with P = E and
  against clones with P = T, and their difference. Hypothesis: clones with P = E are
  harder to detect.
- **S5, branches:** the same cells for the phoneme score alone and for the speaker
  score alone. The speaker score is derived as `(score − 0.8 · phoneme_score) / 0.2`.
  Hypothesis from the clone pilot: in the cell E = German, T = English, the speaker
  branch separates clones with German prompts from genuine English with AUC below 50%.
- **S6, paired score change:** per trial, score under the cross profile minus score
  under the same-language profile, for genuine trials and clones separately.
  Descriptive, because the sigmoid compresses the score scale.
- **S7, sensitivity:** the clone-quality filter of Section 2.

## 7. Confirmation sets

The following data are not used to choose analyses or to develop any later method
change. They are scored once with the pipeline as frozen here, and reported as
confirmation of P1-P4:

- clones from the second and third generator (not generated yet);
- VCC2020 Task 2 conversions of the two German EMIME targets;
- the Finnish talkers of EMIME (genuine-only parts: S3).

Each gets a dated addendum to this plan (data, trial list, commit) before its first
score.

### Addendum A (2026-11-16): Finnish talkers, S3

Written before any Finnish score. Seen before it: all German-talker results of Sections
3-6 (analysis `results/analysis/emime/20261116T141441Z`) and the Finnish file check.

- Data: EMIME v1.0 Finnish talkers FF1-FF7, FM1-FM7, 22 kHz, microphone 0
  (`scripts/prepare_emime.py --group finnish`, report
  `results/datasets/emime-finnish-talkers-check.json`): 7,970 files, no decode error, no
  identical audio, no near-silent file, nothing excluded. FM4 and FM7 lack microphone-0
  Finnish sentences 5-54 (FM7 not stated in the README); listed in
  `finnish/missing.csv` and left out.
- Phoneme recognizer setup check on 4 Finnish files: plausible Finnish phonemes, all
  inside the upstream phoneme classes (`vocab.json`).
- Same sentence split as the German talkers (same seed and types); no clones, so the
  reference sentences are unused.
- Protocols in `finnish/`: `emime_fi_enroll.csv` (798 files; FM4 and FM7 39 each),
  `emime_en_enroll.csv` (840), `emime_test.csv` (2,046 genuine: Finnish 996, English 1,050).
- Runs: Finnish enrollment and English enrollment over the full test list, upstream code
  settings (`configs/pvp-emime-finnish-{fi,en}-enroll-mms.json`), one replay each.
- Analysis: S3 only (Section 6): per profile, AUC of genuine speech in the profile's
  language against genuine speech in the other language, talker bootstrap as in
  Section 4; per-talker values (AUC minus 50) with the sign-flip test. Reading: the
  language effect on genuine speech is confirmed for a profile if its 95% interval lies
  entirely above 50%. Reported next to the German talkers' S3 values (German profile
  87.58, English profile 82.75).

### Addendum B (2026-11-16): generators 2 and 3

Written before any clone of these generators exists. Seen before it: all German-talker
results and the Finnish S3 result. User approval of both generators and their weight
downloads: 2026-11-16.

- Generators: VoxCPM2 (`openbmb/VoxCPM2`, revision `7936080b`, Apache 2.0, 48 kHz output;
  model card "ultimate cloning": the voice prompt as reference and as prompt with its
  transcript, `cfg_value` 2.0, 10 steps, no denoiser, no text normalization) and XTTS v2
  (`coqui/XTTS-v2`, revision `6c2b0d75`, Coqui Public Model License, non-commercial, 24 kHz;
  `Xtts.synthesize` with the checkpoint's settings, voice prompt audio only, no transcript).
  Weights pinned in `configs/clone-generators.json`; environments from
  `requirements/{voxcpm,xtts}-linux-py312.lock`. Setup checks on Qwen's example voice
  (`scripts/check_clone_generator.py`): offline loading, seeded repeats bit-identical.
- Each clone uses the same voice-prompt recording as the Qwen3-TTS clone of the same trial;
  its seed comes from its own file name. Test lists `emime_test_voxcpm2.csv` and
  `emime_test_xtts.csv` (`scripts/make_generator_protocol.py`): genuine rows identical to
  `emime_test.csv`, clone paths and `Source` renamed.
- Gate before full generation, as for Qwen3-TTS: 40-clone pilot (GF1, GM1), quality checks
  and the user's listening verdict. No clone is excluded for quality reasons (D4); quality
  results are reported per generator.
- Runs: German and English enrollment per generator over its full test list, upstream code
  settings, one replay each. The wrong-speaker control is not repeated.
- Analysis: P1-P4 with the statistics and readings of Section 4, per generator, plus the
  four cells (S1). A primary result is confirmed for a generator if its reading equals the
  Qwen3-TTS reading (P1 and P2: "meaningful loss"; RQ2: "the genuine shift dominates").
  Genuine trials are the same files as in the main runs, so their scores must equal the
  main runs' scores; any difference is reported.

## 8. Not covered here

- **Phoneme-level analysis (RQ3).** It needs per-occurrence log-likelihoods, which the
  runner does not store yet. One rule is fixed now: a phoneme type is *shared* for a
  talker if the recognizer finds it at least 5 times in both the German and the English
  enrollment set (5 is upstream's minimum for a phoneme model); otherwise it is
  specific to the language where it reaches 5. The groups depend on enrollment audio
  only. The remaining details get an addendum before the first per-phoneme result.
- **Enrollment mixing curve (RQ4), native English talkers:** addendum before their
  first score.
- **Calibration measures such as Cllr:** PVP scores are not likelihood ratios; not
  part of this plan.
- **Exploratory, always labelled as such:** talker sex, sentence type (Europarl, news,
  semantically unpredictable), duration, single tiers, fusion-weight sweeps.

Known limits that no analysis here removes: one generator; English is the talkers'
second language, so language and accent cannot be separated; read studio speech; clones
are about 14-20% shorter than genuine speech (median ratio 0.80-0.86, all clones) and come from 24 kHz output while
genuine files are 22.05 kHz (upstream resamples both to 16 kHz).

## 9. Amendments

Changes after the commit are listed here with date, reason and the run IDs that existed
at that time. Results affected by a change made after seeing scores are labelled
exploratory.

| Date | Change | Reason | Scores seen at that time |
| --- | --- | --- | --- |
| - | - | - | - |
