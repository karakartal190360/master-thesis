# PVP: paper and implementation inspection

Inspected on 2026-11-03. This is a source review, not an experimental result.
The project roadmap and phase status live only in [AGENTS.md](../AGENTS.md).

## References

- Scientific source: [local paper](../Papers/profiling-the-voice.pdf), Xue et al.,
  IJCAI-26, printed pages 5991-5999 (9 PDF pages).
  SHA-256: `dc919ef320500c304f40a0f0dd3e52c24af7abbfc622b63fc14605dadfba7ed1`.
- Code source: [official PVP repository at the inspected commit](https://github.com/JunXue-tech/PVP/tree/1eed3f43309e6728583a5e43595e68121472b2ca).
  The complete unchanged snapshot is in [original/PVP](../original/PVP/).
  [Provenance](../original/UPSTREAM.md) and [file hashes](../original/manifest.json)
  identify the exact version. The code review covered every Python file, the
  installer, README, vocabulary, and all four protocol CSVs.

## Method described in the paper

A person of interest (POI) is the speaker being protected. Enrollment means building
a profile from genuine recordings of that person. A phoneme is a speech sound;
an embedding is a numerical representation of audio.

1. **Extract two representations.** A frozen self-supervised learning (SSL) backbone,
   fine-tuned for deepfake detection, produces acoustic vectors for each audio frame.
   A separate Connectionist Temporal Classification (CTC) phoneme recognizer provides
   sound boundaries. Mean-pool the acoustic frames inside each phoneme occurrence.
   An ECAPA-TDNN speaker-verification model also produces one global voice embedding
   per recording. (PDF p. 4, Fig. 3 and Sec. 4.1.)
2. **Fit genuine speaker profiles.** Fit a diagonal-covariance Gaussian Mixture Model
   (GMM), a mixture of Gaussian distributions, to each phoneme's enrollment vectors.
   Fit a separate GMM to global speaker embeddings. No spoof examples are used for
   profile fitting; this does not mean the pretrained acoustic backbone never saw
   spoof data. (PDF pp. 4-5, Sec. 4.2.)
3. **Select stable sounds.** Estimate phoneme reliability from average enrollment
   log-likelihood: `weight = exp(mean_log_likelihood / reliability_scale)`. Keep the
   top 12 phoneme types. (PDF p. 5, Eqs. 5-6 and Sec. 4.3.)
4. **Score new audio.** Compute the likelihood of test vectors under the enrolled
   models and map it through a sigmoid to a consistency score. Prefer salient
   phonemes; otherwise use other enrolled phonemes, then broad categories such as
   vowels and plosives. (PDF p. 5, Algorithm 1 and Secs. 4.4-4.5.)
5. **Fuse local and global evidence.** The final score is
   `0.8 * phoneme_score + 0.2 * speaker_score`. Higher scores mean greater consistency
   with the genuine speaker profile. Evaluate Equal Error Rate (EER, where false
   acceptance and rejection meet) and area under the receiver operating characteristic
   curve (ROC-AUC). Lower EER and higher AUC are better. (PDF pp. 5-6.)

The paper's Chinese and English benchmark evaluation does not by itself establish
German-to-English enrollment transfer for the same speaker. That requires a separate
bilingual protocol, with speaker identity and language controlled.

## Files for the first experiment

Paths below are relative to `original/PVP/`.

| File | Role |
| --- | --- |
| `2_PVP.py` | Main hybrid PVP entry point; enrollment, speaker GMM, fusion, and per-dataset evaluation. |
| `analyzer.py` | CTC decoding, frame pooling, phoneme and broad-class GMMs, reliability weights, and tiered scores. |
| `backbones/mms.py` | Default MMS-300M acoustic backbone and checkpoint loading. |
| `vocab.json` | Phoneme-to-broad-category mapping; required for the third scoring tier. |
| `utils.py` | EER calculation; AUC is computed with scikit-learn in `2_PVP.py`. |
| `protocols/zh_famous_enroll.csv`, `protocols/zh_famous_test.csv` | Candidate first benchmark's fixed enrollment and test lists. |
| `install.sh` | Environment reference only; review compatibility before installation. |

`2_PVP.py` also runs Famous Figures after ZH-Famous, using its two protocol CSVs.
It has no command-line option to select just one benchmark. A later small wrapper
must select a single run without changing the snapshot's scientific behavior.
Its relative paths assume the upstream directory is the working directory, with
`vocab.json`, `pretrained_models/`, and audio roots available there. Our later wrapper
must make paths explicit and route caches/outputs outside the source snapshot.

`data.py` is **not** imported by the main PVP path. It is relevant to the generic
backbone baseline, `1_backbone_baseline.py`. The other backbone wrappers are only
needed when selecting those models. Motivation and ablation scripts are not needed
for the first PVP run.

## Paper/code differences and ambiguities

Line references refer to the unchanged pinned source, not its README descriptions.
Do not silently change these settings or assume which version generated the tables.

| Topic | Paper | Inspected code and consequence |
| --- | --- | --- |
| Enrollment amount | Sec. 5.1, PDF p. 6: 1% of each speaker's available genuine speech. | Supplied protocols enroll 100-118 recordings per ZH speaker and 100-215 per English speaker. Many exceed 1%; details below. The main script uses the supplied lists without resampling. Exact published enrollment settings need clarification. |
| Phoneme GMM components | Sec. 4.2 mentions adapting to sample count; Sec. 5.1 says five components. | `analyzer.py:125-139,215-220`: skip phonemes with fewer than 5 occurrences; use one component for 5-9, otherwise five. Broad categories need at least 10 occurrences (`245-255`). Record these thresholds. |
| Global speaker GMM | Sec. 5.1: five components, covariance regularization `1e-3`. | `2_PVP.py:66-74`: five components only for more than 10 embeddings, otherwise one; regularization `1e-4`, `max_iter=200`, seed 42. The phoneme GMM uses `1e-3`. |
| Speaker sigmoid | Sec. 5.1 gives center `-2000` and scale `200`. | Phoneme scoring uses those values, but `2_PVP.py:45-51` uses center `-1800`, scale `100` for speaker likelihoods. This changes fused scores. |
| Reliability weights | Eq. 6 gives an exponential, without the code's extra rescaling. Its scale is not numerically specified in Sec. 5.1. | `analyzer.py:164-176,227-258`: exponential scale 1000, then separate min-max rescaling to `[0.1, 1]` for phonemes/categories. Ranking is preserved; weighted averages change. |
| Fallback trigger and averaging | Algorithm 1, PDF p. 5: any overlap with enrolled phonemes permits the second tier; its second/third tiers use an ordinary average. Nearby prose calls the second tier weighted. | `analyzer.py:270-318`: top tier needs at least one occurrence; second tier needs at least three occurrences and uses reliability weights; broad tier is weighted too. Repeated phonemes contribute repeatedly. Paper prose and algorithm are also inconsistent here. |
| Missing phonetic evidence | Paper does not specify a neutral numeric failure score. | `analyzer.py:261-263,310-318`: no features or no usable profile returns 0.5. The hybrid detector still combines this with speaker evidence. Empty profile-weight collections can also fail during fitting. Log failure/tier counts. |
| Alignment implementation | Paper calls this CTC-based alignment. | `analyzer.py:55-89` uses greedy phoneme decoding and character offsets, without a supplied transcript. It directly applies offset indices to backbone frames and increments `end_offset`. Phase 2's offline check confirms that Transformers 4.46.3 supports these offsets and uses an exclusive end index; the code adds a frame when one remains. On 2026-11-05 the loaded CTC model and MMS gave equal frame counts on 4-30 s synthetic input (`docs/environment.md`); other backbones and real speech are untested. No correction has been made. |
| Saved evaluation evidence | Scientific reproduction needs raw scores and exact trial accounting. | `2_PVP.py:143-194` catches prediction exceptions and continues; it returns aggregate/per-speaker metrics only. `save_results` writes these as JSON inside `.txt` files. Raw trial scores, failed-trial records, profiles, and environment are not exported by this entry point. |
| Main versus GMM ablation | Paper's full method uses GMMs. | `2_PVP.py` inherits the GMM detector in `analyzer.py`. Despite the upstream README's reversed description, `4_ablation_GMM.py` replaces both phoneme and speaker GMMs with cosine centroids; its hybrid `fit` defaults to top 8, rather than 12. Do not use it as the full-method entry point. |
| Generic baseline execution | Table 2 compares original backbones with PVP. | `1_backbone_baseline.py:21-34` imports an absent `antideepfake` package while wrappers are in `backbones/`. It also expects separate sample directories/labels and randomly crops audio to 32,000 samples (`MAX_LEN`; `data.py:24-29`). Its protocol/preprocessing is not automatically identical to the PVP run. |
| Other ablations | Table 4 covers removal of individual components. | `4_ablation_phoneme.py:15` also imports absent `antideepfake.mms`; several scripts select CUDA indices 2 or 6. `4_ablation_top_k.py` bypasses the salient tier. No dedicated speaker-branch-removal entry point was found; setting fusion alpha to 1 would require an explicitly recorded ablation configuration. |

Operational issues: `2_PVP.py` creates ECAPA at import time, selects `cuda:0`, and
does not create its `results/` directory. It skips speakers with fewer than five
enrollment paths, but test-profile lookup occurs outside its prediction exception
handler. `analyzer.py:90-92` prints feature-extraction errors and returns no features, which
then scores 0.5; `analyzer.py:113-117` silently uses an empty broad-category map if
`vocab.json` is missing from the working directory. These issues must be accounted
for in a later wrapper and checks.
They have not been fixed or tested through model execution in Phase 1.

## Dataset and protocol inventory

Public metadata checked on 2026-11-03; access approval and release contents are unverified.
No audio was downloaded.

| Dataset | Source and current access | Required mapping |
| --- | --- | --- |
| ZH-Famous, Mandarin | [PVP-ADD/ZH-Famous](https://huggingface.co/datasets/PVP-ADD/ZH-Famous); gated, requires accepting access conditions; page lists 64.3 GB and Parquet format. | Main script expects files below `zh_famous/wav/`; numeric protocol keys receive `.wav`. Map dataset audio to the exact original IDs; confirm release and schema after access. |
| Famous Figures / EN-Famous, English | [issf/famousfigures](https://huggingface.co/datasets/issf/famousfigures); gated, requests an affiliation email and manual review (page states 2-3 days); lists 170 GB and Parquet format. | Main script expects `FamousFigure/wav/`; supplied protocol keys already include `.wav`. Exact mapping from the release to author filenames remains unverified. |
| German/English bilingual data | Not selected. | Requires the same speakers across both languages, genuine enrollment, labeled genuine/spoof tests, and controlled splits. Separate Stage A evaluation; no substitute benchmark has been assumed. |

Dataset page sizes are storage indicators, not verified download/extracted disk sizes.
Do not accept access terms or download data on the user's behalf without authorization.

### ZH-Famous download record (2026-11-05)

User-approved download on the WSL server; `scientific_result=false` (data acquisition only).

| Item | Value |
| --- | --- |
| Repository | `PVP-ADD/ZH-Famous` (dataset), MIT licence, gated with automatic approval |
| Revision | `115a2a36297c734e275a5b7e8e43172256a12914` (last modified 2026-01-20) |
| Files | 23 Parquet shards `data/train-000NN-of-00023.parquet`, `README.md`, `.gitattributes` |
| Size | 64,337,515,426 bytes |
| Location | `data/zh-famous/` on the server (ignored by Git) |
| Command | `huggingface-cli download PVP-ADD/ZH-Famous --repo-type dataset --revision <revision> --local-dir data/zh-famous` (huggingface_hub 0.25.2) |
| Verification | `scripts/verify_dataset_download.py`: 25/25 files match Hub sizes; all 23 shards match published SHA-256. Report: `results/datasets/zh-famous-download.json` |

Dataset card schema: `file_name` (audio, 16 kHz), `label`, `person`, `method`,
`duration_sec`, `transcription`; one `train` split with 204,935 rows.

Shard schema confirmed with pyarrow 21.0.0: `file_name` is `struct<bytes: binary, path: string>`
(encoded audio plus original path, e.g. `1.wav`); shard 0 has 8,911 rows in 90 row groups.

### ZH-Famous protocol mapping check (2026-11-05)

`scripts/check_dataset_protocols.py` compares Parquet metadata (path, label, person, method;
audio not decoded) with the pinned upstream protocols. Protocol keys are mapped with
the upstream rule `<key>.wav` in one flat folder (`2_PVP.py:119-141`).
Report: `results/datasets/zh-famous-protocol-check.json`; `scientific_result=false`.

| Check | Result |
| --- | --- |
| Dataset rows / unique paths | 204,935 / 204,935; no duplicate paths, so the flat `wav/` layout is safe |
| Protocol rows / unique keys | 204,944 / 204,944 (1,023 enrollment + 203,921 test) |
| Enrollment/test overlap | none |
| Dataset files not in protocols | 0 |
| Speaker (`Speaker` vs `person`) | 0 mismatches |
| Label | all `bonafide` → `Real` (34,014), all `spoof` → `Fake` (170,921); no conflicts |
| Source vs `method` | identical names for all five spoof systems; bona fide empty in both |
| Protocol rows missing from release | **9**, all test spoof from LLaSA |

Missing rows: HuaYu `95041`, `95093`, `95100`, `95135`, `95169`, `95197`, `95379`, `95398`;
QiangdongLiu `109001`. All enrollment and bona fide test rows are present. The missing
rows are 9 of 170,930 spoof test trials (0.005%). Whether the authors' published run
included them is unknown.

Consequence: unchanged upstream `2_PVP.py:143-147` would catch the file error, print it,
and silently skip these trials. Our runner must keep the full protocol, record these 9
as failed trials with reason `missing_from_release`, and report metrics with the actual
trial count. Do not edit the protocol files.

The shipped CSV header is `Speaker,AudioName,Source,Label`, although the upstream README
documents `speaker,path,source,label`. The main script skips the header and assigns its
own lowercase column names by position. Preserve column order; do not rename source files.

Counts below were computed from the full pinned CSVs with Python's standard `csv` module.

| Benchmark | Speakers | Enrollment, all genuine | Test genuine | Test spoof | Total test |
| --- | ---: | ---: | ---: | ---: | ---: |
| ZH-Famous | 10 | 1,023 | 32,991 | 170,930 | 203,921 |
| Famous Figures | 10 | 1,115 | 49,151 | 297,870 | 347,021 |

Relative to all genuine enrollment plus test rows, enrollment is 3.01% for ZH-Famous
and 2.22% for Famous Figures. Individual shares range from about 0.99% to 32.47%
and 1.00% to 20.88%, respectively. This is inconsistent with a uniform 1% reading of
Sec. 5.1. The origin of the apparent enrollment floor is not documented in the inspected code.

ZH enrollment plus test counts match Table 1's 204,944 total, 34,014 genuine,
and 170,930 spoof utterances (PDF p. 3). The paper's descriptions of balance should
not be read as equal genuine/spoof class counts. Five ZH spoof sources appear:
LLaSA, OpenAudio-S1-mini, VOXCPM, F5, and indextts. The English test CSV has 11 spoof sources.

No exact duplicate full rows or exact `(speaker, path)` enrollment/test overlaps were
found in either benchmark. This checks metadata only. It does **not** establish audio
integrity or rule out shared source recordings, overlapping segments, alternate filenames,
or spoof-generation reference leakage. Those checks require the actual dataset and its provenance.

## Published comparison targets, not our results

Table 2, PDF p. 6 / printed p. 5996; both metrics below are percentages.

| MMS-300M setting | ZH-Famous EER | ZH-Famous AUC | EN-Famous EER | EN-Famous AUC |
| --- | ---: | ---: | ---: | ---: |
| Original backbone baseline | 21.13 | 84.11 | 13.97 | 93.99 |
| With PVP | 11.37 | 94.58 | 7.24 | 96.61 |

No reproduced or German/English evaluation results exist yet. The default code makes
MMS-300M on ZH-Famous a suitable candidate first run, subject to dataset access and
documented resolution of the differences above. An upstream-code run and a paper-setting
run must be labeled separately if they use different settings; do not tune toward table values.

`utils.py:44-51` computes EER as the false-positive rate at the sampled ROC point
nearest the false-negative rate. It does not interpolate an exact crossing.
Use and record this definition for a code-faithful comparison. `bonafide` is positive
label 1 and `spoof` is 0. Code AUC and EER are stored as fractions; printed EER is
multiplied by 100, while printed AUC remains a fraction. Convert consistently for the paper.
The main script reports pooled trials within each dataset plus per-speaker metrics,
but does not compute the paper's cross-dataset `Pooled` column. Its aggregation rule
must be established before reproducing that column.

## Phase 1 verification record

- Read all nine PDF pages; visually inspected architecture, Algorithm 1, parameters,
  and Table 2 on PDF pages 4-6.
- Inspected the local repository, its remote URL and initial commit; fetched the
  connected GitHub README and confirmed its original title matched the local file.
- Verified all 23 exported upstream file contents against their pinned Git blob IDs;
  retained SHA-256 hashes and sizes in the manifest.
- Counted full protocol metadata, labels, speakers, duplicate rows, and exact split-key overlaps.
- Checked all 14 upstream Python files with `ast.parse`, without imports or execution.
- Verified the unchanged paper hash, exact `CLAUDE.md` import, authored relative links,
  Git whitespace checks, and ignore rules for representative runtime artifacts.
- No detector, package installation, CUDA check, model inference, or benchmark was run.
  Phase 2 environment compatibility and Phase 3 audio integrity remain unverified.
