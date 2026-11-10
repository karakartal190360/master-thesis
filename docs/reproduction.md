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
| Famous Figures / EN-Famous, English | [issf/famousfigures](https://huggingface.co/datasets/issf/famousfigures); gated, requests an affiliation email and manual review (page states 2-3 days); lists 170 GB and Parquet format. | Main script expects `FamousFigure/wav/`; supplied protocol keys already include `.wav`. Release paths repeat across systems; mapped one-to-one to protocol keys with an inferred suffix rule (see "Famous Figures protocol mapping check"). |
| German/English bilingual data | Not selected. | Requires the same speakers across both languages, genuine enrollment, labeled genuine/spoof tests, and controlled splits. Separate Stage A evaluation; no substitute benchmark has been assumed. |

Dataset page sizes are storage indicators, not verified download/extracted disk sizes.
Do not accept access terms or download data on the user's behalf without authorization.

### Famous Figures access check (2026-11-06)

The user reported granted access. Confirmed with Hub metadata and HTTP HEAD requests only;
no Parquet data was downloaded.

| Item | Value |
| --- | --- |
| Repository | `issf/famousfigures` (dataset), CC BY 4.0, gated with manual review |
| Revision | `51f591988c68c6fe0922df621aca4b174c4942cb` (last modified 2025-09-20) |
| Files | 557 (Parquet shards per speaker, e.g. `data/Anthony_Blinken-00000-of-00032.parquet`, plus `README.md`, `.gitattributes`) |
| Size | 170,444,698,180 bytes |
| Access | HEAD on `README.md` and one shard succeeded with the server's token |
| Server disk | 723 GB free before download |

### Famous Figures download record (2026-11-06)

User-approved download on the WSL server; `scientific_result=false` (data acquisition only).

| Item | Value |
| --- | --- |
| Revision | `51f591988c68c6fe0922df621aca4b174c4942cb` |
| Files | 555 Parquet shards `data/<Speaker>-000NN-of-000MM.parquet` (one card split per speaker), `README.md`, `.gitattributes` |
| Size | 170,444,698,180 bytes |
| Location | `data/famous-figures/` on the server (ignored by Git); log `data/famous-figures-download.log` |
| Command | `huggingface-cli download issf/famousfigures --repo-type dataset --revision <revision> --local-dir data/famous-figures` (huggingface_hub 0.25.2) |
| Verification | `scripts/verify_dataset_download.py`: 557/557 files match Hub sizes; all 555 shards match published SHA-256. Report: `results/datasets/famous-figures-download.json` |

Dataset card schema: `Audio`, `Label`, `Source`, `Speaker`, `AudioName`; ten speaker
splits with 348,136 examples in total. This equals the protocol total
(1,115 enrollment + 347,021 test). Row-level mapping: see the next section.

Download problems, all resolved before the final verification:

- WSL shut down twice when the Claude session ended (11:05 and 11:20), killing the
  download. It was then started through a hidden Windows `wsl.exe` process, which keeps WSL running.
- After an interruption, huggingface_hub 0.25.2 crashed once with `UnboundLocalError`
  while reading a damaged resume metadata file. The library deletes that file first, so a rerun succeeds.
- The first verification found 3 shards truncated to 436,207,616 bytes
  (`Barack_Obama-00110`, `-00112`, `-00114`), written just before the 11:05 shutdown.
  They were deleted, re-downloaded, and the full verification was repeated.

Additional checks after the final verification (one-off script, not kept):
no `.incomplete` files remain; files on disk equal the Hub file list; shard numbering
is complete for all 10 speakers; all 555 Parquet footers open with pyarrow 21.0.0 and
share one schema; row counts equal the card's `num_examples` for every speaker.
No WSL restart occurred after the final download attempt; data was flushed with `sync`.

### Famous Figures protocol mapping check (2026-11-06)

Shard schema confirmed with pyarrow 21.0.0: `Audio` is `struct<bytes: binary, path: string>`;
`Label`, `Source`, `Speaker`, `AudioName` are strings; there is no duration column.
`scripts/check_dataset_protocols.py --dataset famous-figures` reads metadata only (audio not
decoded). Column names per release are in `scripts/parquet_datasets.py`.
Report: `results/datasets/famous-figures-protocol-check.json`; `scientific_result=false`.

Matching by path alone, as for ZH-Famous, fails: the release has 348,136 rows but only
258,204 unique `Audio.path` values (31,613 paths occur 2-8 times), and 89,932 protocol keys
do not exist as release paths. A flat `wav/` folder named by release path would overwrite files.
Example: `Anthony_Blinken_00001.wav` is the release path of a bona fide row and of spoof rows
from other systems. Protocol keys for these rows add the source system to the stem:
`Anthony_Blinken_00766_-.wav` (bona fide, Source `-`), `..._PARLER.wav`, `..._XTTSV2.wav`.

Matching rule (our inference from the data, not documented upstream or on the card):
a protocol row matches the release row with the same speaker, the same source, and the path
equal to the protocol key; if none exists, the key with its final `_<Source>` removed
(`<stem>_<Source>.wav` → `<stem>.wav`).

| Check | Result |
| --- | --- |
| Dataset rows / unique paths | 348,136 / 258,204 |
| Protocol rows / unique keys | 348,136 / 348,136 (1,115 enrollment + 347,021 test) |
| Protocol keys absent from release paths | 89,932 |
| Matched, key equals path | 258,204 |
| Matched after removing `_<Source>` | 89,932: bona fide 24,776, XTTSV2 24,793, STYLETTS2 21,338, PARLER 15,047, F5TTS 1,243, E2TTS 1,001, FISHSPEECH 1,000, MASKGCT 391, SSRSPEECH 343 |
| Unmatched / ambiguous protocol rows | 0 / 0 |
| Release rows matched twice / unused | 0 / 0 (one-to-one) |
| Enrollment/test overlap | none |
| Speaker, source | equal by construction of the rule; shard speaker prefix equals `Speaker` for all rows |
| Label | `bonafide` → `bonafide` 50,266, `spoof` → `spoof` 297,870; no conflicts |
| `AudioName` | always equals `Audio.path` without `.wav` |

The rule gives an exact one-to-one mapping with no ambiguity, which supports it, but it is
still an assumption: the authors' local file names and how they created the protocols are
not published. All 12 source systems (bona fide plus 11 spoof systems) are present. Unlike
ZH-Famous, no protocol row is missing from this release.

The map from protocol key to shard and row is `results/datasets/famous-figures-protocol-map.csv`
(columns `file, split, speaker, source, label, shard, row, release_path, rule`); its SHA-256
is in the report. The extraction names files by protocol key, so unchanged upstream code
finds them below `FamousFigure/wav/`. Do not edit the protocol files.

### Famous Figures audio extraction (2026-11-06)

`scripts/extract_parquet_audio.py --dataset famous-figures --map <protocol map>` wrote each
Parquet audio entry unchanged to `data/famous-figures/wav/<protocol key>` (flat; no
re-encoding, no resampling) and decoded every file once with soundfile 0.12.1 (libsndfile
1.2.0). There is no duration column, so durations are not cross-checked. Started through a
hidden Windows `wsl.exe` process (launcher `results/datasets/famous-figures-extract.sh`,
log `data/famous-figures-extraction.log`), followed by `sync`. Run on an uncommitted
checkout (`git_dirty=true`, script SHA-256 values in the report); `scientific_result=false`.

| Item | Result |
| --- | --- |
| Files | 348,136 WAV (one per protocol row; 0 not in map, 0 name conflicts); 179.78 GB; 761.5 h |
| Decoding | all files decode and are non-empty |
| Bit identity | written bytes equal Parquet bytes; after the run, `sha256sum -c` of all files on disk against the manifest passed |
| Encoding | PCM 16-bit 295,206; 32-bit float 52,930 (COZYVOICE2 all, FISHSPEECH 6,371 of 7,371, SSRSPEECH 2) |
| Channels | 13,399 stereo files, all bona fide: Barack_Obama 10,430 and Kamala_Harris 2,969 (170 of them in enrollment); all others mono |
| Report status | `failed` only because of the mono check (13,399 × `channels=2`); no other problem |
| Evidence | `results/datasets/famous-figures-extraction.json`, `results/datasets/famous-figures-wav-manifest.csv` |

The two channels of every stereo file differ (not duplicated mono). Upstream loads with
`librosa.load(path, sr=16000)` and the default `mono=True`, which averages the channels before
resampling. These files are therefore usable by unchanged upstream code; they are a dataset
property, not failed trials. Keep them unchanged on disk.

Stored sample rates by source system:

| Source | Sample rate | Files |
| --- | ---: | ---: |
| Bona fide (`-`) | 16,000 / 22,050 | 27,208 / 23,058 |
| LLASA, SSRSPEECH | 16,000 | 14,356, 6,747 |
| COZYVOICE2, E2TTS, F5TTS, MASKGCT, STYLETTS2, XTTSV2 | 24,000 | 46,557, 33,633, 32,590, 20,348, 33,318, 31,924 |
| FISHSPEECH, PARLER, ZONOS | 44,100 | 7,371, 22,604, 48,422 |

Bona fide rate per speaker: 16 kHz mono for most speakers; 22.05 kHz for Joe_Biden (all),
JD_Vance (2,003 of 2,821), and the stereo Barack_Obama and Kamala_Harris files. As for
ZH-Famous, sample rate, channel count and sample format are closely tied to the source
system, and for some speakers channel count is tied to the label. Upstream's resampling and
downmixing remove most of this, but note it before interpreting per-system results.

Runtime layout: upstream reads `./FamousFigure/wav` relative to its working directory;
`scripts/link_pretrained.py` links it to `data/famous-figures/wav` (see "Runtime layout check").

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

`scripts/check_dataset_protocols.py --dataset zh-famous` compares Parquet metadata (path, label, person, method;
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

On 2026-11-06 the scripts were generalized for Famous Figures. The new version
(`--dataset zh-famous`) reproduced the same counts, label pairs and 9 missing rows, and
one re-extracted shard (8,911 files) had the same SHA-256 values as the manifest. These were
scratch runs; the ZH-Famous reports above were not replaced.

### ZH-Famous audio extraction (2026-11-05)

`scripts/extract_parquet_audio.py` wrote each Parquet audio entry unchanged to
`data/zh-famous/wav/<path>` (flat; no re-encoding, no resampling) and decoded every file
once with soundfile to check channels, frames and duration against `duration_sec`
(tolerance 1 ms). `scientific_result=false`.

| Item | Result |
| --- | --- |
| Files | 204,935 WAV, mono, PCM 16-bit; 66.25 GB; 412.5 h |
| Checks | 0 failures: all decode, mono, non-empty, duration matches |
| Bit identity | written bytes equal Parquet bytes; SHA-256 per file in the manifest |
| Evidence | `results/datasets/zh-famous-extraction.json`, `results/datasets/zh-famous-wav-manifest.csv` |

Stored sample rates differ by system. The card's 16 kHz is the `datasets` decoding
rate, not the storage rate:

| Source | Sample rate | Files |
| --- | ---: | ---: |
| Bona fide | 16,000 | 34,014 |
| LLaSA | 16,000 | 34,513 |
| VOXCPM | 16,000 | 34,156 |
| F5 | 24,000 | 34,069 |
| indextts | 22,050 | 34,049 |
| OpenAudio-S1-mini | 44,100 | 34,134 |

Upstream loads all audio with `librosa.load(path, sr=16000)` (`analyzer.py:57`,
`2_PVP.py:40`), so files are resampled to 16 kHz at load time. Keep native rates on disk
to match this behaviour. Resampler: librosa 0.10.2.post1 default `res_type="soxr_hq"`,
soxr 1.1.0; the authors' librosa version, and therefore their resampler, is not recorded
upstream. Sample rate is perfectly tied to the source system for three spoof systems;
note this as a dataset property before interpreting per-system results.

Runtime layout: upstream reads `./zh_famous/wav` relative to its working directory;
`scripts/link_pretrained.py` links it to `data/zh-famous/wav` (see "Runtime layout check").

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
The audio-based checks on 2026-11-08 found exact copies across enrollment and test; see
"Audio integrity and leakage checks".

### Runtime layout check (2026-11-08)

Setup check on the WSL server, `scientific_result=false`; run before committing the
`link_data` change to `scripts/link_pretrained.py`. The script now also links the data
that `2_PVP.py:190-200` reads relative to its working directory:

| Upstream lookup | Link target |
| --- | --- |
| `./protocols/` | `original/PVP/protocols/` (unchanged upstream CSVs) |
| `./zh_famous/wav/` | `data/zh-famous/wav/` |
| `./FamousFigure/wav/` | `data/famous-figures/wav/` |

```bash
.venv/bin/python scripts/link_pretrained.py results/runtime
```

A second run made no changes. From `results/runtime`, every protocol row was read and
resolved with the upstream rule (`pd.read_csv(..., names=[...], header=0)`, append `.wav`
unless present, `os.path.join(audio_root, key)`, `2_PVP.py:110-141`), then checked with
`os.path.isfile`:

| Benchmark | Enrollment rows found | Test rows found | Unique paths / files in root |
| --- | --- | --- | --- |
| ZH-Famous | 1,023 / 1,023 | 203,912 / 203,921 | 204,944 / 204,935 |
| Famous Figures | 1,115 / 1,115 | 347,021 / 347,021 | 348,136 / 348,136 |

The 9 unresolved ZH-Famous paths are exactly the 9 LLaSA test rows missing from the
release (see "ZH-Famous protocol mapping check"). No WAV file in either root is unused.
This check confirms file presence only; decoding, duration and leakage checks follow.

`save_results` (`2_PVP.py:183-185`) opens `results/<benchmark>_<backbone>.txt` in the
working directory and does not create `results/`. The link script does not create it
either: output location is a decision for the future experiment wrapper.

### Audio integrity and leakage checks (2026-11-08)

Setup checks on the WSL server, `scientific_result=false`, run on an uncommitted checkout
(script SHA-256 values in the reports). Nothing was removed or changed; protocols stay as
published. Thresholds are our descriptive choices; upstream defines none.

```bash
.venv/bin/python scripts/check_dataset_audio.py --audio-root data/<dataset>/wav \
  --enroll original/PVP/protocols/<benchmark>_enroll.csv \
  --test original/PVP/protocols/<benchmark>_test.csv \
  --output results/datasets/<dataset>-audio-check.csv \
  --report results/datasets/<dataset>-audio-check.json --workers 12
.venv/bin/python scripts/check_dataset_leakage.py --dataset <dataset> \
  --audio-check results/datasets/<dataset>-audio-check.csv \
  --manifest results/datasets/<dataset>-wav-manifest.csv \
  [--parquet-dir data/zh-famous/data] \
  --groups results/datasets/<dataset>-identical-audio.csv \
  --report results/datasets/<dataset>-leakage.json
```

**Integrity.** `check_dataset_audio.py` decodes every protocol file with soundfile 0.12.1
(float32, all channels) and measures level on the channel mean, as upstream's
`librosa.load(mono=True)` does. It also hashes the decoded samples (sample rate, shape and
values), so identical audio is found even in a different container.

| Check | ZH-Famous | Famous Figures |
| --- | --- | --- |
| Decoded / protocol rows | 204,935 / 204,944 (9 known missing LLaSA files) | 348,136 / 348,136 |
| Decode errors, empty, non-finite, digital silence, above full scale | 0 | 0 |
| Shorter than 0.5 s | 2 (LLaSA) | 3,099 (LLASA 3,028) |
| Shorter than 25 ms (400 samples at 16 kHz) | 1 (LLaSA) | 322 (LLASA 320, ZONOS 2) |
| Longer than 30 s | 4,106 (enrollment 10, test bona fide 389, spoof 3,707); max 50.0 s | 446 (all test spoof; LLASA 329, SSRSPEECH 110); max 79.5 s |
| Mean level below -60 dBFS | 362 (LLaSA 356, VOXCPM 6) | 1,888 (LLASA 1,863) |
| More than 0.1% of samples at or above 0.999 | 6,800 (F5 4,856; enrollment 88) | 71 (test bona fide 69) |

Duration summaries per source are in the JSON reports. Many very short or very quiet LLaSA
outputs in both benchmarks look like failed generations. They remain valid spoof trials of
the published protocol.

Consequences for Phase 4: (1) files shorter than 400 samples at 16 kHz are probably too
short for the wav2vec2-style feature encoder; upstream would print the error and skip the
trial (`2_PVP.py:143-147`), so our runner must record such trials as failed, with reason.
Check this on these files in the small end-to-end run. (2) The GPU memory check covered up
to 30 s, but files reach 50 s (ZH) and 79.5 s (Famous Figures). Check peak memory on the
longest file before a full run.

**Identical audio.** `check_dataset_leakage.py` groups files by decoded-sample hash. In both
benchmarks every group is also byte-identical (same file SHA-256). No group mixes labels.

| Groups (files) | ZH-Famous | Famous Figures |
| --- | --- | --- |
| All | 6,259 (12,716) | 1,112 (2,418) |
| Enrollment file = test bona fide file, same speaker | **60** (DengFan) | **34** (Barack_Obama 24, Tim_Walz 9, Joe_Biden 1) |
| Within enrollment | 0 | 1 (`Tim_Walz_00377.wav` = `Tim_Walz_00378.wav`) |
| Within test, bona fide | 3,066 (DengFan) | 1,000 (mostly Barack_Obama) |
| Within test, spoof | 3,133 (all OpenAudio-S1-mini; sizes up to 92) | 77 (LLASA 69, SSRSPEECH 8; sizes up to 77) |
| Several speakers | 0 | 9, all 20 ms near-silent LLASA fragments |

ZH-Famous DengFan: 3,126 of 10,585 bona fide clips are exact copies of other DengFan clips.
The later IDs (6,047-10,585) repeat earlier IDs (1-4,428) at four fixed ID offsets, so part
of the source material seems to have been included twice. The 60 enrollment/test copies are
0.18% of ZH bona fide test trials; the 34 are 0.07% for Famous Figures. These test trials
contain exactly an enrollment utterance, so their genuine scores may be too high. The
enrollment duplicate gives one Tim_Walz utterance double weight in the speaker profile.

**Shared transcripts (ZH-Famous).** The release has a `transcription` column. Exact
same-speaker matches with an enrollment transcript: 136 bona fide test trials (including
the 60 copies) and 1,155-1,159 spoof test trials per system. Each spoof system
re-synthesizes the bona fide texts, so test spoofs often say the same sentence as an
enrollment utterance. This is a design property of the benchmark, not a file error.

**Voice prompts (Famous Figures).** Many spoof file names end with the bona fide clip used
as the voice prompt, e.g. `..._ZONOS__Barack_Obama_06562.wav`; LLASA names give only the
number. This reading of the names is our inference. Of 297,870 test spoofs, 186,554 name a
prompt and 111,316 do not (MASKGCT, PARLER, STYLETTS2, most XTTSV2, few others). Every speaker
has exactly one prompt clip, and all ten are bona fide **test** files, not enrollment files.
So the names show no enrollment-prompt leakage. A single test bona fide trial per speaker
is the voice source for up to 73,720 spoofs (Barack_Obama).

**Not checked.** Overlapping or partly shared segments, the same recording at another rate
or encoding, and shared recording sessions between enrollment and test (consecutive ZH IDs
suggest continuous sessions). These need audio fingerprinting or provenance data.

**Handling.** The first baseline uses the official protocols unchanged, as required. Report
these findings with the results. Any analysis without the leaked trials is a separate,
clearly labelled additional analysis, decided before looking at test scores.

| Evidence | Files in `results/datasets/` |
| --- | --- |
| ZH-Famous | `zh-famous-audio-check.{csv,json,log}`, `zh-famous-identical-audio.csv`, `zh-famous-leakage.json` |
| Famous Figures | `famous-figures-audio-check.{csv,json,log}`, `famous-figures-identical-audio.csv`, `famous-figures-leakage.json` |

## PVP runner (Phase 4)

[run_pvp.py](../scripts/run_pvp.py) runs one benchmark with the unchanged upstream code.
Upstream's `run_poi_specificity_experiment` returns only aggregate metrics and drops failed
trials, so the runner repeats its loop (`2_PVP.py:103-153`) with the same upstream objects:
`UniversalPhonemeAnalyzer`, `HybridPOIDetector.fit` and `.predict`, the same protocol reading
and `.wav` path rule, and the same "at least 5 enrollment files" rule. Additions only record:

- every test trial gets a row in `scores.csv`: score, phoneme score, scoring tier, or a
  failure reason. A trial is failed when `predict` raises (upstream prints and skips it) or
  the speaker has no profile (upstream would crash with `KeyError`);
- the tier text from `predict` shows upstream's silent fallbacks, e.g. `Extract Fail + ASV`
  (no phoneme features, phoneme score 0.5, speaker score still used);
- files without phoneme features are observed by wrapping the analyzer method without
  changing its result;
- fitted profiles (phoneme GMMs, weights, top-k set, broad-class GMMs, speaker GMM) are
  saved per speaker in `profiles/`.

Settings decision: runs use the **upstream code settings** at the pinned commit, not the
paper text (differences in "Paper/code differences and ambiguities"). They are listed with
their source lines in [pvp-zh-famous-mms.json](../configs/pvp-zh-famous-mms.json): MMS,
`cuda:0`, alpha 0.8, top-k 12, minimum 5 enrollment files. All other values (GMM settings,
`random_state=42`, sigmoid centers and scales, reliability weights) are fixed in upstream code.
A run with paper settings would be a separate, labelled configuration.

[evaluate_pvp.py](../scripts/evaluate_pvp.py) computes the metrics from `scores.csv` with
upstream `utils.compute_eer` and `roc_auc_score`, pooled and per speaker, as
`2_PVP.py:160-176`. Failed trials have no score and are excluded from the metrics, as
upstream does, but they are counted and listed. The runner calls the same function, so a
saved run can be re-evaluated later without the models.
[test_evaluate_pvp.py](../tests/test_evaluate_pvp.py) checks the evaluator on toy scores:
equality with the upstream functions, score direction, failed-trial accounting and the
percent conversion for the paper comparison.

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python scripts/run_pvp.py configs/pvp-zh-famous-mms.json --scientific
.venv/bin/python scripts/evaluate_pvp.py results/runs/<run_id>/scores.csv --config configs/pvp-zh-famous-mms.json
```

Each run gets a new ID (`<UTC time>-<8 hex>`) and folder `results/runs/<run_id>/`:
`run.json` (status, command, Git state incl. untracked files, upstream commit, config,
protocol and weight SHA-256, Python, platform, GPU and driver, CUDA, cuDNN settings, package
versions, peak GPU memory), `config.resolved.json`, `enrollment.json`, `scores.csv`,
`metrics.json` (with paper comparison) and `profiles/`. The W&B run uses the same ID and
uploads these files as a `reproducibility` artifact. `--scientific` refuses a dirty
checkout or a protocol override; every other run is `scientific_result=false`.

### Small end-to-end check (2026-11-10)

Setup check, `scientific_result=false` (dirty checkout, subset protocol). Full ZH-Famous
enrollment (10 speakers, 1,023 files) and a 210-trial test subset
`results/checks/zh_famous_test_smoke.csv`: the first 10 bona fide and 2 spoof trials per
speaker and system, plus hard cases from the Phase 3 checks (the two shortest files, the
three longest, two low-level files, one missing file, two copies of enrollment files).
W&B run [`20261110T160411Z-47465f87`](https://wandb.ai/umuttanriverdi/personalized-speech-deepfake-detection/runs/20261110T160411Z-47465f87).

| Item | Result |
| --- | --- |
| Model loading | offline from linked files; no Hub access |
| Enrollment | all 10 speakers; 7-15 s each; 70-81 phoneme GMMs, 12 top-k phonemes, 5-component speaker GMM; no enrollment file without phoneme features |
| Trials | 210: 208 scored, 2 failed |
| Failed | `95041.wav` missing from release; `95084.wav` (20 ms LLaSA) raises `RuntimeError` in ECAPA (padding larger than input). Upstream would print and skip both. |
| Fallbacks | 5 LLaSA spoofs (short or near-silent) `Extract Fail + ASV`: phoneme score 0.5, fused about 0.60. Tiers: Level 1 139, Level 2 63, Level 3 1. |
| Peak GPU memory | 3,802 MiB allocated, including the 50 s files (longest ZH-Famous files) |
| Speed | 13.3 test trials/s; full ZH-Famous (203,921 trials) estimated at about 4.5 h |
| Replay | second run `20261110T160708Z-f4cdd955` (W&B disabled): all scores, tiers and failures bit-identical |
| Standalone evaluation | same metrics as the runner |

Scores are strongly saturated: the median bona fide score and the 90th spoof percentile are
both about 0.99999, and the EER threshold is 0.9999915. The saved full-precision `repr`
values keep all 208 scores distinct, so this does not create ties here. This comes from the
upstream sigmoid settings; keep full precision in all score files.

The subset EER (15.09%) and AUC (92.72%) are **not** results: the subset is small, not
representative, and deliberately contains hard cases.

GPU driver: the run record shows driver 621.14; Phase 2 verified 610.88. The Windows driver
was updated in between. The full run records its own driver version.

### Full ZH-Famous MMS run (2026-11-10)

Scientific run `20261110T161146Z-a7e7dcab` (`scientific_result=true`), clean checkout at
`f806ba8`, upstream `1eed3f4`, config [pvp-zh-famous-mms.json](../configs/pvp-zh-famous-mms.json)
(upstream code settings), full official enrollment and test protocols. W&B run
[`20261110T161146Z-a7e7dcab`](https://wandb.ai/umuttanriverdi/personalized-speech-deepfake-detection/runs/20261110T161146Z-a7e7dcab)
with the `reproducibility` artifact; local record `results/runs/20261110T161146Z-a7e7dcab/`
on the server. Command: `.venv/bin/python scripts/run_pvp.py configs/pvp-zh-famous-mms.json --scientific --wandb online`.

| Item | Result |
| --- | --- |
| Environment | WSL2 Ubuntu, RTX 4070 Ti SUPER (driver 621.14), Python 3.9.25, PyTorch CUDA 12.4, cuDNN 9.1, `cudnn.deterministic=False` |
| Run time | 16:11-19:23 UTC (3 h 12 min); 17.9 test trials/s on average |
| Enrollment | all 10 speakers (100-118 files each); 70-81 phoneme GMMs, 5-component speaker GMM; no enrollment file without phoneme features |
| Trials | 203,921: 203,911 scored (32,991 bona fide, 170,920 spoof), 10 failed |
| Failed | 9 LLaSA files missing from the release (the 9 known from the protocol mapping); `95084.wav` (20 ms LLaSA) `RuntimeError` in ECAPA |
| Tiers | Level 1 140,007; Level 2 62,559; Level 3 285; `Extract Fail + ASV` 1,060 (1,056 LLaSA, 4 VOXCPM; all spoof) |
| Peak GPU memory | 3,802 MiB allocated |
| Ties | 6,199 score values occur more than once (mostly identical DengFan clips); none mixes bona fide and spoof, so the metrics are not affected |

**Our result compared with the paper** (Table 2, MMS-300M with PVP, ZH-Famous; percent):

| Source | EER | ROC-AUC |
| --- | ---: | ---: |
| Paper-reported | 11.37 | 94.58 |
| Our run (upstream code settings) | 12.24 | 93.62 |
| Difference (ours minus paper) | +0.87 | -0.96 |

Metric: upstream `utils.compute_eer` and `roc_auc_score`, pooled over all scored trials of the
dataset, higher score = bona fide (as `2_PVP.py:160-176`). The result is close to the paper but
not equal: EER is 0.87 points higher and AUC 0.96 points lower. This is an approximate
reproduction, not an exact one. The cause of the gap is not established. Possible causes are the
documented settings differences (enrollment amount, speaker sigmoid, GMM regularization;
"Paper/code differences and ambiguities"), the 9 missing release files, an unknown code or
environment state behind the published table, and non-deterministic GPU kernels. We did not
tune anything toward the table values.

Per speaker (EER / AUC, percent): DengFan 13.67 / 93.28, HongyiZhou 31.97 / 72.21,
HuaYu 4.36 / 99.10, JunLei 12.47 / 94.32, MinghongYu 12.12 / 92.72, QiangdongLiu 8.13 / 97.28,
XiangLuo 3.99 / 98.81, YanMo 10.64 / 96.34, YonghaoLuo 4.64 / 98.47, YunMa 11.36 / 92.45.
HongyiZhou is a clear outlier; the reason is not investigated yet.

Additional analyses of the same scores (not paper comparisons; the main result above is unchanged):

| Analysis | EER | AUC |
| --- | ---: | ---: |
| All bona fide vs F5 only | 8.48 | 97.27 |
| All bona fide vs LLaSA only | 10.18 | 94.81 |
| All bona fide vs indextts only | 11.47 | 93.34 |
| All bona fide vs OpenAudio-S1-mini only | 13.39 | 93.19 |
| All bona fide vs VOXCPM only | 16.79 | 89.47 |
| Without the 60 bona fide test copies of enrollment files | 12.26 | 93.61 |
| 10 failed spoof trials counted as accepted (worst case) | 12.24 | 93.61 |

The enrollment/test copies and the failed trials change the result by at most 0.02 points.
All 1,060 fallback trials (phoneme score 0.5, fused 0.576-0.600) are spoofs scored far below
the EER threshold (0.99999), so upstream's fallback rejects them correctly here.
Replay of the full run has not been done; the smoke-check replay was bit-identical.

### Famous Figures small end-to-end check (2026-11-10)

Setup check, `scientific_result=false` (dirty checkout, subset protocol), config
[pvp-famous-figures-mms.json](../configs/pvp-famous-figures-mms.json): same upstream code
settings as ZH-Famous; only protocols, audio root and paper target differ (second benchmark of
`2_PVP.py:196-200`). Full enrollment (10 speakers, 1,115 files) and a 194-trial test subset
`results/checks/famous_figures_test_smoke.csv`: the first 8 bona fide and 1 spoof trial per
speaker and system, plus hard cases from the Phase 3 checks (the three longest files incl. two
79.5 s files, the two longest stereo files, the three shortest files, two low-level files, one
clipping file, two copies of enrollment files).
W&B run [`20261110T193420Z-ec42b331`](https://wandb.ai/umuttanriverdi/personalized-speech-deepfake-detection/runs/20261110T193420Z-ec42b331).

| Item | Result |
| --- | --- |
| Enrollment | all 10 speakers (100-215 files); 6-20 s each; 42-51 phoneme GMMs (ZH-Famous: 70-81), 5-component speaker GMM; no enrollment file without phoneme features |
| Trials | 194: 191 scored, 3 failed |
| Failed | the three shortest files (10-12 ms; 2 ZONOS, 1 LLASA): no phoneme features, then `RuntimeError` in ECAPA (padding larger than input). Upstream would print and skip them. |
| Fallbacks | 6 LLASA spoofs `Extract Fail + ASV` (fused 0.58-0.60); all other trials Level 1 |
| Peak GPU memory | 4,676 MiB allocated, including the 79.5 s files (longest Famous Figures files) |
| Stereo files | scored normally (upstream downmix on load) |
| Speed | 15.5 test trials/s; full test protocol (347,021 trials) estimated at about 5.5-6.5 h |
| Replay | second run `20261110T193658Z-38b6699d` (W&B disabled): `scores.csv` and metrics bit-identical |
| Standalone evaluation | same metrics as the runner |

The subset EER (6.60%) and AUC (98.45%) are **not** results (small, non-representative subset
with deliberate hard cases). The full release has 322 files under 25 ms, so a few hundred
failed trials are expected in the full run; each will be recorded with its reason.

## Published comparison targets, not our results

Table 2, PDF p. 6 / printed p. 5996; both metrics below are percentages.

| MMS-300M setting | ZH-Famous EER | ZH-Famous AUC | EN-Famous EER | EN-Famous AUC |
| --- | ---: | ---: | ---: | ---: |
| Original backbone baseline | 21.13 | 84.11 | 13.97 | 93.99 |
| With PVP | 11.37 | 94.58 | 7.24 | 96.61 |

Our first reproduced result (ZH-Famous, MMS-300M with PVP, upstream code settings) is in
"Full ZH-Famous MMS run (2026-11-10)" above. No German/English evaluation results exist yet. The default code makes
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
