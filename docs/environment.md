# Environment setup

Status on 2026-11-05: the macOS development environment passes offline checks with
the CPU thread limits below. The home server is reachable over SSH, and Ubuntu
WSL2 can see the NVIDIA GPU. The Linux installation, pinned fairseq build, and
offline CUDA checks pass. W&B logging is verified in a private project; see
[tracking setup](tracking.md). The three pinned models load offline through unchanged
upstream code and run together on the GPU; see the
[model loading check](#model-loading-and-inference-check-2026-11-05). Phase status lives in
[AGENTS.md](../AGENTS.md).

## Machines and responsibilities

- **macOS:** MacBook Pro M5 Pro, 48 GB unified memory. Use for editing, small checks,
  protocol inspection, and result analysis. CUDA is not available on this machine.
  Do not change the detector's methodology to target Apple Silicon.
- **Official experiments:** Ubuntu under WSL2 on the Windows desktop, NVIDIA RTX
  4070 Ti SUPER with 16 GB VRAM. The machine is intended as a private GPU server;
  reachable with the Mac's SSH alias `homeserver`. On 2026-11-04, inspection confirmed
  Ubuntu 24.04.5 under WSL2, driver 610.88, and 16,376 MiB GPU memory. WSL exposes
  about 15 GiB system RAM and 4 GiB swap. The source copy is in
  `/home/mut/projects/master-thesis`, with its own Linux environment and caches.
  It is a Git working copy on `practical-work`, based on local repository history,
  with the GitHub origin configured. Current uncommitted files were preserved.

PVP fits small CPU-side statistical models, but feature extraction requires large
pretrained neural models. `2_PVP.py` holds the CTC recognizer, acoustic backbone,
and speaker model in the same process. Sixteen GB is a target to test, not a measured
VRAM guarantee. Full-length audio is used in the main PVP path. Measure peak memory
on representative clips before a full run. Do not silently crop clips or lower precision
to fit memory; these would be changes that need separate validation and documentation.

## Dependencies stated by upstream

Source: [pinned install.sh](../original/PVP/install.sh), plus imports in
`2_PVP.py`, `analyzer.py`, `backbones/`, `data.py`, and the other experiment scripts.

| Package or tool | Upstream setting | Notes |
| --- | --- | --- |
| Python | `3.9.0` via Conda | Starting reference, not a claim of cross-platform compatibility. |
| pip | `24.0` via Conda | Keep installer/build tooling versions in provenance. |
| torch / torchaudio | `2.6.0` / `2.6.0` | Select a matching CUDA build for Linux; installer does not specify a CUDA wheel index or driver version. |
| fairseq | Git commit `862efab86f649c04ea31545ce28d13c59560113d`, editable install | Used by every acoustic wrapper; upstream explicitly avoids the PyPI 0.10.2 package. |
| speechbrain | `1.0.2` | ECAPA speaker embeddings. |
| numpy | `1.21.2` | Legacy numeric stack; do not upgrade without checking fairseq and scientific behavior. |
| pandas | `1.4.3` | Protocol parsing. |
| scipy | `1.7.2` | Numeric dependency; assess compatibility with selected Python and other packages. |
| scikit-learn | Unpinned | GMMs and ROC/AUC. GMM/metric behavior should be versioned. |
| transformers / huggingface_hub | Unpinned | CTC recognizer, tokenizer offsets, and model retrieval; API/version compatibility is unresolved. |
| librosa | Imported but not explicitly installed | Main PVP audio loading and resampling. Its version and transitive dependencies need explicit selection. |
| joblib / tqdm | Imported but not explicitly pinned | Profile serialization / progress display. Do not rely on accidental transitive versions. |
| soundfile | Unpinned install | Audio decoding backend. Verify codec support and actual audio loading. |
| lightning, tensorboard, tensorboardX, pandarallel | Unpinned install | Not directly needed by the main PVP path; Lightning is used by generic baseline/data utilities. |
| pytorch-lightning | Imported by `data.py` | Distinguish this package/import from `lightning`; not needed by the direct main PVP path. |
| matplotlib / seaborn | Imported by motivation script, not explicitly installed | Later plots only; avoid expanding the first inference environment without need. |

The installer has no complete dependency lock and no automated environment check.
Old NumPy/SciPy pins combined with unpinned modern libraries and build dependencies
may create resolver or API conflicts. This is a compatibility risk, not an observed
installation failure. The [pinned fairseq setup](https://github.com/pytorch/fairseq/blob/862efab86f649c04ea31545ce28d13c59560113d/setup.py)
also builds native extensions. Check build tools and its dependency constraints
before changing pins. The Phase 2 selections and exceptions are described below.

The macOS and Linux environments may need separate dependency files. In Phase 2,
prefer `uv` and `.venv` for our development code; use an isolated legacy environment
for upstream when necessary. Export exact working package/build versions on Linux.
Do not copy the full CUDA experiment stack to macOS without a concrete need.

## Required pretrained resources

These are model identities from the paper/code, not already downloaded resources.
All weight downloads require user approval. Record exact revisions and file hashes;
the upstream code uses mutable model IDs without revision pins.

| Resource | Official source | Upstream behavior |
| --- | --- | --- |
| Phoneme CTC model and processor | [facebook/wav2vec2-xlsr-53-espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft) | `analyzer.py:19,29-30` loads both through Transformers. Check phoneme tokenizer offset support. Model card expects 16 kHz speech. |
| Global speaker model | [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | `2_PVP.py:24-28` loads it at import time, saving to `pretrained_models/spkrec-ecapa-voxceleb`. Model card expects mono 16 kHz input to `encode_batch`. |
| Default MMS-300M anti-deepfake backbone | [nii-yamagishilab/mms-300m-anti-deepfake](https://huggingface.co/nii-yamagishilab/mms-300m-anti-deepfake) | `backbones/mms.py:67-73,132-138` requires a local state dictionary named `pretrained_models/mms_300m.ckpt`. It does not download it automatically. Verify the source file's format, key layout, and hash before placing or naming it. |
| Optional wav2vec2 small / large | [small](https://huggingface.co/nii-yamagishilab/wav2vec-small-anti-deepfake), [large](https://huggingface.co/nii-yamagishilab/wav2vec-large-anti-deepfake) | Loaded through `PyTorchModelHubMixin.from_pretrained` when selected in `analyzer.py`. Not required for the default MMS run. |
| Optional XLS-R 1B | [nii-yamagishilab/xls-r-1b-anti-deepfake](https://huggingface.co/nii-yamagishilab/xls-r-1b-anti-deepfake) | Larger alternative loaded when selected; memory feasibility unmeasured. |
| Optional HuBERT xlarge | [nii-yamagishilab/hubert-xlarge-anti-deepfake](https://huggingface.co/nii-yamagishilab/hubert-xlarge-anti-deepfake) | Wrapper additionally loads `pretrained_models/hubert_xtralarge_ll60k.pt` before hub weights. Exact base-checkpoint source/revision is not supplied in the PVP setup instructions; resolve before selecting it. |

Use `models/` and a local ignored cache for future weights. Because upstream expects
working-directory-relative paths, a later wrapper must provide an explicit working
directory/path mapping without modifying the pinned source. Keep generated files
outside `original/PVP/`.

## Environment verification

- The Linux dependency installation, pinned fairseq source build, package consistency,
  CUDA visibility, and a PyTorch operation on the desktop passed on 2026-11-04.
  Package, OS, driver, and PyTorch/CUDA build versions were saved.
- Import checks ran separately from model-loading scripts, with hub access offline.
- Checkpoint loading, CTC/MMS frame-count agreement, ECAPA embedding extraction, and
  GPU memory up to 30 s passed on 2026-11-05 with synthetic noise (section below).
- Still to verify on a small approved dataset sample: 16 kHz audio decoding, CTC phoneme
  offsets on real speech, peak memory for the longest clips, and that scoring accounts
  for every trial.

Dataset access, storage estimates, protocol schemas, and paper/code discrepancies
are recorded in [reproduction.md](reproduction.md). No dataset audio or detection
performance has been verified.

## Verified macOS development environment

The local environment uses CPython **3.9.25**, keeping the upstream Python 3.9
minor version while using a newer patch release than 3.9.0. `uv 0.13.1` created
the environment and installed 46 packages from a hash-checked lockfile.

- [requirements.txt](../requirements.txt): direct development dependencies.
- [macos-py39.lock](../requirements/macos-py39.lock): complete resolved package versions
  and distribution hashes for the macOS ARM64/Python 3.9 development target.
- [check_environment.py](../scripts/check_environment.py): repeatable offline checks,
  using synthetic data and a tiny temporary phoneme vocabulary.
- Local machine-readable report: `results/environment/macos.json` (ignored by Git).

The development environment includes NumPy 1.21.2, pandas 1.4.3, scikit-learn 1.1.3,
librosa 0.10.2.post1, Numba 0.56.4, SoundFile 0.12.1, PyTorch 2.6.0, Transformers
4.46.3, and huggingface-hub 0.25.2. The full transitive set is in the lockfile.
It does not include fairseq, torchaudio, or SpeechBrain; it is not a complete PVP
inference environment.

**Local-only exception:** SciPy is 1.7.3. The official
[SciPy 1.7.2 distribution metadata](https://pypi.org/pypi/scipy/1.7.2/json)
has no CPython 3.9 macOS ARM64 wheel. A binary-only resolution with SciPy 1.7.3
succeeded, and the installed version passed the checks below. The Linux candidate
retains the authors' SciPy 1.7.2 pin. Do not report macOS development checks as
official reproduction measurements.

**Local runtime constraint:** combined CTC-model imports and GMM fitting crashed
with a native bus error in NumPy's dot operation using default thread settings on
this Mac. The same complete check passes with `OPENBLAS_NUM_THREADS=1` and
`OMP_NUM_THREADS=1`. NumPy and SciPy report bundled OpenBLAS 0.3.17, and PyTorch
loads its OpenMP library. This demonstrates a local native-runtime compatibility
problem; its exact cause is not established. Use the thread limits for local checks
and analysis. This does not modify the detector, and it does not establish whether
the Linux stack has the same problem. The report records the thread settings.

Passed checks:

- Dependency metadata consistency: `uv pip check`.
- Numeric imports and a five-component diagonal GMM fitted to synthetic vectors.
- The pinned EER function on perfectly separated and reversed scores; ROC-AUC direction.
- Synthetic WAV decoding and resampling from 8 kHz to 16 kHz through librosa.
- CTC class imports and phoneme-token decoding with character offsets.
- CPU tensor multiplication and NumPy/PyTorch conversion.

CUDA availability is false on the Mac, as expected. No pretrained models were
constructed, no weights were loaded, and no benchmark audio was used.

To recreate this environment on an Apple Silicon Mac with uv installed:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
uv python install 3.9.25
uv venv --python 3.9.25 .venv
uv pip sync --python .venv/bin/python --require-hashes --only-binary :all: requirements/macos-py39.lock
uv pip check --python .venv/bin/python
.venv/bin/python scripts/check_environment.py --output results/environment/macos.json
```

Create a fresh `.venv` on each machine. Do not copy the Mac environment or cached
Python interpreter to the desktop. Environments and caches are ignored by Git.

## Linux dependency candidate

[pvp-linux.in](../requirements/pvp-linux.in) preserves the upstream NumPy 1.21.2,
pandas 1.4.3, SciPy 1.7.2, torch/torchaudio 2.6.0, and SpeechBrain 1.0.2 pins.
Its [candidate lockfile](../requirements/pvp-linux-py39-candidate.lock) resolves
94 packages for CPython 3.9 on Linux x86-64. This set was first resolved on the Mac
with an explicit Linux target, then installed on the WSL server. The pinned fairseq
source was built separately, bringing the installed package count to 95. Dependency
consistency, PVP imports, and offline CUDA checks pass. The file retains its original
candidate name because pretrained-model validation has not yet been completed.

The previously unpinned packages have explicit candidate choices, including
Transformers 4.46.3, huggingface-hub 0.25.2, librosa 0.10.2.post1, Numba 0.56.4,
scikit-learn 1.1.3, joblib 1.4.2, and tqdm 4.67.1. These are our environment
selections, not recovered authors' versions. The selected older Transformers/hub
APIs and numeric stack are checked locally where applicable. Numerical equivalence
to the authors' original environment is not established.

Additional findings:

- The pinned fairseq requires Hydra `>=1.0.7,<1.1` and OmegaConf `<2.1`. Resolution
  selects Hydra 1.0.7, OmegaConf 2.0.6, and `antlr4-python3-runtime==4.8`.
  A wheel-only resolution fails because antlr 4.8 is source-only; allowing its
  source package resolves this. Do not replace Hydra with a newer incompatible version.
- uv normalized OmegaConf's legacy requirement `PyYAML>=5.1.*` to `PyYAML>=5.1`
  with a warning. This is a packaging-metadata issue, not a detector modification.
- Cython 0.29.37 and wheel 0.45.1 are explicit build tools. The legacy fairseq native
  build passed on the desktop, producing `fairseq==1.0.0a0+862efab`. The source revision is
  installed separately; the dependency lock does not contain a prebuilt fairseq wheel.
- The Linux lock uses PyPI's torch 2.6.0 build and resolves NVIDIA CUDA 12.4 runtime
  libraries, including `nvidia-cuda-runtime-cu12==12.4.127`. The CUDA check passed on
  driver 610.88. Do not install a different torch wheel
  over this lock without recording and resolving the changed environment.
- The actual CTC model's [tokenizer configuration](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft/raw/main/tokenizer_config.json)
  names the phoneme tokenizer and espeak backend. Its constructor defaults to
  initializing phonemization. The Linux candidate therefore includes phonemizer
  3.3.0 and needs the system `espeak-ng` library, even though the detector decodes
  predicted phonemes rather than phonemizing supplied transcripts.

Candidate locks were resolved with `--exclude-newer 2026-11-03`. Hashes constrain
downloaded package distributions; they do not make untested code paths verified.

## Run directly on the desktop

Run the following in the desktop's own Windows and Ubuntu terminals. No remote
host, SSH credentials, or Mac connection is needed.

1. In **Windows**, install a WSL-compatible NVIDIA driver. In administrator
   PowerShell, install/update WSL and verify that Ubuntu uses version 2:

   ```powershell
   wsl --install -d Ubuntu
   wsl --update
   wsl --list --verbose
   ```

   Restart if requested, launch Ubuntu, and finish creating its Linux user.
   If Ubuntu already exists, use that installation rather than installing a duplicate.
   Follow the [Microsoft WSL guide](https://learn.microsoft.com/en-us/windows/wsl/install)
   and [NVIDIA WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
   The NVIDIA driver belongs on Windows; do not install a Linux display driver inside WSL.

2. In **Ubuntu**, verify `nvidia-smi` (or `/usr/lib/wsl/lib/nvidia-smi`) and install
   the software build/audio prerequisites:

   ```bash
   sudo apt update
   sudo apt install build-essential git curl libsndfile1 espeak-ng
   curl -LsSf https://astral.sh/uv/0.13.1/install.sh -o /tmp/uv-install.sh
   sh /tmp/uv-install.sh
   ```

   Open a new Ubuntu terminal if needed so `uv` is on `PATH`.
   The versioned installer follows [uv's official installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

3. Clone the project's `practical-work` branch into Ubuntu's Linux filesystem,
   for example `~/projects/master-thesis`, following the README command. When
   copying an existing checkout instead, exclude `.venv`, `.cache`, and generated
   results. Enter the project directory and run:

   ```bash
   bash scripts/setup_pvp_linux.sh
   ```

   The script checks prerequisites and GPU visibility, installs Python 3.9.25 and
   the candidate locked dependencies, fetches the authors' exact fairseq commit into
   `.cache/fairseq`, and builds it with the installed dependencies. It runs an offline
   import/numeric/audio/tokenizer/CUDA check. It downloads software dependencies only;
   it does not download model weights or datasets, and does not run the detector.

4. Review the generated files in `results/environment/`: `linux.json`,
   `linux-installed.txt`, `fairseq-commit.txt`, `nvidia-smi.txt`, and `uv-version.txt`.
   Repeat the verification independently with:

   ```bash
   .venv/bin/python scripts/check_environment.py --pvp --cuda --output results/environment/linux.json
   ```

The Linux script completed successfully on the desktop on 2026-11-04. A future
build/import/CUDA failure must be inspected and documented before changing versions.
This software check alone does not load pretrained models; the separate
[model loading check](#model-loading-and-inference-check-2026-11-05) does.

Verified server software: Python 3.9.25; torch/torchaudio 2.6.0; PyTorch CUDA build
12.4; SpeechBrain 1.0.2; fairseq commit `862efab86f649c04ea31545ce28d13c59560113d`;
NumPy 1.21.2 and SciPy 1.7.2. No Mac-only numeric version exception or CPU thread
limit was needed for these Linux checks.

W&B 0.26.1 was added using a separate hash-locked dependency extension. It added
12 packages and preserved all 95 existing package versions, including the research
stack above. Package consistency and all five environment checks passed afterwards.
Offline logging and an online artifact upload/download check also passed. These
are setup checks, not detector experiments. The Linux setup script now installs
this tracking extension too. W&B authentication is separate from package installation.

The server's `results/environment/` files were also retrieved to the Mac under
`results/environment/homeserver/`. These ignored artifacts contain the JSON report,
installed package list, exact fairseq commit, driver report, uv version, and setup log.
The report records all five checks as passed and `pretrained_models_checked: false`.

## Access from the MacBook

The configured SSH alias `homeserver` was verified on 2026-11-04. It opens a Windows
shell. To work interactively from the Mac terminal, run these commands in sequence
at the Mac, Windows SSH, and Ubuntu prompts respectively:

```text
ssh homeserver
wsl.exe -d Ubuntu
cd /home/mut/projects/master-thesis
```

Run Linux setup/check commands from that Ubuntu prompt. Keep the Mac `.venv` separate
from the server `.venv`. The source and Git history were cloned without Git pushes,
datasets, Mac caches, or the Mac environment. The GitHub origin is
`https://github.com/karakartal190360/master-thesis.git`; the current branch is
`practical-work`. The local uncommitted project files were copied as well, since a
fresh GitHub clone would omit them. This server copy is not automatically synchronized;
transfer later source changes deliberately before running them.

## Pinned model download plan

[pretrained-models.json](../configs/pretrained-models.json) records the three required
model revisions, selected filenames, sizes, and available LFS SHA-256 hashes from
the official Hugging Face metadata, inspected on 2026-11-04. The selected files total
2,622,150,843 bytes (about 2.62 GB). The user approved downloading these models on
2026-11-04. All 14 planned files were downloaded directly to the WSL project under
`/home/mut/projects/master-thesis/models/`, with one directory per model.

[download_pretrained.py](../scripts/download_pretrained.py) uses the pinned revisions
and explicit file list without importing or loading a model. Every file matched its
expected size; all four available published SHA-256 weight hashes matched. SHA-256
hashes were also recorded for all 14 files, including the small configuration files.
The completed report is `results/environment/pretrained-download.json` on WSL, with
a copy under `results/environment/homeserver/` on the Mac. It records the command,
script and executed-plan hashes, SDK version, revisions, file paths, sizes, and hashes.
The exact executed plan was also saved on the Mac as `pretrained-models.executed.json`
in that report directory before updating the plan's status field.

To repeat or resume the same downloads from the WSL project directory:

```bash
.venv/bin/python scripts/download_pretrained.py
```

Weights and Hugging Face download metadata stay in ignored server directories.
No model files were copied to the Mac or uploaded to W&B. Download integrity is
verified; loading and CUDA inference are verified in the section below.

The inspected MMS repository currently provides `model.safetensors`, while the PVP
wrapper expects a PyTorch state dictionary at `pretrained_models/mms_300m.ckpt`.
Do not rename the safetensors file to `.ckpt`. The keys were inspected and matched, and a
documented lossless conversion was done on 2026-11-05 (section below). The original
downloaded file and its hash are kept. No model tensor or detector change is
authorized by a packaging-format conversion.

## Model loading and inference check (2026-11-05)

Status: passed on the WSL server. This is a setup check (`scientific_result=false`),
run from a working copy with uncommitted changes. It used synthetic noise, not speech,
so it gives no detection or phoneme-quality evidence. These steps were done directly in
the WSL project; the Mac copy does not have them until transferred deliberately.

**1. MMS key inspection.** `model.safetensors` holds 431 float32 tensors: 429 under
`m_ssl.model.` (backbone) and `proj_fc.weight` [2, 1024] / `proj_fc.bias` [2] (output
layer). An empty fairseq `Wav2Vec2Model` built from upstream `mms_configs` expects
exactly the same 429 names and shapes: none missing, none extra, no shape mismatch.
So the file was made for this upstream wrapper; only its container format differs.

**2. Lossless MMS conversion.** [convert_mms.py](../scripts/convert_mms.py) checks the
pinned source hash, saves the unchanged tensors with `torch.save` as
`models/mms-300m-anti-deepfake/mms_300m.ckpt`, reloads it with `torch.load` (as upstream
does), and compares every tensor's name, dtype, shape and values. All 431 are equal.
The original safetensors file is kept. Report: `results/environment/mms-conversion.json`.

| File | SHA-256 | Bytes |
| --- | --- | --- |
| `model.safetensors` (downloaded source) | `9bd5d9785bf72f91dfca8508651354946c4b307921cb358505e1b94035102224` | 1,269,622,944 |
| `mms_300m.ckpt` (converted) | `c4dc55e229d974961dc59a0865d9f5af04b24a7a03aee5da94c55f1675cd2b10` | 1,269,711,239 |

The tensor-by-tensor comparison is the proof of correctness. The `.ckpt` hash identifies
this file only: another torch version may write different bytes for identical tensors.

**3. Local runtime layout.** Upstream finds models by names relative to its working
directory. [link_pretrained.py](../scripts/link_pretrained.py) creates symlinks in a
chosen working directory, outside `original/PVP/`, so the unchanged code uses the
pinned local files:

| Upstream lookup | Link target in `models/` | Why this name works |
| --- | --- | --- |
| `pretrained_models/mms_300m.ckpt` (`backbones/mms.py:67,132`) | converted MMS file | Direct `torch.load` path. |
| `facebook/wav2vec2-xlsr-53-espeak-cv-ft` (`analyzer.py:19`) | CTC model directory | Transformers loads an existing local directory with that name before trying the Hub. |
| `speechbrain/spkrec-ecapa-voxceleb` (`2_PVP.py:25`) | ECAPA directory | SpeechBrain 1.0.2 treats the source as local only if that directory exists. Otherwise it always contacts the Hub (`allow_updates=True`), even when `savedir` already holds the files. Found when the first offline run failed. |
| `pretrained_models/spkrec-ecapa-voxceleb/*` (`savedir`) | one link per ECAPA file | SpeechBrain writes into `savedir` (it added `custom.py` and `label_encoder.ckpt` links). Per-file links keep these writes out of `models/`. |
| `vocab.json` (`analyzer.py:114`) | `original/PVP/vocab.json` | Upstream silently uses an empty category map if this file is missing. |

The script refuses to replace any existing path that is not the expected link.

**4. Loading and inference.** [check_models.py](../scripts/check_models.py) forces
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` with an empty private `HF_HOME`, then uses
upstream's own code: it imports `2_PVP.py` (ECAPA at import) and creates
`UniversalPhonemeAnalyzer(device=DEVICE, feat_model_name=BACKBONE)` as in `2_PVP.py:112`
(`MMS`, `cuda:0`). It runs the same steps as `get_phoneme_features_from_file` and
`_extract_embedding` directly, so errors are raised rather than printed and skipped,
then calls the upstream function on the same file. Input: seeded Gaussian noise
(seed 0, 16 kHz mono) of 4, 15 and 30 seconds.

```bash
.venv/bin/python scripts/check_models.py
```

| Result | Value |
| --- | --- |
| Hub access | None. Private cache afterwards contains only `hub/version.txt`, an empty-cache marker. |
| Loading time (all three models) | 6-8 s |
| Weights equal to files, tensor by tensor | MMS backbone 429/429; CTC 424/424 (Transformers renames `weight_g`/`weight_v` to torch `parametrizations.weight.original0/1`; compared after mapping); ECAPA embedding model 231/231 |
| GPU memory allocated after loading | ECAPA 86 MiB; all three models 2,502 MiB |
| Frame counts, CTC versus MMS | Equal at every length: 199, 749, 1,499 frames (20 ms per frame). Upstream applies CTC offsets directly to MMS frames, so this is required. |
| CTC output size | 392 tokens per frame |
| ECAPA embedding | 192 values |
| Decoded phonemes on noise | 0 (expected; not a quality test) |

| Audio length | Peak allocated | Peak reserved | Inference time |
| --- | --- | --- | --- |
| 4 s | 2,635 MiB | 2,696 MiB | 0.8-1.3 s (first call includes CUDA warm-up) |
| 15 s | 2,929 MiB | 3,122 MiB | 0.08 s |
| 30 s | 3,311 MiB | 3,972 MiB | 0.19 s |

GPU: RTX 4070 Ti SUPER, driver 610.88, 16,376 MiB. Packages: torch 2.6.0 (CUDA 12.4),
fairseq `862efab`, Transformers 4.46.3, SpeechBrain 1.0.2, huggingface-hub 0.25.2,
safetensors 0.7.0. Report: `results/environment/model-check.json`, which records the
command, Git commit and dirty state, and the SHA-256 of `check_models.py` and
`link_pretrained.py`.

**Interpretation and limits.** About 2.5 GB is the fixed weight cost; activation memory
grows with clip length. Up to 30 s, peak use is far below 16 GB. Attention memory grows
roughly quadratically with length, so check the longest clip in the selected dataset
before a full run. No cropping or precision change was used or is planned. The ECAPA
normalization and classifier files were loaded but not compared tensor by tensor; the
classifier is not used for embeddings. Phoneme decoding, offsets and scores on real
speech remain untested until approved dataset audio is available.

Two problems occurred and were fixed in our scripts, not in upstream: WSL keeps
`nvidia-smi` in `/usr/lib/wsl/lib/` outside `PATH` (same fallback as
`setup_pvp_linux.sh`), and the SpeechBrain Hub lookup described above.
