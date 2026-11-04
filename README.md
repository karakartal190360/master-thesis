# master-thesis

Personalized speech deepfake detection research for the Master's in Artificial
Intelligence at Johannes Kepler University Linz (JKU).

The starting point is Xue et al. (2026), *Profiling the Voice: Speaker-Specific
Phoneme Fingerprinting for Speech Deepfake Detection* (PVP).
The [paper PDF](Papers/profiling-the-voice.pdf) is the scientific reference;
the [official implementation](https://github.com/JunXue-tech/PVP) is the code reference.

## Scope and current status

- **Practical Work, November 2026-February 2027:** reproduce PVP, compare measured
  EER and ROC-AUC with the paper, then evaluate German/English speech and suitable
  cross-language enrollment. Initial reproduction target: end of November 2026.
- **Master's Thesis, March-July 2027:** investigate an additional contribution in
  cross-language generalization/adaptation. The final question needs supervisor
  agreement; thesis algorithm development is not authorized yet.

Phase 1 is complete. Phase 2 is In Progress: the macOS development environment
passes offline checks with the documented CPU thread limits.
The home server is reachable from the MacBook, and its WSL2 Ubuntu can see the GPU.
The Linux software installation, pinned fairseq build, and CUDA checks pass;
pretrained model loading still needs verification.
The private [W&B project](https://wandb.ai/umuttanriverdi/personalized-speech-deepfake-detection)
is configured and verified with a non-scientific setup run. See [tracking setup and
experiment rules](docs/tracking.md). The configurable experiment runner is still pending.
**No detector experiment has been run and no reproduction result is claimed.**

The single project roadmap and persistent instructions are in [AGENTS.md](AGENTS.md).
[CLAUDE.md](CLAUDE.md) imports that file without duplicating it.

## How PVP works

PVP builds a profile from genuine recordings of one speaker. It finds phonemes
(speech sounds), extracts acoustic embeddings (numerical audio representations),
and fits a Gaussian Mixture Model (GMM) for each sound and for the speaker's global
voice. It checks new speech against these profiles, prioritizes reliable sounds,
and combines phoneme consistency with speaker consistency. Higher scores indicate
greater agreement with the genuine profile.

The [paper/code inspection](docs/reproduction.md) explains the pipeline, required
files, published comparison targets, and important differences. In particular,
the supplied enrollment lists are not uniformly 1% of genuine speech, and the code's
speaker GMM and score normalization differ from the paper's implementation settings.

## Initial setup

```bash
git clone --branch practical-work https://github.com/karakartal190360/master-thesis.git
cd master-thesis
```

1. Read [AGENTS.md](AGENTS.md), the [paper](Papers/profiling-the-voice.pdf), and
   [reproduction notes](docs/reproduction.md).
2. Inspect the unchanged source in `original/PVP/`. It is pinned to
   `1eed3f43309e6728583a5e43595e68121472b2ca`; see
   [provenance and hashes](original/UPSTREAM.md). No submodule initialization is needed.
3. Read [environment requirements](docs/environment.md) before installing packages.
   The authors use Python 3.9.0, PyTorch/torchaudio 2.6.0, a pinned fairseq commit,
   and SpeechBrain 1.0.2, with an incomplete set of other dependency pins.

The Mac development environment uses Python 3.9.25 and a separate locked dependency
set. Run its offline verification with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/check_environment.py --output results/environment/macos.json
```

For the desktop, first follow the Windows/WSL2 and Ubuntu prerequisites in
[environment setup](docs/environment.md#run-directly-on-the-desktop). Then run this
**inside Ubuntu on the desktop**, from the project directory:

```bash
bash scripts/setup_pvp_linux.sh
```

The script installs software dependencies and the pinned fairseq source, then
checks imports, synthetic audio/numerics/tokenizer behavior, and CUDA. It downloads
no model weights or datasets. Its first Linux run passed on the home server.
The current source files and Git history are in `/home/mut/projects/master-thesis`
in WSL, on `practical-work`, with the GitHub origin configured. Unpushed working
files were preserved. Mac environments/caches were excluded; the server has its own
Linux `.venv`. Later edits are not automatically synchronized.

To use that copy from the Mac, connect with `ssh homeserver`, enter Ubuntu with
`wsl.exe -d Ubuntu`, and run `cd /home/mut/projects/master-thesis`.

Do not import the authors' experiment entry points as a harmless setup check:
the main script loads a model at import time and assumes CUDA.

Use the MacBook for local development and analysis. Run official experiments on
Ubuntu/WSL2 with the RTX 4070 Ti SUPER (16 GB VRAM); GPU visibility and PyTorch
CUDA execution are verified, while pretrained inference memory remains unmeasured.
Do not assume CUDA on macOS.

Both [ZH-Famous](https://huggingface.co/datasets/PVP-ADD/ZH-Famous) and
[Famous Figures](https://huggingface.co/datasets/issf/famousfigures) require access
conditions/review. Dataset audio is absent. The three pinned pretrained models are
downloaded and file integrity is verified on WSL; model loading remains pending.
Weights are stored only in the server's ignored `models/` directory. Obtain approval
before downloading large datasets or weights. Raw data, weights, caches, and outputs
are excluded from Git. No changes may be pushed without approval.
The user confirms ZH-Famous access; Famous Figures approval is pending.
Dataset licenses remain pending review.

## Initial repository structure

```text
master-thesis/
├── .gitignore
├── AGENTS.md
├── CLAUDE.md
├── README.md
├── requirements.txt          # Direct macOS development dependencies
├── requirements/
│   ├── macos-py39.lock
│   ├── pvp-linux.in
│   ├── pvp-linux-py39-candidate.lock
│   ├── tracking.in
│   └── tracking-linux-py39.lock
├── scripts/
│   ├── check_environment.py
│   ├── check_tracking.py
│   ├── download_pretrained.py
│   └── setup_pvp_linux.sh
├── configs/
│   ├── pretrained-models.json
│   └── tracking.json
├── Papers/
│   └── profiling-the-voice.pdf
├── original/
│   ├── UPSTREAM.md
│   ├── manifest.json
│   └── PVP/                  # Complete unchanged upstream snapshot
│       ├── 2_PVP.py
│       ├── analyzer.py
│       ├── backbones/
│       ├── protocols/
│       └── ...
└── docs/
    ├── reproduction.md
    ├── environment.md
    └── tracking.md
```

Local `.venv/`, `.cache/`, and `results/environment/` contain the development
environment and verification output; Git ignores them. Add `src/`,
`notebooks/`, `data/`, `models/`, and `tests/` when needed. Their responsibilities
are defined in `AGENTS.md`; no empty Python modules are created for structure alone.
