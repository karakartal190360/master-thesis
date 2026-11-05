#!/usr/bin/env bash
set -euo pipefail

# Run inside Ubuntu/WSL2 on the RTX desktop.
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "Run this script on Linux x86-64, including Ubuntu under WSL2."
    exit 1
fi

for command in uv git g++ espeak-ng; do
    if ! command -v "$command" >/dev/null; then
        echo "Missing prerequisite: $command. See docs/environment.md."
        exit 1
    fi
done

if command -v nvidia-smi >/dev/null; then
    NVIDIA_SMI="$(command -v nvidia-smi)"
elif [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
    NVIDIA_SMI=/usr/lib/wsl/lib/nvidia-smi
else
    echo "NVIDIA GPU is not visible. Complete the WSL2/driver check first."
    exit 1
fi
"$NVIDIA_SMI"

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export UV_CACHE_DIR="$PROJECT_DIR/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PROJECT_DIR/.cache/python"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MAX_JOBS=2

uv python install 3.9.25
if [[ ! -d .venv ]]; then
    uv venv --python 3.9.25 .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:3] == (3, 9, 25), "Expected Python 3.9.25; create a fresh .venv as documented"'
uv pip sync --python .venv/bin/python --require-hashes requirements/pvp-linux-py39-candidate.lock

# Build the authors' exact fairseq revision, outside the source snapshot.
FAIRSEQ_COMMIT=862efab86f649c04ea31545ce28d13c59560113d
if [[ ! -d .cache/fairseq/.git ]]; then
    git init .cache/fairseq
    git -C .cache/fairseq remote add origin https://github.com/pytorch/fairseq.git
fi
git -C .cache/fairseq fetch --depth 1 origin "$FAIRSEQ_COMMIT"
git -C .cache/fairseq checkout --detach "$FAIRSEQ_COMMIT"
uv pip install --python .venv/bin/python --no-build-isolation --no-deps .cache/fairseq
uv pip install --python .venv/bin/python --require-hashes -r requirements/tracking-linux-py39.lock
uv pip install --python .venv/bin/python --require-hashes -r requirements/data-linux-py39.lock
uv pip check --python .venv/bin/python

mkdir -p results/environment
"$NVIDIA_SMI" > results/environment/nvidia-smi.txt
uv --version > results/environment/uv-version.txt
git -C .cache/fairseq rev-parse HEAD > results/environment/fairseq-commit.txt
uv pip freeze --python .venv/bin/python > results/environment/linux-installed.txt
.venv/bin/python scripts/check_environment.py --pvp --cuda --output results/environment/linux.json
echo "Software checks passed. Pretrained model loading still needs approved weights."
