#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES=""
export HIP_VISIBLE_DEVICES=""
export ROCR_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="$PWD/.analysis_runtime/huggingface"
export MPLCONFIGDIR="$PWD/.analysis_runtime/matplotlib"
[[ -f data/multilingual_translation_v1_complete.zip ]] || { echo 'Missing experiment archive in data/.' >&2; exit 1; }
mkdir -p .analysis_runtime
exec 9>.analysis_runtime/analysis.lock
flock -n 9 || { echo 'Another analysis is already running.' >&2; exit 1; }
if [[ ! -x .analysis_runtime/venv/bin/python ]]; then
    python3 -m venv --system-site-packages .analysis_runtime/venv
fi
analysis_python="$PWD/.analysis_runtime/venv/bin/python"
if ! "$analysis_python" -c 'import torch'; then
    "$analysis_python" -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cpu
fi
if ! "$analysis_python" - <<'PY'
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
try:
    assert all(version(name) == wanted for name, wanted in (line.split('==') for line in Path('requirements_analysis.txt').read_text().splitlines() if line))
except (PackageNotFoundError, AssertionError):
    raise SystemExit(1)
PY
then
    "$analysis_python" -m pip install -r requirements_analysis.txt
fi
exec "$analysis_python" -m analysis.run_analysis "$@"
