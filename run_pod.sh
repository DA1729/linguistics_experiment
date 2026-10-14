#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
project_dir="$PWD"
mode="${1:-all}"
config_path="${2:-$project_dir/config/experiment.toml}"
case "$mode" in all|smoke|pilot|main|resume|package) ;; *) echo 'Usage: bash run_pod.sh [all|smoke|pilot|main|resume|package] [config.toml]' >&2; exit 64 ;; esac
[[ "$(uname -s)" == Linux ]] || { echo 'Linux is required.' >&2; exit 1; }
command -v python3 >/dev/null || { echo 'Choose a RunPod PyTorch template with Python 3.11–3.13.' >&2; exit 1; }
python3 -c 'import sys; assert (3,11) <= sys.version_info[:2] <= (3,13), "Use a Python 3.11–3.13 PyTorch template"'
if [[ -n "${LINGUISTIC_ROOT:-}" ]]; then
    runtime_root="$LINGUISTIC_ROOT"
elif [[ -d /workspace && -w /workspace ]]; then
    runtime_root=/workspace/linguistic_big_runtime
else
    runtime_root="$project_dir/.runtime"
    echo "No writable /workspace found; using $runtime_root. Verify its durability."
fi
mkdir -p "$runtime_root/logs"
runtime_root="$(cd "$runtime_root" && pwd)"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export USE_FLAX=0
export HF_HOME="$runtime_root/cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
# Pod templates may enable the optional hf_transfer package without installing it.
# Use the download backends provided by our pinned requirements instead.
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_ETAG_TIMEOUT=30
export HF_HUB_DOWNLOAD_TIMEOUT=60
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ "$mode" == package ]]; then
    exec python3 -m src.runner --mode package --config "$config_path" --root "$runtime_root"
fi
command -v flock >/dev/null
command -v timeout >/dev/null
exec 9>"$runtime_root/.setup.lock"
flock -n 9 || { echo 'Another workflow is using this runtime directory.' >&2; exit 1; }
exec > >(tee -a "$runtime_root/logs/bootstrap.log") 2>&1
started_at="$(date +%s)"
max_seconds="$(python3 - "$config_path" <<'PY'
import sys
from pathlib import Path
from src.runner import read_config
print(int(read_config(Path(sys.argv[1]))['runtime']['max_hours'] * 3600))
PY
)"
remaining_seconds() {
    local remaining=$((started_at + max_seconds - $(date +%s)))
    (( remaining > 0 )) || remaining=1
    echo "$remaining"
}
finish() {
    local result=$?
    trap - EXIT INT TERM
    # The Python runner normally packages itself. This also handles setup failure or hard timeout.
    python3 -m src.runner --mode package --config "$config_path" --root "$runtime_root" || true
    echo "Workflow exit code: $result. Bootstrap log: $runtime_root/logs/bootstrap.log"
    echo 'The script does not stop or terminate the Pod. Stop it in RunPod to avoid further GPU charges.'
    exit "$result"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo "Linux: $(uname -sr); Python: $(python3 --version); persistent root: $runtime_root"
command -v nvidia-smi >/dev/null || { echo 'No NVIDIA GPU tools found. Use a GPU PyTorch/CUDA Pod.' >&2; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
venv_path="$runtime_root/venv"
if [[ ! -x "$venv_path/bin/python" ]]; then
    if ! python3 -m venv --system-site-packages "$venv_path"; then
        timeout "$(remaining_seconds)" python3 -m pip install --user virtualenv==20.35.3
        python3 -m virtualenv --system-site-packages "$venv_path"
    fi
fi
runtime_python="$venv_path/bin/python"
if ! "$runtime_python" - <<'PY'
import torch
assert tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) >= (2, 6)
assert torch.cuda.is_available()
x = torch.ones(4, device='cuda')
assert float(x.sum()) == 4
print('Reusing CUDA-compatible PyTorch', torch.__version__, 'CUDA', torch.version.cuda)
PY
then
    echo 'Installing pinned CUDA 12.4 PyTorch fallback (requires a compatible NVIDIA driver).'
    timeout "$(remaining_seconds)" "$runtime_python" -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
fi
if ! "$runtime_python" - <<'PY'
from importlib.metadata import version
from pathlib import Path
for line in Path('requirements.txt').read_text().splitlines():
    if line and not line.startswith('#'):
        name, expected = line.split('==')
        assert version(name) == expected, name
PY
then
    timeout "$(remaining_seconds)" "$runtime_python" -m pip install --disable-pip-version-check -r requirements.txt
fi
"$runtime_python" - <<'PY'
import torch
from transformers import AutoModelForSeq2SeqLM as auto_model, AutoTokenizer as auto_tokenizer
assert torch.cuda.is_available(), 'CUDA unavailable after setup; check the Pod template/driver'
print('CUDA inference environment ready:', torch.__version__)
PY
# max_time limits normal generation; GNU timeout covers hung native calls as a last resort.
set +e
timeout --signal=TERM --kill-after=120 "$(remaining_seconds)" "$runtime_python" -m src.runner \
    --mode "$mode" --config "$config_path" --root "$runtime_root" --started-at "$started_at"
result=$?
set -e
exit "$result"
