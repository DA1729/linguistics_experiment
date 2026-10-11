#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mode="${1:-benchmark}"
backend="${2:-auto}"
config_path="${3:-$PWD/config/experiment.toml}"
case "$mode" in benchmark|run|package|setup|test) ;; *) echo 'Usage: bash run_local.sh [benchmark|run|package|setup|test] [auto|rocm|cpu] [config.toml]' >&2; exit 64 ;; esac
case "$backend" in auto|rocm|cpu) ;; *) echo 'Backend must be auto, rocm or cpu.' >&2; exit 64 ;; esac
if [[ "$mode" == test ]]; then exec python3 -m unittest discover -s tests -v; fi
python3 -c 'import sys; assert (3,11) <= sys.version_info[:2] <= (3,13), "Python 3.11–3.13 is required"'
runtime_root="${LAPTOP_ROOT:-$PWD/.runtime}"
mkdir -p "$runtime_root/logs" "$runtime_root/probes"
runtime_root="$(cd "$runtime_root" && pwd)"
export HF_HOME="$runtime_root/cache/huggingface"
export PIP_CACHE_DIR="$runtime_root/cache/pip"
export USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_ETAG_TIMEOUT=20 HF_HUB_DOWNLOAD_TIMEOUT=60
if [[ "$mode" == package ]]; then
    if [[ "$backend" == auto ]]; then backend="$(cat "$runtime_root/selected_backend" 2>/dev/null || echo cpu)"; fi
    exec python3 -m src.runner --mode package --device "$backend" --config "$config_path" --root "$runtime_root"
fi
exec 9>"$runtime_root/.setup.lock"
flock -n 9 || { echo 'Another laptop workflow is running.' >&2; exit 1; }
exec > >(tee -a "$runtime_root/logs/setup.log") 2>&1
python3 - "$config_path" <<'PY'
from pathlib import Path
import sys
from src.runner import read_config
from src.hardware import system_status, stop_reason
config = read_config(Path(sys.argv[1]))
status = system_status()
reason = stop_reason(config['laptop'], status)
if reason:
    raise SystemExit(reason)
print('Laptop preflight:', status)
PY
install_requirements() {
    local python_path="$1"
    if ! "$python_path" - <<'PY'
from importlib.metadata import version
from pathlib import Path
for line in Path('requirements.txt').read_text().splitlines():
    if line and not line.startswith('#'):
        name, expected = line.split('==')
        assert version(name) == expected, name
PY
    then
        timeout --kill-after=15 300 "$python_path" -m pip install --disable-pip-version-check -r requirements.txt
    fi
}
try_rocm() {
    local env_path="$runtime_root/venv_rocm"
    if [[ ! -x "$env_path/bin/python" ]]; then python3 -m venv "$env_path" || return 1; fi
    if ! "$env_path/bin/python" -c 'from importlib.metadata import version; assert version("torch") == "2.12.0+rocm7.14.1"' 2>/dev/null; then
        timeout --kill-after=15 900 "$env_path/bin/python" -m pip install --disable-pip-version-check \
            --index-url https://repo.amd.com/rocm/whl-multi-arch/ 'torch[device-gfx1152]==2.12.0+rocm7.14.1' || return 1
    fi
    install_requirements "$env_path/bin/python" || return 1
    timeout --signal=TERM --kill-after=10 60 "$env_path/bin/python" -m src.hardware \
        --backend rocm --output "$runtime_root/probes/rocm.json" || return 1
    runtime_python="$env_path/bin/python"
}
if [[ "$backend" != cpu ]]; then
    if try_rocm; then
        backend=rocm
    elif [[ "$backend" == rocm ]]; then
        echo 'ROCm check failed. Details are in .runtime/logs/setup.log; no experiment started.' >&2
        exit 1
    else
        echo 'ROCm is unavailable. Testing the CPU fallback; benchmark results will identify this backend.'
        backend=cpu
    fi
fi
if [[ "$backend" == cpu ]]; then
    env_path="$runtime_root/venv_cpu"
    if [[ ! -x "$env_path/bin/python" ]]; then python3 -m venv --system-site-packages "$env_path"; fi
    runtime_python="$env_path/bin/python"
    if ! "$runtime_python" -c 'import torch; assert tuple(map(int,torch.__version__.split("+")[0].split(".")[:2])) >= (2,6)' 2>/dev/null; then
        timeout --kill-after=15 600 "$runtime_python" -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cpu
    fi
    install_requirements "$runtime_python"
    timeout --kill-after=10 60 "$runtime_python" -m src.hardware --backend cpu --output "$runtime_root/probes/cpu.json"
fi
printf '%s\n' "$backend" > "$runtime_root/selected_backend"
if [[ "$mode" == setup ]]; then echo "Setup passed: $backend"; exit 0; fi
max_seconds="$(python3 - "$config_path" "$mode" <<'PY'
from pathlib import Path
import sys
from src.runner import read_config
c = read_config(Path(sys.argv[1]))
seconds = c['laptop']['benchmark_minutes'] * 60 if sys.argv[2] == 'benchmark' else c['runtime']['max_hours'] * 3600
print(int(seconds))
PY
)"
finish() {
    local result=$?
    trap - EXIT INT TERM
    python3 -m src.runner --mode package --device "$backend" --config "$config_path" --root "$runtime_root" || true
    echo "Local workflow exit code: $result. No background experiment remains running."
    exit "$result"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Leave 30 seconds for graceful checkpointing within the selected time budget.
soft_seconds=$((max_seconds > 30 ? max_seconds - 30 : 1))
timeout --signal=TERM --kill-after=30 "$soft_seconds" "$runtime_python" -m src.runner \
    --mode "$mode" --device "$backend" --config "$config_path" --root "$runtime_root"
