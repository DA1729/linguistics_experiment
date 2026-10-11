import argparse
import json
from pathlib import Path
import platform
import time

from .storage import atomic_json, utc_now


def system_status() -> dict:
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        memory[key] = int(value.strip().split()[0]) * 1024
    supplies = []
    for path in Path('/sys/class/power_supply').glob('*/type'):
        if path.read_text().strip() in ('Mains', 'USB', 'USB_C'):
            online = path.with_name('online')
            if online.exists():
                supplies.append(online.read_text().strip() == '1')
    temperatures = []
    paths = [*Path('/sys/class/thermal').glob('thermal_zone*/temp'),
             *Path('/sys/class/hwmon').glob('hwmon*/temp*_input')]
    for path in paths:
        try:
            temperature = int(path.read_text().strip()) / 1000
            if 0 < temperature < 150:
                temperatures.append(temperature)
        except (OSError, ValueError):
            pass
    return {'available_memory_gib': memory['MemAvailable'] / 2**30,
            'total_memory_gib': memory['MemTotal'] / 2**30,
            'on_ac_power': any(supplies) if supplies else None,
            'temperature_c': max(temperatures) if temperatures else None,
            'kernel': platform.release(), 'checked_at': utc_now()}


def stop_reason(config: dict, status: dict) -> str | None:
    if config['require_ac_power'] and status['on_ac_power'] is not True:
        return 'AC power is disconnected or cannot be verified'
    if status['available_memory_gib'] < config['minimum_free_memory_gib']:
        return 'Available system memory is below the configured reserve'
    if status['temperature_c'] is not None and status['temperature_c'] >= config['maximum_temperature_c']:
        return 'Laptop temperature reached the configured stopping threshold'
    return None


def probe_backend(backend: str) -> dict:
    import torch
    report = {'backend_requested': backend, 'system': system_status(),
              'torch': torch.__version__, 'hip': torch.version.hip, 'cuda': torch.version.cuda,
              'status': 'failed', 'checked_at': utc_now()}
    try:
        if backend == 'rocm':
            if not torch.version.hip or not torch.cuda.is_available():
                raise RuntimeError('ROCm PyTorch cannot access an AMD GPU')
            device = 'cuda:0'
            report['gpu'] = torch.cuda.get_device_name(0)
            report['gpu_memory_bytes'] = torch.cuda.get_device_properties(0).total_memory
            report['gpu_architecture'] = getattr(torch.cuda.get_device_properties(0), 'gcnArchName', None)
        else:
            device = 'cpu'
        torch.set_num_threads(4)
        torch.manual_seed(1729)
        reference_input = torch.randn(64, 64)
        reference = reference_input @ reference_input.T
        start = time.monotonic()
        value = reference_input.to(device)
        output = (value @ value.T).cpu()
        if not torch.allclose(output, reference, atol=1e-4, rtol=1e-4):
            raise RuntimeError('Device matrix multiplication failed its CPU comparison')
        report.update(status='passed', device=device, check_seconds=time.monotonic() - start)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bounded CPU or ROCm device validation')
    parser.add_argument('--backend', choices=['cpu', 'rocm'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = probe_backend(args.backend)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'passed' else 1)
