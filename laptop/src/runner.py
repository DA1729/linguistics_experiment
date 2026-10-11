import argparse
import copy
import logging
import json
import math
from pathlib import Path
import re
import signal
import sys
import time
import tomllib
import uuid

from .dataset import prepare_corpus
from .experiment import run_experiment
from .languages import language_names, observation_rounds, select_pool
from .package_data import package_data
from .storage import (append_rows, atomic_bytes, atomic_json, digest, exclusive_lock, file_hash,
                      read_json, read_jsonl, utc_now)

project_root = Path(__file__).resolve().parent.parent


def read_config(path: Path) -> dict:
    with path.open('rb') as stream:
        config = tomllib.load(stream)
    phase_keys = {'sentences', 'rounds', 'repetitions', 'observe_every', 'observation_rounds'}
    schema = {
        'experiment': {'name', 'seed', 'selection_seed', 'language_count', 'languages'},
        'model': {'name', 'revision', 'precision', 'batch_size', 'max_input_tokens', 'max_new_tokens', 'max_batch_seconds', 'device', 'cpu_threads'},
        'dataset': {'local_file', 'local_license'},
        'runtime': {'max_hours', 'gpu_hourly_usd', 'progress_seconds'},
        'smoke': phase_keys, 'pilot': phase_keys, 'main': phase_keys,
        'laptop': {'benchmark_minutes', 'estimate_safety_factor', 'minimum_free_memory_gib', 'maximum_temperature_c', 'require_ac_power'},
    }
    if set(config) != set(schema) or any(set(config[k]) != schema[k] for k in schema):
        raise ValueError('Configuration must contain exactly the documented sections and keys')
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,79}', config['experiment']['name']):
        raise ValueError('experiment.name must be a short lowercase directory-safe identifier')
    select_pool(config['experiment'])
    for key in ('seed', 'selection_seed', 'language_count'):
        if type(config['experiment'][key]) is not int or config['experiment'][key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    for phase in ('smoke', 'pilot', 'main'):
        for key in ('sentences', 'rounds', 'repetitions', 'observe_every'):
            value = config[phase][key]
            if type(value) is not int or value < (1 if key in ('sentences', 'repetitions') else 0):
                raise ValueError(f'{phase}.{key} is invalid')
        if any(type(r) is not int or r < 0 for r in config[phase]['observation_rounds']):
            raise ValueError(f'{phase}.observation_rounds must be nonnegative integers')
    if config['smoke']['repetitions'] != 1 or config['smoke']['rounds'] < 2 or config['smoke']['sentences'] < 2:
        raise ValueError('Smoke checks require one repetition, at least two rounds and two sentences')
    if config['pilot']['rounds'] < 1:
        raise ValueError('Pilot needs at least one round for throughput measurement')
    if not 0 < config['runtime']['max_hours'] <= 4:
        raise ValueError('Laptop execution is capped at four hours')
    local = config['laptop']
    if not 0 < local['benchmark_minutes'] <= 15 or not 1.5 <= local['estimate_safety_factor'] <= 5:
        raise ValueError('Use a benchmark of at most 15 minutes and safety factor >= 1.5')
    if not 0.5 <= local['minimum_free_memory_gib'] <= 8 or not 60 <= local['maximum_temperature_c'] <= 95:
        raise ValueError('Invalid laptop memory/temperature limit')
    if type(local['require_ac_power']) is not bool:
        raise ValueError('require_ac_power must be a boolean')
    model = config['model']
    if model['device'] not in ('auto', 'cpu', 'rocm', 'cuda') or not 1 <= model['cpu_threads'] <= 8:
        raise ValueError('Invalid device or CPU thread count')
    if not re.fullmatch('[0-9a-f]{40}', model['revision']):
        raise ValueError('Use an exact 40-character model commit revision')
    if model['precision'] not in ('float32', 'float16', 'bfloat16'):
        raise ValueError('Unsupported inference precision')
    for key, minimum, maximum in [('batch_size', 1, 4096), ('max_input_tokens', 4, 1024), ('max_new_tokens', 4, 1023)]:
        if type(model[key]) is not int or not minimum <= model[key] <= maximum:
            raise ValueError(f'Invalid model.{key}')
    for value in (model['max_batch_seconds'], config['runtime']['max_hours'], config['runtime']['progress_seconds']):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Time limits must be positive and finite')
    price = config['runtime']['gpu_hourly_usd']
    if not isinstance(price, (int, float)) or not math.isfinite(price) or price < 0:
        raise ValueError('Hourly GPU cost must be finite and nonnegative')
    if config['dataset']['local_file']:
        config['dataset']['local_file'] = str((path.resolve().parent / config['dataset']['local_file']).resolve())
    return config


def source_files() -> list[Path]:
    return sorted((project_root / 'src').glob('*.py')) + [project_root / name for name in ('run_local.sh', 'requirements.txt')]


def initialize(root: Path, config: dict, config_path: Path) -> None:
    scientific = copy.deepcopy(config)
    scientific.pop('runtime')
    sources = {str(p.relative_to(project_root)): file_hash(p) for p in source_files()}
    identity = {'config': scientific, 'source_sha256': sources}
    path = root / 'identity.json'
    if path.exists():
        if read_json(path) != identity:
            raise ValueError('Scientific configuration or code changed. Restore it to resume, or choose a new experiment.name.')
    else:
        for p in source_files() + sorted((project_root / 'tests').glob('*.py')) + [project_root / 'README.md', project_root / 'ATTRIBUTION.md']:
            atomic_bytes(root / 'reproduction' / p.relative_to(project_root), p.read_bytes())
        atomic_json(root / 'effective_config.json', config)
        atomic_bytes(root / 'reproduction/config/experiment.toml', config_path.read_bytes())
        atomic_json(path, identity)
    atomic_json(root / 'last_invocation_config.json', config)


def smoke_check(root: Path, config: dict, corpus: list[dict], pool: list[str], environment: dict,
                translate, should_stop) -> None:
    report_path = root / 'smoke_report.json'
    if report_path.exists():
        report = read_json(report_path)
        if report['environment_digest'] != digest(environment):
            raise ValueError('Smoke environment changed')
        logging.info('Saved local smoke test already passed')
        return
    # All probes are preserved separately from experimental trajectories.
    attempt = root / 'smoke_probes' / uuid.uuid4().hex
    attempt.mkdir(parents=True)
    probes = [
        {'sentence_id': 'probe_en_fr', 'text': 'The cat is sleeping on the chair.', 'source_language': 'eng_Latn', 'target_language': 'fra_Latn'},
        {'sentence_id': 'probe_fr_de', 'text': 'Le chat dort sur la chaise.', 'source_language': 'fra_Latn', 'target_language': 'deu_Latn'},
        {'sentence_id': 'probe_ar_de', 'text': 'القط ينام على الكرسي.', 'source_language': 'arb_Arab', 'target_language': 'deu_Latn'},
    ]
    def persist_probes(mode, items, results):
        append_rows(attempt / f'{mode}.jsonl', [{'request': item, 'result': result, 'mode': mode}
                                              for item, result in zip(items, results)])

    mixed = translate(probes, should_stop=should_stop,
                      on_batch=lambda items, results: persist_probes('mixed', items, results))
    single = []
    for item in probes:
        single.extend(translate([item], should_stop=should_stop,
                                on_batch=lambda items, results: persist_probes('individual', items, results)))
    if any(row['status'] != 'ok' or row['warnings'] for row in mixed + single):
        raise RuntimeError('Local smoke probes returned failures or limit warnings; inspect smoke probe data')
    matches = [a['output_text'] == b['output_text'] for a, b in zip(mixed, single)]
    if not all(matches):
        logging.warning('Local smoke: mixed/single greedy outputs differ. Raw comparisons retained; floating point batching is not bitwise invariant.')
    smoke_dir = root / 'runs/smoke_001'
    first = run_experiment(smoke_dir, config, 'smoke', 1, corpus, pool, environment, translate,
                           should_stop, stop_after_round=1)
    if should_stop():
        raise InterruptedError('Stopped during smoke test')
    prior = {str(p): file_hash(p) for p in smoke_dir.glob('rounds/round_*/*') if p.is_file()}
    resumed = run_experiment(smoke_dir, config, 'smoke', 1, corpus, pool, environment, translate, should_stop)
    if resumed['status'] != 'complete':
        raise InterruptedError('Stopped during smoke recovery test')
    if any(file_hash(Path(p)) != expected for p, expected in prior.items()):
        raise RuntimeError('Smoke recovery modified a committed round')
    for path in smoke_dir.glob('rounds/round_*/*.jsonl'):
        if path.name not in ('translations.jsonl', 'observations.jsonl', 'journal.jsonl'):
            continue
        rows = read_jsonl(path)
        keys = [(row['kind'], row['round'], row['sentence_id']) for row in rows]
        if len(keys) != len(set(keys)):
            raise RuntimeError('Smoke recovery produced duplicate records')
    atomic_json(report_path, {'status': 'passed', 'environment_digest': digest(environment),
                             'probe_directory': str(attempt.relative_to(root)),
                             'checked_at': utc_now(), 'mixed_individual_exact_matches': matches,
                             'source_token_checks': 'passed on every encoded input',
                             'target_token_checks': 'passed on every generation',
                             'checkpoint_recovery': 'passed', 'first_stop_round': first['last_completed_round']})


def estimate_main(config: dict, pilot_results: list[dict]) -> dict:
    seconds = sum(row['collection_seconds'] for row in pilot_results)
    completed = sum(row['model_translations'] for row in pilot_results)
    if completed == 0 or seconds <= 0:
        raise ValueError('Pilot has no measured translations; cannot estimate main runtime')
    main = config['main']
    work = main['sentences'] * main['repetitions'] * (main['rounds'] + len(observation_rounds(main)) - 1)
    hours = work * seconds / completed / 3600
    return {'pilot_translations': completed, 'pilot_collection_seconds': seconds,
            'translations_per_second': completed / seconds, 'main_translation_upper_count': work,
            'main_hours_estimate': hours, 'main_gpu_cost_usd_estimate': hours * config['runtime']['gpu_hourly_usd'],
            'assumed_gpu_hourly_usd': config['runtime']['gpu_hourly_usd'],
            'method': 'Measured pilot time per translation, including diagnostics, scaled to main requests; not a bound or billing guarantee'}


def work_count(settings: dict) -> int:
    return settings['sentences'] * settings['repetitions'] * (settings['rounds'] + len(observation_rounds(settings)) - 1)


def estimate_local(config: dict, pilot_results: list[dict], successful: int, limited: int,
                   startup_seconds: float, environment: dict) -> dict:
    estimate = estimate_main(config, pilot_results)
    seconds_per_translation = 1 / estimate['translations_per_second']
    factor = config['laptop']['estimate_safety_factor']
    scenarios = {
        'configured_local_experiment': work_count(config['main']),
        'original_pod_pilot': 6000,
        'original_pod_main': 123000,
    }
    estimates = {}
    for name, count in scenarios.items():
        expected = count * seconds_per_translation
        guarded = expected * factor + startup_seconds + 120
        estimates[name] = {'model_translations': count, 'expected_hours': expected / 3600,
                           'planning_hours': guarded / 3600,
                           'within_budget': guarded <= config['runtime']['max_hours'] * 3600}
    adequate = estimate['pilot_translations'] >= 32 and successful == estimate['pilot_translations'] and limited == 0
    return {**estimate, 'device': environment['backend'], 'precision': config['model']['precision'],
            'config_digest': digest(config), 'environment_digest': digest(environment),
            'generated_at': utc_now(), 'successful_benchmark_translations': successful,
            'limit_or_truncation_events': limited, 'safety_factor': factor,
            'max_hours': config['runtime']['max_hours'], 'adequate_benchmark': adequate,
            'scenarios': estimates,
            'allow_local_run': adequate and estimates['configured_local_experiment']['within_budget'],
            'caveat': 'Short-run extrapolation, not a guarantee. Length drift, thermal throttling and batching can change runtime.'}


def execute(args) -> int:
    from .hardware import stop_reason, system_status
    config = read_config(args.config)
    config['model']['device'] = args.device
    root = args.root.resolve() / 'data' / f"{config['experiment']['name']}_{args.device}"
    archives = args.root.resolve() / 'archives'
    if args.mode == 'package' and not root.exists():
        raise ValueError(f'No data at {root}')
    with exclusive_lock(root):
        if args.mode == 'package':
            print(f'Archive: {package_data(root, archives)}')
            return 0
        log_directory = root / 'logs'
        log_directory.mkdir(exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                            handlers=[logging.StreamHandler(), logging.FileHandler(log_directory / 'execution.log', encoding='utf-8')], force=True)
        stopped = False
        last_health_check = 0.0
        health_reason = None
        invocation_start = time.time()
        seconds_limit = config['laptop']['benchmark_minutes'] * 60 if args.mode == 'benchmark' else config['runtime']['max_hours'] * 3600
        deadline = invocation_start + seconds_limit

        def handle_signal(signum, frame):
            nonlocal stopped
            stopped = True
            logging.warning('Stopping at the next batch boundary after signal %s', signum)

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        def should_stop():
            nonlocal last_health_check, health_reason
            now = time.time()
            if now - last_health_check > 2:
                health_reason = stop_reason(config['laptop'], system_status())
                last_health_check = now
            return stopped or now >= deadline or health_reason is not None

        workflow = {'status': 'incomplete', 'requested_mode': args.mode, 'invocation_started_at': utc_now()}
        invocation_path = root / 'invocations' / f'{uuid.uuid4().hex}.json'
        result = 0
        try:
            initialize(root, config, args.config)
            atomic_json(root / 'workflow_status.json', workflow)
            atomic_json(invocation_path, {**workflow, 'max_seconds': seconds_limit})
            health = system_status()
            atomic_json(root / 'laptop_hardware.json', health)
            reason = stop_reason(config['laptop'], health)
            if reason:
                raise InterruptedError(reason)
            pool = select_pool(config['experiment'])
            atomic_json(root / 'language_pool.json', [{'code': code, 'name': language_names.get(code, code)} for code in pool])
            count = max(config[phase]['sentences'] for phase in ('smoke', 'pilot', 'main'))
            corpus, _ = prepare_corpus(root / 'corpus', config['dataset'], count, config['experiment']['selection_seed'])
            if args.mode == 'run':
                report_path = root / 'runtime_estimate.json'
                if not report_path.exists():
                    raise ValueError('Run the short benchmark first; no experiment was started')
                report = read_json(report_path)
                if report['config_digest'] != digest(config) or not report['allow_local_run']:
                    raise ValueError('The configured experiment lacks a passing estimate within the four-hour budget')
            if should_stop():
                raise InterruptedError(health_reason or 'Benchmark deadline reached')
            from .translation import load_translator
            start = time.monotonic()
            translate, environment, engine = load_translator(config['model'], pool, config['experiment']['seed'])
            startup_seconds = time.monotonic() - start
            path = root / 'environment.json'
            if path.exists() and read_json(path) != environment:
                raise ValueError('Execution environment changed; choose a new experiment.name')
            atomic_json(path, environment)
            logging.info('Backend=%s GPU=%s precision=%s threads=%s', environment['backend'], environment['gpu'],
                         environment['precision'], environment['cpu_threads'])
            smoke_check(root, config, corpus, pool, environment, translate, should_stop)
            if args.mode == 'benchmark':
                results = []
                for repetition in range(1, config['pilot']['repetitions'] + 1):
                    status = run_experiment(root / 'runs' / f'pilot_{repetition:03d}', config, 'pilot', repetition,
                                            corpus, pool, environment, translate, should_stop)
                    if status['status'] != 'complete':
                        raise InterruptedError(health_reason or 'Short benchmark time limit reached')
                    results.append(status)
                records = [row for p in root.glob('runs/pilot_*/rounds/round_*/*.jsonl')
                           if p.name in ('translations.jsonl', 'observations.jsonl')
                           for row in read_jsonl(p) if row['model_invoked']]
                report = estimate_local(config, results, sum(row['status'] == 'ok' for row in records),
                                        sum(any(flag in row['warnings'] for flag in ('input_truncated', 'generation_token_limit', 'generation_time_limit')) for row in records),
                                        startup_seconds, environment)
                atomic_json(root / 'runtime_estimate.json', report)
                print(json.dumps(report['scenarios'], indent=2), flush=True)
                print(f"Local experiment permitted by estimate: {report['allow_local_run']}. Benchmark stopped; no main experiment started.", flush=True)
            else:
                if report['environment_digest'] != digest(environment):
                    raise ValueError('Benchmark environment differs from the current runtime')
                for repetition in range(1, config['main']['repetitions'] + 1):
                    status = run_experiment(root / 'runs' / f'main_{repetition:03d}', config, 'main', repetition,
                                            corpus, pool, environment, translate, should_stop)
                    if status['status'] != 'complete':
                        raise InterruptedError(health_reason or 'Four-hour limit or stop signal reached')
            workflow.update(status='complete', completed_at=utc_now())
        except (InterruptedError, KeyboardInterrupt) as exc:
            logging.warning('%s', exc)
            workflow['reason'] = str(exc)
            result = 2
        except Exception as exc:
            logging.exception('Local execution stopped')
            workflow['reason'] = str(exc)
            result = 1
        finally:
            workflow['invocation_ended_at'] = utc_now()
            workflow['elapsed_seconds'] = time.time() - invocation_start
            atomic_json(invocation_path, workflow)
            atomic_json(root / 'workflow_status.json', workflow)
            for handler in logging.getLogger().handlers:
                handler.flush()
            print(f"Status: {workflow['status']}\nArchive: {package_data(root, archives)}", flush=True)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description='Bounded laptop translation experiment; defaults to benchmark only')
    parser.add_argument('--mode', choices=['benchmark', 'run', 'package'], default='benchmark')
    parser.add_argument('--device', choices=['cpu', 'rocm'], default='cpu')
    parser.add_argument('--config', type=Path, default=project_root / 'config/experiment.toml')
    parser.add_argument('--root', type=Path, default=project_root / '.runtime')
    args = parser.parse_args()
    return execute(args)


if __name__ == '__main__':
    sys.exit(main())
