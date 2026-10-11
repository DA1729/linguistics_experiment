import argparse
import copy
import logging
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
                      read_json, read_jsonl, utc_now, write_jsonl)

project_root = Path(__file__).resolve().parent.parent


def read_config(path: Path) -> dict:
    with path.open('rb') as stream:
        config = tomllib.load(stream)
    with (project_root / 'config/experiment.toml').open('rb') as stream:
        template = tomllib.load(stream)
    if set(config) != set(template) or any(set(config[k]) != set(template[k]) for k in template):
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
        if any(type(r) is not int or r < 0 or r > config[phase]['rounds'] for r in config[phase]['observation_rounds']):
            raise ValueError(f'{phase}.observation_rounds must lie within the experiment')
    if config['smoke']['repetitions'] != 1 or config['smoke']['rounds'] < 2 or config['smoke']['sentences'] < 2:
        raise ValueError('Smoke checks require one repetition, at least two rounds and two sentences')
    if config['pilot']['rounds'] < 1:
        raise ValueError('Pilot needs at least one round for throughput measurement')
    model = config['model']
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
    return sorted((project_root / 'src').glob('*.py')) + [project_root / name for name in ('run_pod.sh', 'requirements.txt')]


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
        for p in source_files() + [project_root / 'README.md', project_root / 'ATTRIBUTION.md']:
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
        logging.info('Saved GPU smoke test already passed')
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
        raise RuntimeError('GPU smoke probes returned failures or limit warnings; inspect smoke probe data')
    matches = [a['output_text'] == b['output_text'] for a, b in zip(mixed, single)]
    if not all(matches):
        logging.warning('GPU smoke: mixed/single greedy outputs differ. Raw comparisons retained; floating point batching is not bitwise invariant.')
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


def execute(args) -> int:
    config = read_config(args.config)
    root = args.root.resolve() / 'data' / config['experiment']['name']
    archive_directory = args.root.resolve() / 'archives'
    if args.mode == 'package' and not root.exists():
        raise ValueError(f'No experiment data to package: {root}')
    with exclusive_lock(root):
        if args.mode == 'package':
            copy_bootstrap_log(args.root, root)
            archive = package_data(root, archive_directory)
            print(f'Archive: {archive}', flush=True)
            return 0
        log_directory = root / 'logs'
        log_directory.mkdir(exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                            handlers=[logging.StreamHandler(), logging.FileHandler(log_directory / 'execution.log', encoding='utf-8')], force=True)
        stopped = False

        def handle_signal(signum, frame):
            nonlocal stopped
            stopped = True
            logging.warning('Signal %s received; preserving batch journal and checkpoint', signum)

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)
        deadline = (args.started_at or time.time()) + config['runtime']['max_hours'] * 3600

        def should_stop():
            return stopped or time.time() >= deadline

        workflow = {'status': 'incomplete', 'requested_mode': args.mode, 'invocation_started_at': utc_now()}
        exit_code = 0
        try:
            initialize(root, config, args.config)
            atomic_json(root / 'workflow_status.json', workflow)
            pool = select_pool(config['experiment'])
            atomic_json(root / 'language_pool.json', [{'code': code, 'name': language_names.get(code, code)} for code in pool])
            count = max(config[phase]['sentences'] for phase in ('smoke', 'pilot', 'main'))
            corpus, provenance = prepare_corpus(root / 'corpus', config['dataset'], count, config['experiment']['selection_seed'])
            if should_stop():
                raise InterruptedError('Deadline reached during setup')
            from .translation import load_translator
            logging.info('Loading pinned model on CUDA; precision=%s', config['model']['precision'])
            translate, environment, engine = load_translator(config['model'], pool, config['experiment']['seed'])
            env_path = root / 'environment.json'
            if env_path.exists() and read_json(env_path) != environment:
                raise ValueError('Model/runtime environment changed; restore the original environment to resume')
            atomic_json(env_path, environment)
            logging.info('GPU=%s VRAM=%.1f GiB', environment['gpu'], environment['gpu_memory_bytes'] / 2**30)
            smoke_check(root, config, corpus, pool, environment, translate, should_stop)
            phases = [] if args.mode == 'smoke' else ['pilot']
            if args.mode in ('all', 'main', 'resume'):
                phases.append('main')
            pilot_results = []
            for phase in phases:
                if phase == 'main':
                    estimate = estimate_main(config, pilot_results)
                    atomic_json(root / 'runtime_estimate.json', estimate)
                    logging.info('Main estimate: %.2f hours, ~$%.2f at assumed $%.3f/hour; session time left %.2f hours. This is not a billing limit.',
                                 estimate['main_hours_estimate'], estimate['main_gpu_cost_usd_estimate'],
                                 estimate['assumed_gpu_hourly_usd'], max(0, deadline - time.time()) / 3600)
                for repetition in range(1, config[phase]['repetitions'] + 1):
                    status = run_experiment(root / 'runs' / f'{phase}_{repetition:03d}', config, phase, repetition,
                                            corpus, pool, environment, translate, should_stop)
                    if phase == 'pilot':
                        pilot_results.append(status)
                    if status['status'] != 'complete':
                        raise InterruptedError('Runtime limit or interruption; resume with the same command')
                    logging.info('GPU allocated memory %.2f GiB', engine['torch'].cuda.memory_allocated() / 2**30)
            workflow['status'] = 'complete'
            workflow['completed_at'] = utc_now()
        except (InterruptedError, KeyboardInterrupt) as exc:
            logging.warning('%s', exc)
            workflow['reason'] = str(exc) or 'interrupted'
            exit_code = 2
        except Exception as exc:
            logging.exception('Execution stopped; retained data remain available')
            workflow['reason'] = str(exc)
            exit_code = 1
        finally:
            workflow['invocation_ended_at'] = utc_now()
            atomic_json(root / 'workflow_status.json', workflow)
            logging.info('Invocation status: %s', workflow['status'])
            for handler in logging.getLogger().handlers:
                handler.flush()
            copy_bootstrap_log(args.root, root)
            archive = package_data(root, archive_directory)
            print(f"Status: {workflow['status']}\nArchive: {archive}\nStop the RunPod Pod when finished to stop GPU billing.", flush=True)
        return exit_code


def copy_bootstrap_log(runtime_root: Path, root: Path) -> None:
    path = runtime_root / 'logs/bootstrap.log'
    if path.exists():
        atomic_bytes(root / 'logs/bootstrap.log', path.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser(description='Repeated multilingual translation data collection')
    parser.add_argument('--mode', choices=['all', 'smoke', 'pilot', 'main', 'resume', 'package'], default='all')
    parser.add_argument('--config', type=Path, default=project_root / 'config/experiment.toml')
    parser.add_argument('--root', type=Path, default=project_root / '.runtime')
    parser.add_argument('--started-at', type=float)
    args = parser.parse_args()
    return execute(args)


if __name__ == '__main__':
    sys.exit(main())
