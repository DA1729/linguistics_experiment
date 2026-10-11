import logging
from pathlib import Path
import time

from .languages import derive_seed, make_schedule, observation_rounds
from .storage import (append_rows, atomic_json, digest, read_json, read_jsonl, recover_journal,
                      seal_round, utc_now, verify_round, write_jsonl)


def initial_state(corpus: list[dict]) -> list[dict]:
    return [{'sentence_id': row['sentence_id'], 'text': row['text'], 'language': 'eng_Latn',
             'active': True, 'termination_reason': None} for row in corpus]


def make_requests(states: list[dict], corpus: list[dict], schedules: dict, round_index: int,
                  kind: str, run_id: str, seed: int, experiment_id: str) -> list[dict]:
    requests = []
    for state, original in zip(states, corpus):
        target = 'eng_Latn' if kind == 'observation' else schedules[state['sentence_id']][round_index - 1]
        requests.append({
            'experiment_id': experiment_id, 'run_id': run_id, 'seed': seed,
            'sentence_id': state['sentence_id'], 'round': round_index, 'kind': kind,
            'source_language': state['language'], 'target_language': target,
            'input_text': state['text'], 'original_sentence': original['text'],
            'trajectory_active': state['active'], 'termination_reason': state['termination_reason'],
        })
    return requests


def validate_record(row: dict, request: dict) -> None:
    if row.get('record_digest') != digest({key: value for key, value in row.items() if key != 'record_digest'}):
        raise ValueError('Translation record checksum mismatch')
    if row.get('request_digest') != digest(request) or any(row.get(k) != v for k, v in request.items()):
        raise ValueError('Translation record does not match trajectory or language schedule')
    if row.get('status') not in {'ok', 'failed', 'skipped'}:
        raise ValueError('Invalid translation status')
    if row['status'] == 'ok' and (not isinstance(row.get('output_text'), str) or not row['output_text'].strip()):
        raise ValueError('Successful translation must contain nonempty text')
    if request['trajectory_active'] == (row['status'] == 'skipped'):
        raise ValueError('Translation status inconsistent with active trajectory')


def next_state(states: list[dict], records: list[dict]) -> list[dict]:
    next_states = []
    for state, record in zip(states, records):
        if record['status'] == 'skipped':
            next_states.append(dict(state))
        else:
            active = record['status'] == 'ok'
            next_states.append({
                'sentence_id': state['sentence_id'], 'text': record['output_text'],
                'language': record['target_language'], 'active': active,
                'termination_reason': None if active else record.get('error') or 'translation_failed',
            })
    return next_states


def collect(requests: list[dict], journal: Path, saved: dict, translate, should_stop,
            progress_seconds: float) -> list[dict]:
    missing = []
    for request in requests:
        key = (request['kind'], request['sentence_id'])
        if key in saved:
            validate_record(saved[key], request)
        elif not request['trajectory_active'] or request['round'] == 0:
            status = 'ok' if request['trajectory_active'] else 'skipped'
            row = {**request, 'request_digest': digest(request), 'output_text': request['input_text'] if status == 'ok' else None,
                   'status': status, 'warnings': [], 'error': request['termination_reason'],
                   'collection_seconds': 0.0, 'model_invoked': False}
            if request['kind'] == 'observation':
                row['english_translation'] = row['output_text']
                row['current_sentence'] = request['input_text']
            row['record_digest'] = digest(row)
            append_rows(journal, [row])
            saved[key] = row
        else:
            missing.append(request)
    request_by_id = {row['sentence_id']: row for row in missing}
    last_batch = time.monotonic()
    last_report = last_batch
    completed = 0

    def persist_batch(items, results):
        nonlocal last_batch, last_report, completed
        now = time.monotonic()
        rows = []
        for item, result in zip(items, results):
            request = request_by_id[item['sentence_id']]
            row = {**result, **request, 'request_digest': digest(request), 'model_invoked': True,
                   'collection_seconds': (now - last_batch) / len(items)}
            if request['kind'] == 'observation':
                row['english_translation'] = result['output_text']
                row['current_sentence'] = request['input_text']
            row['record_digest'] = digest(row)
            validate_record(row, request)
            rows.append(row)
        append_rows(journal, rows)
        for row in rows:
            saved[(row['kind'], row['sentence_id'])] = row
        completed += len(rows)
        if now - last_report >= progress_seconds:
            memory = max(row.get('gpu_memory_allocated_bytes', 0) for row in rows) / 2**30
            logging.info('%s round=%s %s: %s/%s pending translations saved; GPU memory=%.2f GiB',
                         requests[0]['run_id'], requests[0]['round'], requests[0]['kind'], completed, len(missing), memory)
            last_report = now
        last_batch = now

    items = [{'sentence_id': row['sentence_id'], 'text': row['input_text'],
              'source_language': row['source_language'], 'target_language': row['target_language']} for row in missing]
    if items:
        try:
            translate(items, on_batch=persist_batch, should_stop=should_stop)
        except InterruptedError:
            raise
        except Exception as exc:
            failed_items = getattr(exc, 'translation_inputs', None)
            attempts = failed_items if failed_items is not None else [
                item for item in items if (requests[0]['kind'], item['sentence_id']) not in saved]
            append_rows(journal.parent / 'failure_attempts.jsonl', [{
                **request_by_id[item['sentence_id']], 'status': 'exception', 'output_text': None,
                'error': f'{type(exc).__name__}: {exc}', 'attempted_at': utc_now(),
                'scope': 'failed_batch' if failed_items is not None else 'unfinished_request_set',
                'note': 'No successful result; run halted. Retry is allowed on resume.'} for item in attempts])
            raise
    rows = [saved[(row['kind'], row['sentence_id'])] for row in requests]
    for row in rows:
        if row['kind'] == 'observation':
            row['english_translation'] = row['output_text']
            row['current_sentence'] = row['input_text']
    return rows


def verify_records(records: list[dict], requests: list[dict]) -> None:
    if len(records) != len(requests):
        raise ValueError('Wrong number of records in completed round')
    for row, request in zip(records, requests):
        validate_record(row, request)
        if request['kind'] == 'observation' and (row.get('english_translation') != row['output_text'] or
                                                row.get('current_sentence') != row['input_text']):
            raise ValueError('English observation fields inconsistent')


def run_experiment(directory: Path, config: dict, phase: str, repetition: int, corpus: list[dict],
                   pool: list[str], environment: dict, translate, should_stop=lambda: False,
                   stop_after_round=None) -> dict:
    settings = config[phase]
    corpus = corpus[:settings['sentences']]
    run_id = f'{phase}_{repetition:03d}'
    seed = derive_seed(config['experiment']['seed'], phase, repetition)
    schedule_rows = [{'sentence_id': row['sentence_id'],
                      'seed': derive_seed(seed, row['sentence_id']),
                      'languages': make_schedule(pool, settings['rounds'], derive_seed(seed, row['sentence_id']))}
                     for row in corpus]
    schedules = {row['sentence_id']: row['languages'] for row in schedule_rows}
    observations = observation_rounds(settings)
    identity = {'schema_version': 1, 'run_id': run_id, 'seed': seed, 'phase': phase,
                'experiment_id': config['experiment']['name'], 'settings': settings,
                'model': config['model'], 'language_pool': pool, 'corpus_digest': digest(corpus),
                'schedules_digest': digest(schedule_rows), 'environment_digest': digest(environment),
                'observation_rounds': observations}
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / 'metadata.json'
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        if metadata['identity'] != identity or read_jsonl(directory / 'schedules.jsonl') != schedule_rows:
            raise ValueError(f'Configuration, environment, corpus or schedules changed for {run_id}')
    else:
        write_jsonl(directory / 'schedules.jsonl', schedule_rows)
        metadata = {'identity': identity, 'started_at': utc_now()}
        atomic_json(metadata_path, metadata)
    rounds_path = directory / 'rounds'
    rounds_path.mkdir(exist_ok=True)
    states = initial_state(corpus)
    last_round, seconds, translations = -1, 0.0, 0
    completed_dirs = sorted(rounds_path.glob('round_*'))
    for expected_round, path in enumerate(completed_dirs):
        if path.name != f'round_{expected_round:06d}' or expected_round > settings['rounds']:
            raise ValueError(f'Noncontiguous or unexpected round: {path}')
        checkpoint = verify_round(path)
        actual = read_jsonl(path / 'translations.jsonl')
        diagnostic = read_jsonl(path / 'observations.jsonl')
        requests = make_requests(states, corpus, schedules, expected_round, 'translation', run_id, seed, identity['experiment_id']) if expected_round else []
        verify_records(actual, requests)
        if actual:
            states = next_state(states, actual)
        requests = make_requests(states, corpus, schedules, expected_round, 'observation', run_id, seed, identity['experiment_id']) if expected_round in observations else []
        verify_records(diagnostic, requests)
        if checkpoint['states'] != states or checkpoint['round'] != expected_round or checkpoint['identity_digest'] != digest(identity):
            raise ValueError('Checkpoint and committed records disagree')
        last_round = expected_round
        seconds = checkpoint['collection_seconds']
        translations = checkpoint['model_translations']
    status = {'run_id': run_id, 'status': 'incomplete', 'last_completed_round': last_round,
              'total_rounds': settings['rounds'], 'started_at': metadata['started_at']}
    atomic_json(directory / 'status.json', status)
    for round_index in range(last_round + 1, settings['rounds'] + 1):
        if should_stop():
            break
        pending = rounds_path / f'pending_{round_index:06d}'
        pending.mkdir(exist_ok=True)
        journal = pending / 'journal.jsonl'
        saved = {}
        recover_journal(pending / 'failure_attempts.jsonl')
        for row in recover_journal(journal):
            key = (row['kind'], row['sentence_id'])
            if key in saved or row['round'] != round_index:
                raise ValueError('Duplicate or misplaced journal record')
            saved[key] = row
        requests = make_requests(states, corpus, schedules, round_index, 'translation', run_id, seed, identity['experiment_id']) if round_index else []
        try:
            actual = collect(requests, journal, saved, translate, should_stop, config['runtime']['progress_seconds'])
            proposed = next_state(states, actual) if actual else states
            requests = make_requests(proposed, corpus, schedules, round_index, 'observation', run_id, seed, identity['experiment_id']) if round_index in observations else []
            diagnostic = collect(requests, journal, saved, translate, should_stop, config['runtime']['progress_seconds'])
        except InterruptedError:
            break
        expected_keys = {(row['kind'], row['sentence_id']) for row in actual + diagnostic}
        if set(saved) != expected_keys:
            raise ValueError('Unexpected journal entries')
        write_jsonl(pending / 'translations.jsonl', actual)
        write_jsonl(pending / 'observations.jsonl', diagnostic)
        seconds += sum(row.get('collection_seconds', 0) for row in actual + diagnostic)
        translations += sum(row['model_invoked'] for row in actual + diagnostic)
        checkpoint = {'round': round_index, 'states': proposed, 'identity_digest': digest(identity),
                      'collection_seconds': seconds, 'model_translations': translations, 'committed_at': utc_now()}
        seal_round(pending, rounds_path / f'round_{round_index:06d}', checkpoint)
        states, last_round = proposed, round_index
        status['last_completed_round'] = round_index
        atomic_json(directory / 'status.json', status)
        remaining = settings['sentences'] * (settings['rounds'] - round_index + sum(r > round_index for r in observations))
        rate = translations / seconds if seconds else 0
        logging.info('%s round=%s/%s translations=%s rate=%.2f/s collection_elapsed=%.1fs remaining_estimate=%.1fs active=%s',
                     run_id, round_index, settings['rounds'], translations, rate, seconds,
                     remaining / rate if rate else 0, sum(state['active'] for state in states))
        if stop_after_round is not None and round_index >= stop_after_round:
            break
    complete = last_round == settings['rounds']
    status.update(status='complete' if complete else 'incomplete', last_completed_round=last_round,
                  collection_seconds=seconds, model_translations=translations,
                  active_trajectories=sum(state['active'] for state in states))
    if complete:
        status['completed_at'] = read_json(rounds_path / f'round_{last_round:06d}' / 'checkpoint.json')['committed_at']
    atomic_json(directory / 'status.json', status)
    return status
