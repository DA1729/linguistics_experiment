from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd

from src.experiment import initial_state, make_requests, next_state, verify_records
from src.languages import derive_seed, make_schedule
from src.dataset import select_sentences
from src.package_data import verify_archive
from src.storage import digest
from .common import file_hash, save_json, save_table


def read_rows(archive, name):
    with archive.open(name) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def inspect_archive(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP member names')
        return [{'path': x.filename, 'bytes': x.file_size, 'compressed_bytes': x.compress_size} for x in archive.infolist()]


def compact(row):
    keys = ('sentence_id', 'round', 'kind', 'source_language', 'target_language', 'input_text',
            'output_text', 'status', 'model_invoked', 'error', 'input_tokens', 'output_tokens', 'full_input_tokens')
    result = {k: row.get(k) for k in keys}
    result['warnings'] = '|'.join(row.get('warnings', []))
    return result


def load_and_audit(config):
    path, output = Path(config['archive']), Path(config['output'])
    archive_hash = file_hash(path)
    sidecar = path.with_suffix('.zip.sha256')
    if sidecar.exists() and sidecar.read_text().split()[0] != archive_hash:
        raise ValueError('Download sidecar checksum mismatch')
    print('Checking archive CRC and every manifest SHA-256...', flush=True)
    manifest = verify_archive(path)
    inventory = inspect_archive(path)
    save_table(output, 'archive_inventory', inventory)
    discrepancies, run_summaries, status_counts = [], [], []
    observations, translations, states_by_round = [], [], {}
    def check(condition, detail):
        if not condition:
            discrepancies.append(detail)
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        read = lambda n: json.loads(archive.read(n))
        corpus = read_rows(archive, 'corpus/corpus.jsonl')
        ids = [x['sentence_id'] for x in corpus]
        check(len(ids) == len(set(ids)), 'Duplicate corpus identifiers')
        check(len({x['text'] for x in corpus}) == len(corpus), 'Duplicate original texts')
        provenance = read('corpus/provenance.json')
        check(hashlib.sha256(archive.read('corpus/source_english.devtest')).hexdigest() == provenance['source_sha256'], 'Source corpus hash')
        source_text = archive.read('corpus/source_english.devtest').decode('utf-8')
        check(select_sentences(source_text, len(corpus), provenance['selection_seed']) == corpus, 'Source sentence selection, IDs and line provenance')
        environment = read('environment.json')
        experiment_config = read('effective_config.json')
        pool = read('language_pool.json')
        run_ids = sorted(n.split('/')[1] for n in names if n.startswith('runs/') and n.endswith('/metadata.json'))
        for run_id in run_ids:
            print('Reconstructing and auditing', run_id, flush=True)
            base = f'runs/{run_id}'
            identity = read(base + '/metadata.json')['identity']
            schedules = read_rows(archive, base + '/schedules.jsonl')
            run_corpus = corpus[:identity['settings']['sentences']]
            schedule_map = {r['sentence_id']: r['languages'] for r in schedules}
            check(len(schedule_map) == len(schedules) == len(run_corpus), run_id + ': schedule identifiers')
            check(set(schedule_map) == {r['sentence_id'] for r in run_corpus}, run_id + ': schedule membership')
            check(digest(schedules) == identity['schedules_digest'], run_id + ': schedule digest')
            check(digest(run_corpus) == identity['corpus_digest'], run_id + ': corpus digest')
            check(digest(environment) == identity['environment_digest'], run_id + ': environment digest')
            for row in schedules:
                check(row['seed'] == derive_seed(identity['seed'], row['sentence_id']), run_id + ': schedule seed derivation')
                langs = row['languages']
                check(langs == make_schedule(identity['language_pool'], identity['settings']['rounds'], row['seed']), run_id + ': seeded schedule mismatch ' + row['sentence_id'])
                check(all(a != b for a, b in zip(langs, langs[1:])), run_id + ': adjacent language repeat')
                size = len(identity['language_pool'])
                for offset in range(0, len(langs) - size + 1, size):
                    check(set(langs[offset:offset+size]) == set(identity['language_pool']), run_id + ': invalid full permutation')
            previous = initial_state(run_corpus)
            completed = sorted(int(n.split('/')[-2][6:]) for n in names if n.startswith(base + '/rounds/round_') and n.endswith('/checkpoint.json'))
            check(completed == list(range(identity['settings']['rounds'] + 1)), run_id + ': missing rounds')
            cumulative = 0
            counts = Counter()
            actual_observations = []
            for t in completed:
                prefix = f'{base}/rounds/round_{t:06d}/'
                for filename, expected in read(prefix + 'manifest.json').items():
                    check(manifest['files'].get(prefix + filename) == expected, prefix + filename + ': nested manifest mismatch')
                current = read_rows(archive, prefix + 'translations.jsonl')
                diagnostic = read_rows(archive, prefix + 'observations.jsonl')
                journal = read_rows(archive, prefix + 'journal.jsonl')
                canonical = current + diagnostic
                check(Counter((r['kind'], r['sentence_id'], r['record_digest']) for r in canonical) == Counter((r['kind'], r['sentence_id'], r['record_digest']) for r in journal), prefix + ': journal mismatch')
                for kind, rows, before in [('translation', current, previous)]:
                    requests = make_requests(before, run_corpus, schedule_map, t, kind, run_id, identity['seed'], identity['experiment_id']) if t else []
                    try:
                        verify_records(rows, requests)
                    except (ValueError, KeyError, TypeError) as exc:
                        discrepancies.append(prefix + kind + ': ' + str(exc))
                if current:
                    previous = next_state(previous, current)
                requests = make_requests(previous, run_corpus, schedule_map, t, 'observation', run_id, identity['seed'], identity['experiment_id']) if t in identity['observation_rounds'] else []
                try:
                    verify_records(diagnostic, requests)
                except (ValueError, KeyError, TypeError) as exc:
                    discrepancies.append(prefix + 'observation: ' + str(exc))
                checkpoint = read(prefix + 'checkpoint.json')
                check(checkpoint['round'] == t and checkpoint['states'] == previous, prefix + ': checkpoint continuity')
                check(checkpoint['identity_digest'] == digest(identity), prefix + ': identity digest')
                cumulative += sum(r['model_invoked'] for r in canonical)
                check(cumulative == checkpoint['model_translations'], prefix + ': translation counter')
                if diagnostic:
                    actual_observations.append(t)
                for kind, rows in [('translation', current), ('observation', diagnostic)]:
                    check(len({r['sentence_id'] for r in rows}) == len(rows), prefix + kind + ': duplicate records')
                    for r in rows:
                        counts[kind + '_records'] += 1
                        counts[kind + '_valid'] += int(r['status'] == 'ok' and isinstance(r['output_text'], str) and bool(r['output_text'].strip()))
                        counts['model_invoked'] += int(r['model_invoked'])
                        counts['status_' + r['status']] += 1
                        counts['empty_outputs'] += int(isinstance(r['output_text'], str) and not r['output_text'].strip())
                        for flag in r.get('warnings', []):
                            counts['warning_' + flag] += 1
                        status_counts.append((run_id, t, kind, r['status'], '|'.join(r.get('warnings', []))))
                if run_id == config['run_id']:
                    states_by_round[t] = {s['sentence_id']: (s['language'], s['text'], s['active']) for s in previous}
                    translations.extend(compact(r) for r in current)
                    observations.extend(compact(r) for r in diagnostic)
            status = read(base + '/status.json')
            check(status['model_translations'] == cumulative, run_id + ': final model count')
            check(status['last_completed_round'] == max(completed), run_id + ': final round')
            check(status['active_trajectories'] == sum(s['active'] for s in previous), run_id + ': active population')
            check(actual_observations == identity['observation_rounds'], run_id + ': missing observations')
            run_summaries.append({'run_id': run_id, 'trajectories': len(run_corpus), 'schedules': len(schedules), 'checkpoints': len(completed),
                                  'observation_rounds': actual_observations, 'status': status['status'], **counts})
            if run_id == config['run_id']:
                main_identity, main_schedules = identity, schedules
        audit = {'archive_sha256': archive_hash, 'sidecar_verified': sidecar.exists(), 'manifest_files': len(manifest['files']),
                 'zip_members': len(inventory), 'archive_status': manifest['status'], 'corpus_sentences': len(corpus),
                 'unique_originals': len({r['text'] for r in corpus}), 'source_lines': len(source_text.splitlines()), 'runs': run_summaries,
                 'pending_files': sum('/pending_' in n for n in names), 'discrepancies': discrepancies,
                 'provenance': provenance, 'language_pool': pool, 'model': {k: environment[k] for k in ('model_name','model_revision','precision','torch','transformers','gpu')},
                 'observation_rounds': main_identity['observation_rounds'],
                 'limitations': ['Recorded ok means the collector accepted output, not that a human validated its meaning.',
                                'Language identification and multilingual semantic correctness are not independently validated.',
                                'Round-zero English observations are copies and do not invoke the model.']}
    save_json(output / 'data_quality_summary.json', audit)
    save_table(output, 'data_quality_summary', [{**r, 'observation_rounds': json.dumps(r['observation_rounds'])} for r in run_summaries])
    save_json(output / 'data_discrepancies.json', discrepancies)
    counts = pd.DataFrame(status_counts, columns=['run_id','round','kind','status','warnings']).value_counts(dropna=False).reset_index(name='records')
    save_table(output, 'record_status_counts', counts)
    if discrepancies:
        raise ValueError(f'{len(discrepancies)} reconstruction discrepancies: see data_discrepancies.json; primary analysis stopped rather than silently use inconsistent states')
    return {'audit': audit, 'corpus': pd.DataFrame(corpus), 'observations': pd.DataFrame(observations),
            'translations': pd.DataFrame(translations), 'states': states_by_round,
            'schedules': main_schedules, 'identity': main_identity, 'config': experiment_config}
