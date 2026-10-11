import ast
import copy
import functools
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from src.dataset import corpus_hash, prepare_corpus, select_sentences
from src.experiment import run_experiment
from src.languages import derive_seed, language_names, make_schedule, observation_rounds, select_pool
from src.package_data import package_data, verify_archive
from src.runner import estimate_main, initialize, read_config
from src.storage import (append_rows, atomic_json, file_hash, read_json, read_jsonl,
                         recover_journal, verify_round, write_jsonl, exclusive_lock)
from src.translation import (batch_indices, encode_inputs, resilient_batch, suspicious_output,
                             translate_batches, validate_languages)

project_root = Path(__file__).resolve().parent.parent


class fake_tokenizer:
    def __init__(self):
        self.src_lang = 'eng_Latn'
        self.codes = {'eng_Latn': 10, 'fra_Latn': 11, 'deu_Latn': 12, 'arb_Arab': 13}
        self.all_special_ids = [0, 2, *self.codes.values()]
        self.unk_token_id = 0
        self.eos_token_id = 2

    def convert_tokens_to_ids(self, token):
        return self.codes.get(token, 0)

    def convert_ids_to_tokens(self, token):
        return next((key for key, value in self.codes.items() if value == token), '<unk>')

    def __call__(self, text, truncation=False, max_length=None, add_special_tokens=True):
        ids = [100 + ord(c) for c in text]
        if truncation:
            ids = ids[:max_length - 2]
        ids = [self.codes[self.src_lang], *ids, 2]
        return {'input_ids': ids, 'attention_mask': [1] * len(ids)}


def mock_translate(items, on_batch=None, should_stop=None, calls=None, failures=None, interrupt_at=None):
    def batch_function(batch, target):
        if interrupt_at is not None and len(calls) >= interrupt_at:
            raise InterruptedError('Simulated interruption')
        results = []
        for item in batch:
            if calls is not None:
                calls.append(dict(item))
            failed = item['sentence_id'] in (failures or set())
            results.append({'sentence_id': item['sentence_id'],
                            'output_text': '' if failed else f"{target}({item['text']})",
                            'status': 'failed' if failed else 'ok', 'warnings': ['empty_output'] if failed else [],
                            'error': 'empty_output' if failed else None})
        return results
    return translate_batches(items, 2, batch_function, on_batch, should_stop)


class project_tests(unittest.TestCase):
    def run(self, result=None):
        self.set_up()
        return super().run(result)

    def set_up(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = read_config(project_root / 'config/experiment.toml')
        for phase in ('smoke', 'pilot', 'main'):
            self.config[phase].update(sentences=5, rounds=3, repetitions=1, observe_every=1, observation_rounds=[])
        self.pool = ['fra_Latn', 'deu_Latn', 'arb_Arab']
        self.corpus = select_sentences('one\ntwo\nthree\nfour\nfive\nsix\n', 5, 3)
        self.env = {'model_revision': 'test_only'}
        self.calls = []
        self.translate = functools.partial(mock_translate, calls=self.calls)
        self.run_dir = self.root / 'runs/pilot_001'

    def run_experiment(self, **overrides):
        args = dict(directory=self.run_dir, config=self.config, phase='pilot', repetition=1,
                    corpus=self.corpus, pool=self.pool, environment=self.env, translate=self.translate)
        args.update(overrides)
        return run_experiment(**args)

    def records(self, kind='translations'):
        return [row for p in sorted(self.run_dir.glob(f'rounds/round_*/{kind}.jsonl')) for row in read_jsonl(p)]

    def test_schedule_permutations_and_boundaries(self):
        for seed in range(100):
            path = make_schedule(self.pool, 303, seed)
            self.assertEqual(len(path), 303)
            self.assertTrue(all(a != b for a, b in zip(path, path[1:])))
            for start in range(0, 303, 3):
                self.assertEqual(set(path[start:start + 3]), set(self.pool))

    def test_schedule_reproducible_independent_streams(self):
        a = derive_seed(123, 'pilot', 1, 's_1')
        b = derive_seed(123, 'pilot', 1, 's_2')
        self.assertNotEqual(a, b)
        self.assertEqual(make_schedule(self.pool, 99, a), make_schedule(self.pool, 99, a))
        self.assertNotEqual(make_schedule(self.pool, 99, a), make_schedule(self.pool, 99, b))
        self.assertEqual(make_schedule(self.pool, 0, a), [])

    def test_invalid_pool(self):
        for pool in [[], ['fra_Latn'], ['fra_Latn', 'fra_Latn'], ['eng_Latn', 'fra_Latn']]:
            with self.assertRaises(ValueError):
                select_pool({'languages': pool, 'language_count': 1})
        with self.assertRaises(ValueError):
            make_schedule(['x'], 10, 1)
        self.assertEqual(len(language_names), 50)

    def test_default_observation_rounds(self):
        settings = read_config(project_root / 'config/experiment.toml')['main']
        self.assertEqual(observation_rounds(settings), [0, 1, 2, 3, *range(5, settings["rounds"] + 1, 5)])
        settings.update(rounds=17, observation_rounds=[0], observe_every=0)
        self.assertEqual(observation_rounds(settings), [0, 17])

    def test_corpus_exact_unique_reproducible(self):
        raw = ' alpha \n\nÉlan!\n alpha \nBeta\r\n'
        a = select_sentences(raw, 3, 7)
        self.assertEqual(a, select_sentences(raw, 3, 7))
        self.assertEqual({r['text'] for r in a}, {' alpha ', 'Élan!', 'Beta'})
        self.assertEqual(len({r['sentence_id'] for r in a}), 3)
        with self.assertRaises(ValueError):
            select_sentences(raw, 4, 7)

    def test_custom_corpus_provenance_and_change_detection(self):
        path = self.root / 'custom.txt'
        path.write_text('first\nsecond\nthird\n', encoding='utf-8')
        config = {'local_file': str(path), 'local_license': 'test'}
        rows, provenance = prepare_corpus(self.root / 'corpus', config, 2, 42)
        self.assertEqual(provenance['source_sha256'], file_hash(path))
        self.assertEqual(len(rows), 2)
        self.assertEqual(prepare_corpus(self.root / 'corpus', config, 2, 42)[0], rows)
        path.write_text('first\nCHANGED\nthird\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            prepare_corpus(self.root / 'corpus', config, 2, 42)

    def test_download_checksum_rejected(self):
        path = self.root / 'corpus'
        path.mkdir()
        (path / 'source_english.devtest').write_text('wrong corpus', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            prepare_corpus(path, {'local_file': '', 'local_license': ''}, 1, 1)

    def test_mixed_source_encoding_isolated_and_truncation_flagged(self):
        tokenizer = fake_tokenizer()
        items = [{'text': 'long sentence', 'source_language': 'fra_Latn'},
                 {'text': '短', 'source_language': 'arb_Arab'}]
        rows = encode_inputs(tokenizer, items, 6)
        self.assertEqual(rows[0]['input_ids'][0], 11)
        self.assertEqual(rows[1]['input_ids'][0], 13)
        self.assertEqual(rows[0]['input_ids'][-1], 2)
        self.assertEqual(rows[0]['warnings'], ['input_truncated'])
        self.assertEqual(rows[1]['warnings'], [])
        tokenizer.src_lang = 'eng_Latn'
        self.assertEqual(rows[0]['input_ids'][0], 11)

    def test_language_tokens_validation(self):
        tokenizer = fake_tokenizer()
        validate_languages(tokenizer, ['eng_Latn', 'fra_Latn'])
        with self.assertRaises(ValueError):
            validate_languages(tokenizer, ['fake_Latn'])

    def test_reject_wrong_source_prefix(self):
        tokenizer = fake_tokenizer()
        tokenizer.convert_tokens_to_ids = lambda code: 999
        with self.assertRaisesRegex(RuntimeError, 'layout'):
            encode_inputs(tokenizer, [{'text': 'hi', 'source_language': 'fra_Latn'}], 20)

    def test_target_batches_restore_input_order(self):
        items = [{'sentence_id': str(i), 'text': f'text{i}', 'source_language': self.pool[i % 3],
                  'target_language': self.pool[(i + 1) % 3]} for i in range(11)]
        batches = list(batch_indices(items, 2))
        self.assertTrue(all(len(ids) <= 2 and all(items[i]['target_language'] == target for i in ids) for target, ids in batches))
        result = self.translate(items)
        self.assertEqual([r['sentence_id'] for r in result], [r['sentence_id'] for r in items])
        self.assertTrue(all(r['output_text'].startswith(item['target_language']) for item, r in zip(items, result)))

    def test_bad_backend_mapping_rejected(self):
        items = [{'sentence_id': 'a', 'text': 'x', 'source_language': 'fra_Latn', 'target_language': 'deu_Latn'}]
        with self.assertRaises(RuntimeError):
            translate_batches(items, 2, lambda batch, target: [{'sentence_id': 'wrong'}])
        with self.assertRaises(RuntimeError):
            translate_batches(items, 2, lambda batch, target: [])
        with self.assertRaises(ValueError):
            translate_batches(items * 2, 2, lambda batch, target: [])

    def test_observations_never_replace_states(self):
        status = self.run_experiment()
        self.assertEqual(status['status'], 'complete')
        records = self.records()
        observations = self.records('observations')
        self.assertEqual(len(records), 15)
        self.assertEqual(len(observations), 20)
        previous = {r['sentence_id']: r['text'] for r in self.corpus}
        previous_language = {r['sentence_id']: 'eng_Latn' for r in self.corpus}
        schedules = {r['sentence_id']: r['languages'] for r in read_jsonl(self.run_dir / 'schedules.jsonl')}
        for row in records:
            sid = row['sentence_id']
            self.assertEqual(row['input_text'], previous[sid])
            self.assertEqual(row['source_language'], previous_language[sid])
            self.assertEqual(row['target_language'], schedules[sid][row['round'] - 1])
            self.assertNotIn('eng_Latn(', row['input_text'])
            previous[sid] = row['output_text']
            previous_language[sid] = row['target_language']
        self.assertTrue(all(r['target_language'] == 'eng_Latn' for r in observations))
        self.assertTrue(all(not r['model_invoked'] for r in observations if r['round'] == 0))

    def test_round_checkpoint_resume_no_recompute_or_duplicates(self):
        first = self.run_experiment(stop_after_round=1)
        hashes = {p: file_hash(p) for p in self.run_dir.glob('rounds/round_*/*')}
        count = len(self.calls)
        second = self.run_experiment()
        self.assertEqual(first['status'], 'incomplete')
        self.assertEqual(second['status'], 'complete')
        self.assertEqual(len(self.calls) - count, 20)
        self.assertTrue(all(file_hash(p) == h for p, h in hashes.items()))
        count = len(self.calls)
        self.run_experiment()
        self.assertEqual(len(self.calls), count)
        keys = [(r['round'], r['sentence_id']) for r in self.records()]
        self.assertEqual(len(keys), len(set(keys)))

    def test_partial_batch_journal_resume_and_torn_tail(self):
        translate = functools.partial(mock_translate, calls=self.calls, interrupt_at=2)
        self.assertEqual(self.run_experiment(translate=translate)['status'], 'incomplete')
        journal = self.run_dir / 'rounds/pending_000001/journal.jsonl'
        durable = read_jsonl(journal)
        self.assertTrue(durable)
        with journal.open('ab') as stream:
            stream.write(b'{"partial":')
        status = self.run_experiment()
        self.assertEqual(status['status'], 'complete')
        self.assertEqual(len(self.calls), 30)
        self.assertTrue(list(self.run_dir.glob('rounds/round_000001/*.torn_*')))
        final = self.records()
        for row in durable:
            self.assertIn(row, final)

    def test_interrupt_during_observations_resume(self):
        translate = functools.partial(mock_translate, calls=self.calls, interrupt_at=6)
        self.run_experiment(translate=translate)
        self.assertGreaterEqual(len(self.calls), 6)
        self.run_experiment()
        self.assertEqual(len(self.calls), 30)

    def test_crash_before_atomic_round_rename(self):
        with mock.patch('src.experiment.seal_round', side_effect=RuntimeError('crash')):
            with self.assertRaises(RuntimeError):
                self.run_experiment()
        self.assertEqual(self.run_experiment()['status'], 'complete')
        self.assertEqual(len(self.calls), 30)

    def test_corrupt_round_rejected(self):
        self.run_experiment(stop_after_round=1)
        path = self.run_dir / 'rounds/round_000001/observations.jsonl'
        with path.open('ab') as stream:
            stream.write(b'{}\n')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.run_experiment()

    def test_configuration_and_environment_mismatch_rejected(self):
        self.run_experiment(stop_after_round=1)
        changed = copy.deepcopy(self.config)
        changed['model']['batch_size'] += 1
        with self.assertRaises(ValueError):
            self.run_experiment(config=changed)
        with self.assertRaises(ValueError):
            self.run_experiment(environment={'different': 'runtime'})

    def test_schedules_tampering_rejected(self):
        self.run_experiment(stop_after_round=1)
        path = self.run_dir / 'schedules.jsonl'
        rows = read_jsonl(path)
        rows[0]['languages'][0] = 'wrong_Latn'
        write_jsonl(path, rows)
        with self.assertRaises(ValueError):
            self.run_experiment()

    def test_failure_terminates_without_replacement(self):
        sid = self.corpus[0]['sentence_id']
        translate = functools.partial(mock_translate, calls=self.calls, failures={sid})
        self.run_experiment(translate=translate)
        rows = [r for r in self.records() if r['sentence_id'] == sid]
        self.assertEqual([r['status'] for r in rows], ['failed', 'skipped', 'skipped'])
        self.assertEqual(rows[0]['output_text'], '')
        self.assertIsNone(rows[1]['output_text'])
        self.assertEqual(len([c for c in self.calls if c['sentence_id'] == sid]), 1)

    def test_diagnostic_failure_does_not_terminate_trajectory(self):
        sid = self.corpus[0]['sentence_id']
        def translate(items, on_batch=None, should_stop=None):
            bad = {sid} if items[0]['target_language'] == 'eng_Latn' else set()
            return mock_translate(items, on_batch, should_stop, self.calls, bad)
        self.run_experiment(translate=translate)
        self.assertTrue(all(r['status'] == 'ok' for r in self.records()))
        self.assertTrue(any(r['status'] == 'failed' for r in self.records('observations')))

    def test_generic_translation_exception_preserves_state(self):
        self.run_experiment(stop_after_round=1)
        def broken(*args, **kwargs):
            raise RuntimeError('device error')
        with self.assertRaisesRegex(RuntimeError, 'device error'):
            self.run_experiment(translate=broken)
        evidence = read_jsonl(self.run_dir / 'rounds/pending_000002/failure_attempts.jsonl')
        self.assertTrue(evidence)
        self.assertEqual(evidence[0]['status'], 'exception')
        self.assertIn('device error', evidence[0]['error'])
        self.assertEqual(self.run_experiment()['status'], 'complete')

    def test_deadline_does_not_advance_state(self):
        status = self.run_experiment(should_stop=lambda: True)
        self.assertEqual(status['last_completed_round'], -1)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.run_experiment()['status'], 'complete')

    def test_archive_and_manifest(self):
        self.run_experiment()
        atomic_json(self.root / 'workflow_status.json', {'status': 'complete', 'requested_mode': 'pilot'})
        archive = package_data(self.root, self.root.parent / (self.root.name + '_archives'))
        self.addCleanup(lambda: __import__('shutil').rmtree(archive.parent))
        manifest = verify_archive(archive)
        self.assertEqual(manifest['status'], 'complete')
        self.assertIn('runs/pilot_001/schedules.jsonl', manifest['files'])
        self.assertTrue(archive.with_suffix('.zip.sha256').exists())
        self.assertTrue(self.run_dir.exists())
        with zipfile.ZipFile(archive, 'a') as stream:
            stream.writestr('unexpected.txt', b'changed')
        with self.assertRaises(ValueError):
            verify_archive(archive)

    def test_incomplete_archive_contains_partial_evidence(self):
        self.run_experiment(translate=functools.partial(mock_translate, calls=self.calls, interrupt_at=2))
        archive = package_data(self.root, self.root.parent / (self.root.name + '_archives'))
        self.addCleanup(lambda: __import__('shutil').rmtree(archive.parent))
        manifest = verify_archive(archive)
        self.assertEqual(manifest['status'], 'incomplete')
        self.assertTrue(any('pending_' in name for name in manifest['files']))

    def test_archive_derives_incomplete_status_from_rounds(self):
        self.run_experiment(stop_after_round=1)
        atomic_json(self.root / 'workflow_status.json', {'status': 'complete'})
        archive = package_data(self.root, self.root.parent / (self.root.name + '_archives'))
        self.addCleanup(lambda: __import__('shutil').rmtree(archive.parent))
        self.assertEqual(verify_archive(archive)['status'], 'incomplete')

    def test_archive_refuses_corrupt_completed_round(self):
        self.run_experiment()
        path = self.run_dir / 'rounds/round_000001/checkpoint.json'
        path.write_text('{}\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            package_data(self.root, self.root.parent / 'unused_archives')

    def test_journal_malformed_complete_line_is_not_discarded(self):
        path = self.root / 'journal.jsonl'
        path.write_bytes(b'not-json\n')
        with self.assertRaises(json.JSONDecodeError):
            recover_journal(path)
        self.assertEqual(path.read_bytes(), b'not-json\n')

    def test_duplicate_journal_record_rejected(self):
        self.run_experiment(translate=functools.partial(mock_translate, calls=self.calls, interrupt_at=2))
        path = self.run_dir / 'rounds/pending_000001/journal.jsonl'
        append_rows(path, read_jsonl(path)[:1])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.run_experiment()

    def test_pending_wrong_source_label_rejected(self):
        self.run_experiment(translate=functools.partial(mock_translate, calls=self.calls, interrupt_at=2))
        path = self.run_dir / 'rounds/pending_000001/journal.jsonl'
        rows = read_jsonl(path)
        rows[0]['source_language'] = 'arb_Arab'
        write_jsonl(path, rows)
        with self.assertRaises(ValueError):
            self.run_experiment()

    def test_pending_output_corruption_rejected(self):
        self.run_experiment(translate=functools.partial(mock_translate, calls=self.calls, interrupt_at=2))
        path = self.run_dir / 'rounds/pending_000001/journal.jsonl'
        rows = read_jsonl(path)
        rows[0]['output_text'] = 'corrupted but valid JSON'
        write_jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.run_experiment()

    def test_writer_lock_excludes_second_process(self):
        import subprocess
        import sys
        with exclusive_lock(self.root):
            result = subprocess.run([sys.executable, '-c',
                'from pathlib import Path; from src.storage import exclusive_lock; '
                'import sys; ctx=exclusive_lock(Path(sys.argv[1])); ctx.__enter__()', str(self.root)],
                cwd=project_root, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Another process', result.stderr)

    def test_suspicious_outputs_remain_exact(self):
        self.assertIn('empty_output', suspicious_output('  '))
        self.assertIn('repetitive_output', suspicious_output('repeat ' * 20))
        self.assertIn('unicode_replacement_character', suspicious_output('abc\ufffd'))

    def test_estimate_includes_diagnostics_and_repetitions(self):
        self.config['main']['repetitions'] = 2
        result = estimate_main(self.config, [{'collection_seconds': 60, 'model_translations': 30}])
        self.assertEqual(result['main_translation_upper_count'], 60)
        self.assertAlmostEqual(result['main_hours_estimate'], 120 / 3600)

    def test_source_snapshot_and_runtime_only_changes(self):
        initialize(self.root, self.config, project_root / 'config/experiment.toml')
        self.assertTrue((self.root / 'reproduction/src/translation.py').exists())
        changed = copy.deepcopy(self.config)
        changed['runtime']['max_hours'] = 1
        initialize(self.root, changed, project_root / 'config/experiment.toml')
        changed['main']['rounds'] += 1
        with self.assertRaises(ValueError):
            initialize(self.root, changed, project_root / 'config/experiment.toml')

    def test_oom_adaptive_reduction(self):
        class fake_oom(Exception):
            pass
        torch = mock.Mock()
        torch.cuda.OutOfMemoryError = fake_oom
        engine = {'batch_size': 4, 'torch': torch}
        def generate(engine, items, target):
            if len(items) > 1:
                raise fake_oom('OOM')
            return [{'sentence_id': items[0]['sentence_id'], 'warnings': []}]
        items = [{'sentence_id': str(i)} for i in range(4)]
        with mock.patch('src.translation.generate_batch', side_effect=generate):
            results = resilient_batch(engine, items, 'fra_Latn')
        self.assertEqual(engine['batch_size'], 1)
        self.assertEqual([r['sentence_id'] for r in results], ['0', '1', '2', '3'])
        self.assertEqual(torch.cuda.empty_cache.call_count, 2)

    def test_single_item_oom_propagates(self):
        class fake_oom(Exception):
            pass
        torch = mock.Mock()
        torch.cuda.OutOfMemoryError = fake_oom
        with mock.patch('src.translation.generate_batch', side_effect=fake_oom):
            with self.assertRaises(fake_oom):
                resilient_batch({'batch_size': 1, 'torch': torch}, [{}], 'fra_Latn')
        torch.cuda.empty_cache.assert_not_called()

    def test_user_defined_identifiers_snake_case(self):
        for path in [*project_root.glob('src/*.py'), *project_root.glob('tests/*.py')]:
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    self.assertEqual(node.name, node.name.lower(), f'{path}: {node.name}')


if __name__ == '__main__':
    unittest.main()
