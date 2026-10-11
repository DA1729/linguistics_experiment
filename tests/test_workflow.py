import argparse
import contextlib
import functools
import io
import json
import logging
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from src.package_data import verify_archive
from src.runner import execute, read_config
from src.storage import read_json
from test_project import mock_translate, project_root


class workflow_tests(unittest.TestCase):
    def run(self, result=None):
        with tempfile.TemporaryDirectory() as temp:
            self.root = Path(temp)
            self.config = read_config(project_root / 'config/experiment.toml')
            for phase in ('smoke', 'pilot', 'main'):
                self.config[phase].update(sentences=3, rounds=2, repetitions=1,
                                          observe_every=1, observation_rounds=[])
            self.config['main']['repetitions'] = 2
            self.config['experiment']['languages'] = ['fra_Latn', 'deu_Latn', 'arb_Arab']
            corpus = self.root / 'corpus.txt'
            corpus.write_text('one\ntwo\nthree\nfour\n', encoding='utf-8')
            self.config['dataset']['local_file'] = str(corpus)
            self.config_path = self.root / 'experiment.toml'
            self.write_config()
            self.args = argparse.Namespace(mode='all', root=self.root / 'runtime', config=self.config_path, started_at=None)
            self.calls = []
            self.translate = functools.partial(mock_translate, calls=self.calls)
            self.environment = {'gpu': 'mock_gpu_not_real', 'gpu_memory_bytes': 1000}
            self.engine = {'torch': mock.Mock()}
            self.engine['torch'].cuda.memory_allocated.return_value = 0
            try:
                return super().run(result)
            finally:
                for handler in logging.getLogger().handlers[:]:
                    handler.close()
                    logging.getLogger().removeHandler(handler)

    def write_config(self):
        self.config_path.write_text('\n'.join(f'[{section}]\n' + '\n'.join(f'{k} = {json.dumps(v)}' for k, v in fields.items())
                                               for section, fields in self.config.items()), encoding='utf-8')

    def execute(self):
        with mock.patch('src.translation.load_translator', return_value=(self.translate, self.environment, self.engine)), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return execute(self.args)

    def test_full_workflow_archive_resume_and_cpu_package_command(self):
        self.assertEqual(self.execute(), 0)
        root = self.args.root / 'data/multilingual_translation_v1'
        self.assertEqual(read_json(root / 'workflow_status.json')['status'], 'complete')
        self.assertTrue(read_json(root / 'smoke_report.json')['checkpoint_recovery'] == 'passed')
        self.assertEqual(read_json(root / 'runs/main_002/status.json')['status'], 'complete')
        self.assertTrue((root / 'runtime_estimate.json').exists())
        count = len(self.calls)
        self.args.mode = 'resume'
        self.assertEqual(self.execute(), 0)
        self.assertEqual(count, len(self.calls))
        archive = self.args.root / 'archives/multilingual_translation_v1_complete.zip'
        self.assertEqual(verify_archive(archive)['status'], 'complete')
        import os
        env = dict(os.environ, LINGUISTIC_ROOT=str(self.args.root))
        result = subprocess.run(['bash', 'run_pod.sh', 'package', str(self.config_path)], cwd=project_root,
                                env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(archive), result.stdout)

    def test_smoke_only_then_upgrade_to_full_workflow(self):
        self.args.mode = 'smoke'
        self.assertEqual(self.execute(), 0)
        root = self.args.root / 'data/multilingual_translation_v1'
        self.assertFalse((root / 'runs/pilot_001').exists())
        self.args.mode = 'all'
        self.assertEqual(self.execute(), 0)
        self.assertTrue((root / 'runs/main_001').exists())

    def test_stopped_workflow_archive_is_incomplete(self):
        self.args.started_at = 1.0
        self.assertEqual(self.execute(), 2)
        archive = self.args.root / 'archives/multilingual_translation_v1_incomplete.zip'
        self.assertEqual(verify_archive(archive)['status'], 'incomplete')
        self.assertEqual(self.calls, [])

    def test_runtime_failure_archives_saved_data(self):
        def fail(*args, **kwargs):
            raise RuntimeError('simulated GPU driver failure')
        self.translate = fail
        self.assertEqual(self.execute(), 1)
        archive = self.args.root / 'archives/multilingual_translation_v1_incomplete.zip'
        manifest = verify_archive(archive)
        self.assertIn('corpus/corpus.jsonl', manifest['files'])
        self.assertIn('logs/execution.log', manifest['files'])

    def test_config_validation_rejects_typo_and_bad_limits(self):
        self.config['model']['batch_size'] = 0
        self.write_config()
        with self.assertRaises(ValueError):
            read_config(self.config_path)
        self.config['model']['batch_size'] = 1
        self.config['runtime']['typo'] = True
        self.write_config()
        with self.assertRaises(ValueError):
            read_config(self.config_path)
