import argparse
import contextlib
import functools
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from src.hardware import stop_reason
from src.package_data import verify_archive
from src.runner import execute, estimate_local, read_config
from src.storage import atomic_json, read_json
from test_project import mock_translate, project_root


class workflow_tests(unittest.TestCase):
    def run(self, result=None):
        with tempfile.TemporaryDirectory() as temp:
            self.root = Path(temp)
            self.config = read_config(project_root / 'config/experiment.toml')
            for phase in ('smoke', 'pilot', 'main'):
                self.config[phase].update(sentences=8, rounds=2, repetitions=1,
                                          observe_every=1, observation_rounds=[])
            self.config['experiment']['languages'] = ['fra_Latn', 'deu_Latn', 'arb_Arab']
            corpus = self.root / 'corpus.txt'
            corpus.write_text('\n'.join(f'sentence {i}' for i in range(10)), encoding='utf-8')
            self.config['dataset']['local_file'] = str(corpus)
            self.config_path = self.root / 'experiment.toml'
            self.write_config()
            self.args = argparse.Namespace(mode='benchmark', device='cpu', root=self.root / 'runtime', config=self.config_path)
            self.calls = []
            self.translate = functools.partial(mock_translate, calls=self.calls)
            self.environment = {'backend': 'cpu', 'gpu': None, 'precision': 'float32', 'cpu_threads': 4}
            self.health = {'available_memory_gib': 8, 'total_memory_gib': 15, 'on_ac_power': True,
                           'temperature_c': 45, 'kernel': 'test_only'}
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
        with mock.patch('src.translation.load_translator', return_value=(self.translate, self.environment, {})), \
                mock.patch('src.hardware.system_status', return_value=self.health), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return execute(self.args)

    def data_root(self):
        return self.args.root / 'data' / f"{self.config['experiment']['name']}_cpu"

    def test_default_only_benchmarks_and_then_explicit_run(self):
        self.assertEqual(self.execute(), 0)
        root = self.data_root()
        self.assertFalse((root / 'runs/main_001').exists())
        report = read_json(root / 'runtime_estimate.json')
        self.assertTrue(report['allow_local_run'])
        self.assertEqual(report['scenarios']['original_pod_main']['model_translations'], 123000)
        count = len(self.calls)
        self.assertEqual(self.execute(), 0)
        self.assertEqual(len(self.calls), count)
        self.args.mode = 'run'
        self.assertEqual(self.execute(), 0)
        self.assertEqual(read_json(root / 'runs/main_001/status.json')['status'], 'complete')
        count = len(self.calls)
        self.assertEqual(self.execute(), 0)
        self.assertEqual(len(self.calls), count)

    def test_package_without_inference_or_dependencies(self):
        self.assertEqual(self.execute(), 0)
        result = subprocess.run(['bash', 'run_local.sh', 'package', 'cpu', str(self.config_path)],
                                cwd=project_root, env=dict(os.environ, LAPTOP_ROOT=str(self.args.root)),
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        archive = self.args.root / 'archives' / f"{self.config['experiment']['name']}_cpu_complete.zip"
        self.assertEqual(verify_archive(archive)['status'], 'complete')

    def test_run_without_benchmark_never_loads_model(self):
        self.args.mode = 'run'
        self.assertEqual(self.execute(), 1)
        self.assertEqual(self.calls, [])
        self.assertFalse((self.data_root() / 'runs/main_001').exists())

    def test_slow_or_failed_benchmark_blocks_long_run(self):
        statuses = [{'model_translations': 32, 'collection_seconds': 20000}]
        report = estimate_local(self.config, statuses, 32, 0, 10, self.environment)
        self.assertFalse(report['allow_local_run'])
        self.assertFalse(estimate_local(self.config, statuses, 31, 0, 10, self.environment)['adequate_benchmark'])
        self.assertFalse(estimate_local(self.config, statuses, 32, 1, 10, self.environment)['adequate_benchmark'])

    def test_gate_rejects_changed_configuration(self):
        self.assertEqual(self.execute(), 0)
        path = self.data_root() / 'runtime_estimate.json'
        report = read_json(path)
        report['config_digest'] = 'wrong'
        atomic_json(path, report)
        count = len(self.calls)
        self.args.mode = 'run'
        self.assertEqual(self.execute(), 1)
        self.assertEqual(len(self.calls), count)

    def test_ac_memory_and_thermal_stops(self):
        settings = self.config['laptop']
        self.assertIsNone(stop_reason(settings, self.health))
        for field, value in [('on_ac_power', False), ('available_memory_gib', 0.4), ('temperature_c', 94)]:
            self.assertIsNotNone(stop_reason(settings, {**self.health, field: value}))
        self.health['on_ac_power'] = False
        self.assertEqual(self.execute(), 2)
        self.assertEqual(self.calls, [])

    def test_four_hour_cap_cannot_be_exceeded(self):
        self.config['runtime']['max_hours'] = 4.01
        self.write_config()
        with self.assertRaisesRegex(ValueError, 'four hours'):
            read_config(self.config_path)

    def test_failed_backend_preserves_source_and_logs(self):
        def fail(*args, **kwargs):
            raise RuntimeError('simulated driver failure')
        self.translate = fail
        self.assertEqual(self.execute(), 1)
        archive = self.args.root / 'archives' / f"{self.config['experiment']['name']}_cpu_incomplete.zip"
        manifest = verify_archive(archive)
        self.assertIn('corpus/corpus.jsonl', manifest['files'])
        self.assertIn('logs/execution.log', manifest['files'])
