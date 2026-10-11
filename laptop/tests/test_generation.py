import contextlib
import unittest
from unittest import mock

from src.translation import generate_batch, resilient_batch
from test_project import fake_tokenizer


class padded_inputs(dict):
    def to(self, device):
        return self


class generation_tests(unittest.TestCase):
    def engine(self, sequences, decoded_text='  Exact, Élan!  '):
        tokenizer = fake_tokenizer()
        tokenizer.pad = lambda features, **kwargs: padded_inputs(input_ids=[r['input_ids'] for r in features],
                                                                attention_mask=[r['attention_mask'] for r in features])
        tokenizer.decode = mock.Mock(return_value=decoded_text)
        torch = mock.Mock()
        torch.inference_mode = contextlib.nullcontext
        torch.cuda.memory_allocated.return_value = 0
        model = mock.Mock()
        model.generate.return_value.cpu.return_value.tolist.return_value = sequences
        return {'torch': torch, 'tokenizer': tokenizer, 'model': model,
                'config': {'max_input_tokens': 8, 'max_new_tokens': 4, 'max_batch_seconds': 90},
                'parameters': {'max_new_tokens': 4, 'do_sample': False, 'num_beams': 1}}

    def items(self):
        return [{'sentence_id': 'first', 'text': 'a long input', 'source_language': 'fra_Latn', 'target_language': 'deu_Latn'},
                {'sentence_id': 'second', 'text': 'short', 'source_language': 'arb_Arab', 'target_language': 'deu_Latn'}]

    def test_exact_outputs_tokens_and_target_forcing(self):
        engine = self.engine([[2, 12, 99, 2, 1], [2, 12, 87, 2, 1]])
        result = generate_batch(engine, self.items(), 'deu_Latn')
        self.assertEqual(result[0]['output_text'], '  Exact, Élan!  ')
        self.assertEqual(result[0]['generated_token_ids'], [2, 12, 99, 2])
        self.assertEqual(result[0]['encoded_input_ids'][0], 11)
        self.assertEqual(result[1]['encoded_input_ids'][0], 13)
        self.assertIn('input_truncated', result[0]['warnings'])
        kwargs = engine['model'].generate.call_args.kwargs
        self.assertEqual(kwargs['forced_bos_token_id'], 12)
        self.assertFalse(kwargs['do_sample'])
        self.assertEqual(engine['tokenizer'].decode.call_args.kwargs['clean_up_tokenization_spaces'], False)

    def test_generation_limit_is_flagged_even_with_forced_eos(self):
        engine = self.engine([[2, 12, 99, 99, 2]])
        result = generate_batch(engine, self.items()[:1], 'deu_Latn')[0]
        self.assertEqual(result['status'], 'ok')
        self.assertIn('generation_token_limit', result['warnings'])

    def test_time_limit_preserves_raw_output_but_marks_failure(self):
        engine = self.engine([[2, 12, 99]])
        result = generate_batch(engine, self.items()[:1], 'deu_Latn')[0]
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['output_text'], '  Exact, Élan!  ')
        self.assertIn('generation_time_limit', result['warnings'])

    def test_empty_output_preserved(self):
        engine = self.engine([[2, 12, 2]], decoded_text='')
        result = generate_batch(engine, self.items()[:1], 'deu_Latn')[0]
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['output_text'], '')

    def test_wrong_target_token_rejected(self):
        engine = self.engine([[2, 11, 2]])
        with self.assertRaisesRegex(RuntimeError, 'target-language'):
            generate_batch(engine, self.items()[:1], 'deu_Latn')

    def test_reduced_batches_durable_before_later_failure(self):
        class fake_oom(Exception):
            pass
        torch = mock.Mock()
        torch.cuda.OutOfMemoryError = fake_oom
        engine = {'batch_size': 2, 'torch': torch}
        persisted = []
        def generate(engine, items, target):
            if len(items) == 2:
                raise fake_oom()
            if items[0]['sentence_id'] == 'second':
                raise RuntimeError('later failure')
            return [{'sentence_id': 'first', 'warnings': []}]
        with mock.patch('src.translation.generate_batch', side_effect=generate):
            with self.assertRaisesRegex(RuntimeError, 'later failure'):
                resilient_batch(engine, self.items(), 'deu_Latn', on_batch=lambda items, rows: persisted.extend(rows))
        self.assertEqual(persisted, [{'sentence_id': 'first', 'warnings': ['oom_batch_reduced']}])
