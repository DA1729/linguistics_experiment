from collections import defaultdict
import functools
import importlib.metadata
import logging
import os
from pathlib import Path
import platform
import time

from .storage import file_hash


def validate_languages(tokenizer, codes: list[str]) -> None:
    special = set(tokenizer.all_special_ids)
    for code in codes:
        token = tokenizer.convert_tokens_to_ids(code)
        if token == tokenizer.unk_token_id or token not in special or tokenizer.convert_ids_to_tokens(token) != code:
            raise ValueError(f'Unsupported NLLB language token: {code}')


def encode_inputs(tokenizer, items: list[dict], max_tokens: int) -> list[dict]:
    encoded = []
    for item in items:
        tokenizer.src_lang = item['source_language']
        full = tokenizer(item['text'], truncation=False, add_special_tokens=True)
        data = tokenizer(item['text'], truncation=True, max_length=max_tokens, add_special_tokens=True)
        ids = list(data['input_ids'])
        expected = tokenizer.convert_tokens_to_ids(item['source_language'])
        if not ids or ids[0] != expected or ids[-1] != tokenizer.eos_token_id:
            raise RuntimeError('NLLB source token layout mismatch; expected [source, text, eos]')
        encoded.append({
            'input_ids': ids, 'attention_mask': list(data['attention_mask']),
            'full_input_tokens': len(full['input_ids']),
            'warnings': ['input_truncated'] if len(full['input_ids']) > len(ids) else [],
        })
    return encoded


def batch_indices(items: list[dict], batch_size: int):
    groups = defaultdict(list)
    for index, item in enumerate(items):
        groups[item['target_language']].append(index)
    for target in sorted(groups):
        indices = groups[target]
        for offset in range(0, len(indices), batch_size):
            yield target, indices[offset:offset + batch_size]


def translate_batches(items: list[dict], batch_size: int, batch_function, on_batch=None, should_stop=None) -> list[dict]:
    if len({item['sentence_id'] for item in items}) != len(items):
        raise ValueError('Duplicate sentence identifiers in a translation request')
    outputs = [None] * len(items)
    for target, indices in batch_indices(items, batch_size):
        if should_stop and should_stop():
            raise InterruptedError('Execution deadline or stop signal reached')
        batch = [items[i] for i in indices]
        results = batch_function(batch, target)
        if len(results) != len(batch):
            raise RuntimeError('Translator returned the wrong number of outputs')
        for index, result in zip(indices, results):
            if result['sentence_id'] != items[index]['sentence_id']:
                raise RuntimeError('Translator changed sentence ordering or identifiers')
            outputs[index] = result
        if on_batch:
            on_batch(batch, results)
    return outputs


def suspicious_output(text: str) -> list[str]:
    warnings = []
    words = text.split()
    if not text.strip():
        warnings.append('empty_output')
    if '\ufffd' in text:
        warnings.append('unicode_replacement_character')
    if len(words) >= 12 and len(set(words)) <= len(words) / 4:
        warnings.append('repetitive_output')
    return warnings


def load_translator(config: dict, languages: list[str], seed: int):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('USE_TF', '0')
    os.environ.setdefault('USE_FLAX', '0')
    import torch
    import transformers
    from transformers import AutoModelForSeq2SeqLM as auto_model
    from transformers import AutoTokenizer as auto_tokenizer
    from huggingface_hub import hf_hub_download

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for experiment execution; CPU unit tests need no model')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    dtype = getattr(torch, config['precision'])
    if config['precision'] == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not support bfloat16')
    model_directory = None
    model_files = {}
    for filename in ('config.json', 'generation_config.json', 'pytorch_model.bin', 'tokenizer_config.json',
                     'special_tokens_map.json', 'tokenizer.json', 'sentencepiece.bpe.model'):
        path = hf_hub_download(config['name'], filename, revision=config['revision'])
        model_directory = Path(path).parent
        model_files[filename] = file_hash(Path(path))
    revision = model_directory.name
    if revision != config['revision']:
        raise RuntimeError(f'Model cache revision mismatch: {revision!r}')
    tokenizer = auto_tokenizer.from_pretrained(model_directory, local_files_only=True,
                                             src_lang='eng_Latn', use_fast=True, legacy_behaviour=False)
    validate_languages(tokenizer, list(set(languages + ['eng_Latn', 'fra_Latn', 'deu_Latn', 'arb_Arab'])))
    model = auto_model.from_pretrained(model_directory, local_files_only=True,
                                      torch_dtype=dtype, attn_implementation='eager').to('cuda:0').eval()
    parameters = {
        'do_sample': False, 'num_beams': 1, 'num_return_sequences': 1,
        'max_new_tokens': config['max_new_tokens'], 'max_time': config['max_batch_seconds'],
        'use_cache': True, 'repetition_penalty': 1.0, 'no_repeat_ngram_size': 0,
        'forced_eos_token_id': tokenizer.eos_token_id,
    }
    environment = {
        'model_name': config['name'], 'model_revision': revision,
        'model_file_sha256': model_files,
        'model_license': 'CC-BY-NC-4.0', 'python': platform.python_version(),
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'gpu': torch.cuda.get_device_name(0),
        'gpu_capability': list(torch.cuda.get_device_capability(0)),
        'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
        'precision': config['precision'], 'attention': 'eager',
        'deterministic_algorithms': True, 'tf32': False,
        'generation_parameters': parameters,
        'base_generation_config': model.generation_config.to_dict(),
        'tokenizer': {'class': type(tokenizer).__name__, 'revision': revision,
                      'legacy_behaviour': False, 'vocabulary_size': len(tokenizer)},
        'packages': {name: importlib.metadata.version(name) for name in sorted({
            d.metadata['Name'] for d in importlib.metadata.distributions() if d.metadata['Name']})},
    }
    engine = {'torch': torch, 'model': model, 'tokenizer': tokenizer, 'config': config,
              'parameters': parameters, 'batch_size': config['batch_size']}
    translate = functools.partial(translate_items, engine)
    return translate, environment, engine


def generate_batch(engine: dict, items: list[dict], target: str) -> list[dict]:
    torch, tokenizer = engine['torch'], engine['tokenizer']
    config = engine['config']
    encoded = encode_inputs(tokenizer, items, config['max_input_tokens'])
    features = [{key: row[key] for key in ('input_ids', 'attention_mask')} for row in encoded]
    inputs = tokenizer.pad(features, padding=True, return_tensors='pt').to('cuda:0')
    start = time.monotonic()
    with torch.inference_mode():
        tokens = engine['model'].generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(target),
                                         **engine['parameters'])
    sequences = tokens.cpu().tolist()
    seconds = time.monotonic() - start
    results = []
    for item, sequence, source in zip(items, sequences, encoded):
        if len(sequence) < 2 or sequence[1] != tokenizer.convert_tokens_to_ids(target):
            raise RuntimeError('Generated target-language token mismatch')
        ended = tokenizer.eos_token_id in sequence[2:]
        if ended:
            sequence = sequence[:sequence.index(tokenizer.eos_token_id, 2) + 1]
        text = tokenizer.decode(sequence, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        warnings = source['warnings'] + suspicious_output(text)
        reached_limit = len(sequence) - 1 >= config['max_new_tokens']
        if reached_limit:
            warnings.append('generation_token_limit')
        if not ended and not reached_limit:
            warnings.append('generation_time_limit')
        status = 'ok' if text.strip() and (ended or reached_limit) else 'failed'
        results.append({
            'sentence_id': item['sentence_id'], 'output_text': text, 'status': status,
            'warnings': warnings, 'error': None if status == 'ok' else ','.join(warnings),
            'input_tokens': len(source['input_ids']), 'full_input_tokens': source['full_input_tokens'],
            'encoded_input_ids': source['input_ids'], 'generated_token_ids': sequence,
            'output_tokens': len(sequence) - 1, 'batch_size': len(items), 'batch_seconds': seconds,
            'gpu_memory_allocated_bytes': torch.cuda.memory_allocated(),
        })
    return results


def resilient_batch(engine: dict, items: list[dict], target: str, on_batch=None,
                    should_stop=None, reduced=False) -> list[dict]:
    if should_stop and should_stop():
        raise InterruptedError('Execution deadline or stop signal reached')
    if len(items) > engine['batch_size']:
        size = engine['batch_size']
        return [row for start in range(0, len(items), size)
                for row in resilient_batch(engine, items[start:start + size], target,
                                            on_batch, should_stop, reduced)]
    try:
        results = generate_batch(engine, items, target)
    except engine['torch'].cuda.OutOfMemoryError:
        if len(items) == 1:
            raise
        new_size = max(1, len(items) // 2)
    except Exception as exc:
        exc.translation_inputs = items
        raise
    else:
        if reduced:
            for result in results:
                result['warnings'].append('oom_batch_reduced')
        if on_batch:
            on_batch(items, results)
        return results
    # Leave the exception scope before freeing cached memory, releasing traceback tensors.
    engine['batch_size'] = new_size
    engine['torch'].cuda.empty_cache()
    logging.warning('CUDA OOM: reducing batch size to %s', new_size)
    return resilient_batch(engine, items, target, on_batch, should_stop, reduced=True)


def translate_items(engine: dict, items: list[dict], on_batch=None, should_stop=None) -> list[dict]:
    batch_function = functools.partial(resilient_batch, engine, on_batch=on_batch, should_stop=should_stop)
    return translate_batches(items, engine['batch_size'], batch_function, should_stop=should_stop)
