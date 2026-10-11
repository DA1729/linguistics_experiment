"""Optional CPU integration check using already downloaded tokenizer files, never weights."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.languages import language_names
from src.translation import encode_inputs, validate_languages


def check_tokenizer(directory: Path) -> None:
    from transformers import AutoTokenizer as auto_tokenizer
    tokenizer = auto_tokenizer.from_pretrained(directory, local_files_only=True,
                                              src_lang='eng_Latn', use_fast=True, legacy_behaviour=False)
    validate_languages(tokenizer, list(language_names) + ['eng_Latn'])
    items = [
        {'text': 'Le chat dort.', 'source_language': 'fra_Latn'},
        {'text': 'القط ينام.', 'source_language': 'arb_Arab'},
        {'text': '猫が寝ています。', 'source_language': 'jpn_Jpan'},
    ]
    rows = encode_inputs(tokenizer, items, 512)
    padded = tokenizer.pad([{k: row[k] for k in ('input_ids', 'attention_mask')} for row in rows], return_tensors='pt')
    expected = [tokenizer.convert_tokens_to_ids(item['source_language']) for item in items]
    assert padded['input_ids'][:, 0].tolist() == expected
    assert all(row['input_ids'][-1] == tokenizer.eos_token_id for row in rows)
    separate = [encode_inputs(tokenizer, [item], 512)[0] for item in items]
    assert rows == separate
    long = encode_inputs(tokenizer, [{'text': 'A long sentence. ' * 200, 'source_language': 'eng_Latn'}], 16)[0]
    assert len(long['input_ids']) == 16 and 'input_truncated' in long['warnings']
    print('PASS: 51 language codes; mixed-source tokens, EOS, padding, individual equivalence and truncation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tokenizer_directory', type=Path)
    check_tokenizer(parser.parse_args().tokenizer_directory)
