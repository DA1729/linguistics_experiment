from pathlib import Path
import random
import urllib.request

from .storage import atomic_bytes, atomic_json, digest, file_hash, read_json, read_jsonl, write_jsonl

corpus_revision = '8a8758a392e429432f6e95fcac1445550a4d3ccd'
corpus_hash = '612e9fbe87997617c0fa8fa8929654a4f49b728d96738112c2b86ef6a1d78d88'
corpus_url = f'https://huggingface.co/datasets/DGME/FLORES-200/resolve/{corpus_revision}/devtest/eng_Latn.devtest'


def select_sentences(raw: str, count: int, seed: int) -> list[dict]:
    seen, candidates = set(), []
    # Remove line delimiters only. Spaces, accents and punctuation remain untouched.
    for line_number, text in enumerate(raw.split('\n'), 1):
        text = text.removesuffix('\r')
        if text.strip() and text not in seen:
            seen.add(text)
            candidates.append({'sentence_id': f's_{digest(text)[:24]}', 'source_line': line_number, 'text': text})
    if len(candidates) < count:
        raise ValueError(f'Corpus has {len(candidates)} unique nonempty sentences; requested {count}')
    random.Random(seed).shuffle(candidates)
    return candidates[:count]


def prepare_corpus(root: Path, config: dict, count: int, seed: int) -> tuple[list[dict], dict]:
    root.mkdir(parents=True, exist_ok=True)
    local = config['local_file']
    source = Path(local).expanduser().resolve() if local else root / 'source_english.devtest'
    if not local and not source.exists():
        try:
            with urllib.request.urlopen(corpus_url, timeout=60) as response:
                atomic_bytes(source, response.read())
        except Exception as exc:
            raise RuntimeError('FLORES download failed. Retry or set dataset.local_file to a UTF-8 corpus; no substitute was used.') from exc
    source_sha = file_hash(source)
    if not local and source_sha != corpus_hash:
        raise ValueError('Downloaded FLORES corpus checksum mismatch')
    provenance = {
        'source': str(source) if local else 'DGME/FLORES-200',
        'url': None if local else corpus_url, 'split': 'custom' if local else 'devtest',
        'revision': None if local else corpus_revision, 'source_sha256': source_sha,
        'license': config['local_license'] if local else 'CC-BY-SA-4.0',
        'license_note': 'Upstream Meta FLORES license controls; the mirror card says CC-BY-4.0.' if not local else '',
        'attribution': 'NLLB Team et al. (2022), No Language Left Behind: Scaling Human-Centered Machine Translation' if not local else 'User-supplied corpus',
        'selection': 'Deduplicate exact strings in source order; Python Random(seed).shuffle; first N; retain original line text',
        'selection_seed': seed, 'count': count,
    }
    rows = select_sentences(source.read_bytes().decode('utf-8'), count, seed)
    destination = root / 'corpus.jsonl'
    if destination.exists():
        if read_jsonl(destination) != rows or read_json(root / 'provenance.json') != provenance:
            raise ValueError('Corpus or selection changed. Use a new experiment name.')
    else:
        atomic_json(root / 'provenance.json', provenance)
        write_jsonl(destination, rows)
        if local:
            atomic_bytes(root / 'source_custom.txt', source.read_bytes())
    return rows, provenance
