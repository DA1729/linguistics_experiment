import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def text_id(text):
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def save_table(output, name, rows, columns=None):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    path = Path(output) / 'tables' / (name + '.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format='%.10g')
    return frame


def valid_rows(frame):
    return frame.loc[(frame.status == 'ok') & frame.output_text.map(lambda x: isinstance(x, str) and bool(x.strip()))].copy()


def pair_count(counts):
    a = np.asarray(list(counts), dtype=np.int64)
    return int(np.sum(a * (a - 1)) // 2)
