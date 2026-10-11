import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


def utc_now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def digest(value: object) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    sync_directory(path.parent)


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(path, json_bytes(value))


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    atomic_bytes(path, b''.join(json_bytes(row) for row in rows))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def append_rows(path: Path, rows: list[dict]) -> None:
    with path.open('ab') as stream:
        for row in rows:
            stream.write(json_bytes(row))
        stream.flush()
        os.fsync(stream.fileno())


def recover_journal(path: Path) -> list[dict]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    boundary = raw.rfind(b'\n') + 1
    if boundary != len(raw):
        # Keep torn bytes as evidence, but never treat them as a finished translation.
        atomic_bytes(path.with_name(path.name + '.torn_' + uuid.uuid4().hex), raw[boundary:])
        atomic_bytes(path, raw[:boundary])
    return read_jsonl(path)


@contextlib.contextmanager
def exclusive_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another process holds {root}/.lock') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def seal_round(pending: Path, destination: Path, checkpoint: dict) -> None:
    atomic_json(pending / 'checkpoint.json', checkpoint)
    checksums = {p.name: file_hash(p) for p in sorted(pending.iterdir()) if p.is_file() and p.name != 'manifest.json'}
    atomic_json(pending / 'manifest.json', checksums)
    sync_directory(pending)
    os.rename(pending, destination)
    sync_directory(destination.parent)


def verify_round(path: Path) -> dict:
    checksums = read_json(path / 'manifest.json')
    actual = {p.name for p in path.iterdir() if p.is_file() and p.name != 'manifest.json'}
    if actual != set(checksums):
        raise ValueError(f'Round file list changed: {path}')
    for name, expected in checksums.items():
        if Path(name).name != name or file_hash(path / name) != expected:
            raise ValueError(f'Round checksum mismatch: {path}/{name}')
    return read_json(path / 'checkpoint.json')
