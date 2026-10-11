import argparse
import hashlib
from pathlib import Path
import zipfile

from .storage import atomic_bytes, file_hash, json_bytes, read_json, utc_now, verify_round


def verify_archive(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP CRC validation failed')
        import json
        manifest = json.loads(archive.read('integrity_manifest.json'))
        if set(archive.namelist()) != set(manifest['files']) | {'integrity_manifest.json'}:
            raise ValueError('Archive manifest file list mismatch')
        for name, expected in manifest['files'].items():
            h = hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
            if h.hexdigest() != expected:
                raise ValueError(f'Archive SHA-256 mismatch: {name}')
    return manifest


def package_data(root: Path, archive_directory: Path) -> Path:
    if not root.is_dir():
        raise ValueError(f'No experiment data at {root}')
    for path in root.glob('runs/*/rounds/round_*'):
        verify_round(path)
    archive_directory.mkdir(parents=True, exist_ok=True)
    if archive_directory.resolve().is_relative_to(root.resolve()):
        raise ValueError('Archives must be outside experiment data')
    workflow = read_json(root / 'workflow_status.json') if (root / 'workflow_status.json').exists() else {'status': 'incomplete'}
    status = workflow['status']
    if any(root.glob('runs/*/rounds/pending_*')):
        status = 'incomplete'
    destination = archive_directory / f'{root.name}_{status}.zip'
    temp = destination.with_suffix('.zip.tmp')
    files = [p for p in sorted(root.rglob('*')) if p.is_file() and p.name != '.lock' and not p.name.endswith('.tmp')]
    if any(p.is_symlink() for p in files):
        raise ValueError('Refusing symlinks in experiment data')
    checksums = {str(p.relative_to(root)): file_hash(p) for p in files}
    manifest = {'schema_version': 1, 'created_at': utc_now(), 'status': status,
                'scope': workflow.get('requested_mode'), 'files': checksums}
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in files:
            archive.write(path, str(path.relative_to(root)))
        archive.writestr('integrity_manifest.json', json_bytes(manifest))
    verify_archive(temp)
    temp.replace(destination)
    atomic_bytes(destination.with_suffix('.zip.sha256'), f'{file_hash(destination)}  {destination.name}\n'.encode())
    return destination.resolve()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify an existing portable data ZIP (CPU only)')
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    result = verify_archive(args.archive)
    print(f"Verified {len(result['files'])} files; status={result['status']}")
