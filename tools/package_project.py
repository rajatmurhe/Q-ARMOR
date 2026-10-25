"""Create and verify a portable project ZIP, including untracked data/results."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {'.venv', '.venv-recovery-bootstrap', 'venv', '__pycache__', '.pytest_cache', '.local-identities', 'node_modules'}


def fingerprint(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def excluded(path):
    name = path.name
    return (name == '.DS_Store' or name == '.env' or name.startswith('.env.') and name != '.env.example'
            or name.endswith(('.private.pem', '.key', '.sqlite', '.sqlite-wal', '.sqlite-shm', '.db', '.pyc', '.pyo')))


def package(root, output):
    root, output = Path(root).resolve(), Path(output).expanduser().resolve()
    if output == root or root in output.parents:
        raise ValueError('Choose a ZIP destination outside the project folder')
    if output.exists(): raise FileExistsError(output)
    files, exclusions = [], []
    for folder, dirs, names in os.walk(root, followlinks=False):
        folder = Path(folder)
        for directory in list(dirs):
            path = folder/directory
            if directory in EXCLUDE_DIRS:
                dirs.remove(directory); exclusions.append(dict(path=path.relative_to(root).as_posix()+'/', reason='Local environment/cache/private identities'))
            elif path.is_symlink():
                raise ValueError(f'Unresolved linked directory: {path}; materialize its required content before packaging')
        dirs.sort()
        for name in sorted(names):
            path = folder/name
            if path == root/'DELIVERY_MANIFEST.json':
                exclusions.append(dict(path='DELIVERY_MANIFEST.json', reason='Replaced by newly generated delivery manifest'))
                continue
            if excluded(path):
                exclusions.append(dict(path=path.relative_to(root).as_posix(), reason='Machine metadata, cache or private credentials/state'))
                continue
            if path.is_symlink(): raise ValueError(f'Unresolved linked file: {path}')
            files.append(dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size, sha256=fingerprint(path)))
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_folder=root.name,
                    file_count=len(files), total_uncompressed_bytes=sum(f['bytes'] for f in files),
                    files=files, exclusions=exclusions,
                    interpretation='All available project files, including untracked datasets/results. Missing deleted files cannot be recovered by packaging. See DELIVERY_README.md.')
    partial = output.with_suffix(output.suffix+'.partial')
    if partial.exists(): raise FileExistsError(partial)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
            for item in files: archive.write(root/item['path'], root.name+'/'+item['path'])
            archive.writestr(root.name+'/DELIVERY_MANIFEST.json', json.dumps(manifest, indent=2))
        with zipfile.ZipFile(partial) as archive:
            if len(archive.infolist()) != len(files)+1: raise RuntimeError('ZIP entry count mismatch')
            for item in files:
                h = hashlib.sha256()
                with archive.open(root.name+'/'+item['path']) as stream:
                    for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
                if h.hexdigest() != item['sha256']: raise RuntimeError('File changed during packaging or ZIP verification failed: '+item['path'])
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    result = dict(zip_path=str(output), bytes=output.stat().st_size, sha256=fingerprint(output),
                  verified_entries=len(files)+1, excluded_entries=len(exclusions))
    output.with_suffix(output.suffix+'.sha256').write_text(result['sha256']+'  '+output.name+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(package(ROOT, args.output), indent=2))

if __name__ == '__main__': main()
