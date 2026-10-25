"""Restore only checksum-identical CICIoT2023 archive/sample files; never substitute data."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT/'results/remaining_study_v2/ablation/data_manifest.json'
SPLITS = {'train': 'train/train.csv', 'val': 'validation/validation.csv', 'test': 'test/test.csv'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require_match(path, expected):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Required original file is missing: {path}')
    if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
        raise ValueError(f'File differs from the frozen study fingerprint: {path}. No substitution was made.')


def dataset_status(directory, manifest):
    directory = Path(directory)
    rows = {}
    for key, relative in SPLITS.items():
        path = directory/relative
        if not path.is_file():
            rows[key] = {'status': 'missing', 'path': str(path)}
        else:
            try:
                require_match(path, manifest[key])
                rows[key] = {'status': 'verified', 'path': str(path), 'sha256': manifest[key]['sha256']}
            except ValueError as exc:
                rows[key] = {'status': 'mismatch', 'path': str(path), 'error': str(exc)}
    return {'status': 'verified' if all(r['status'] == 'verified' for r in rows.values()) else 'blocked',
            'scope': 'Frozen bounded archive sample; not the original Kaggle predefined/full split', 'splits': rows}


def install_splits(source, output, manifest):
    """Validate every input before copying; never replace existing mismatched files."""
    source, output = Path(source), Path(output)
    for key, relative in SPLITS.items():
        require_match(source/relative, manifest[key])
        if (output/relative).exists():
            require_match(output/relative, manifest[key])
    for key, relative in SPLITS.items():
        target = output/relative
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.restore-', delete=False) as stream:
            temporary = Path(stream.name)
        try:
            shutil.copyfile(source/relative, temporary)
            require_match(temporary, manifest[key])
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return dataset_status(output, manifest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group()
    source.add_argument('--archive', type=Path, help='Original CSV.zip; SHA-256 must match the frozen manifest')
    source.add_argument('--mirror-archive', type=Path, help='Explicit Kaggle data/<class> repack; accepted only if all three regenerated sample CSV hashes match')
    source.add_argument('--splits', type=Path, help='Previously saved train/validation/test directory with exact hashes')
    p.add_argument('--output', type=Path, default=ROOT/'data/CICIoT23_archive_sample_v1')
    p.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    if a.splits:
        result = install_splits(a.splits.expanduser().resolve(), a.output, manifest)
    elif a.archive or a.mirror_archive:
        archive = (a.archive or a.mirror_archive).expanduser().resolve()
        description = manifest['dataset_description']
        if a.mirror_archive:
            if not archive.is_file(): raise FileNotFoundError(archive)
            print('Explicit mirror recovery: container hash may differ; ALL generated split hashes must match before installation.', file=sys.stderr)
        else:
            require_match(archive, {'bytes': description['archive_bytes'], 'sha256': description['archive_sha256']})
        a.output.parent.mkdir(parents=True, exist_ok=True)
        # Sampling streams directly from the archive; no multi-GB full extraction.
        with tempfile.TemporaryDirectory(dir=a.output.parent, prefix='.qarmor-restore-') as temp:
            stage = Path(temp)/'sample'
            caps = description['caps_per_class']
            command = [sys.executable, '-m', 'tools.prepare_archive_splits', '--archive', str(archive),
                       '--output', str(stage), '--seed', str(description['seed']),
                       '--train-per-class', str(caps['train']), '--validation-per-class', str(caps['validation']),
                       '--test-per-class', str(caps['test'])]
            if a.mirror_archive: command += ['--source-prefix', 'data']
            subprocess.run(command, cwd=ROOT, check=True)
            result = install_splits(stage, a.output, manifest)
            shutil.copy2(stage/'dataset_manifest.json', a.output/'restoration_manifest.json')
    else:
        result = dataset_status(a.output, manifest)
    print(json.dumps(result, indent=2))
    if result['status'] != 'verified':
        p.exit(2, 'Matching source data is unavailable. Restore CSV.zip or exact split CSVs; no replacement was generated.\n')

if __name__ == '__main__':
    main()
