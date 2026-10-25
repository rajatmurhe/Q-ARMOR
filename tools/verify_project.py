"""Fresh, bounded verification of the recovered project; never fabricates missing data."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from tools.restore_data import DEFAULT_MANIFEST, dataset_status

ROOT = Path(__file__).resolve().parents[1]


def save(path, data):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def inspect_sources():
    count = 0
    for folder in ['Agent', 'Server', 'crypto', 'game_theory', 'tools', 'tests']:
        for p in (ROOT/folder).glob('*.py'):
            ast.parse(p.read_text(), filename=str(p)); count += 1
    notebook = json.loads((ROOT/'Training_Report/final-cdl-fl-work (1).ipynb').read_text())
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']), filename=f'notebook cell {i}')
    adapter = (ROOT/'Training_Report/cell27_stackelberg_integrated.py').read_text()
    if ''.join(notebook['cells'][27]['source']) != adapter:
        raise ValueError('Notebook cell27 differs from the exported module adapter')
    return dict(python_modules_parsed=count, notebook_active_cells_parsed=True, cell27_matches_adapter=True)


def inspect_bundle():
    bundle = ROOT/'deployment/archive_sample_seed42'
    manifest = json.loads((bundle/'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        if hashlib.sha256((bundle/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Deployment bundle mismatch: {name}')
    if hashlib.sha256((ROOT/'tools/ids_model.py').read_bytes()).hexdigest() != manifest['architecture_sha256']:
        raise ValueError('Architecture differs from frozen checkpoint source')
    return dict(status='verified', seed=manifest['seed'], method=manifest['method'], files=list(manifest['sha256']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('results/verification_20261021'))
    p.add_argument('--quick', action='store_true', help='Only unit, source, bundle, and saved-study checks')
    a = p.parse_args(); out = a.output.expanduser().resolve()
    if out.exists():
        raise FileExistsError('Use a new verification output directory')
    out.mkdir(parents=True)
    report = dict(status='running', created_utc=datetime.now(timezone.utc).isoformat(),
                  python=sys.version, platform=platform.platform(), machine=platform.machine(), checks=[])
    def step(name, command, timeout):
        start = time.perf_counter()
        log = out/(name+'.log')
        print('Running:', name, flush=True)
        try:
            with log.open('w') as stream:
                process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                         timeout=timeout, env=dict(os.environ, MPLBACKEND='Agg', PYTHONUNBUFFERED='1'))
            code = process.returncode
            result = dict(name=name, status='passed' if code == 0 else 'failed', exit_code=code)
        except subprocess.TimeoutExpired:
            result = dict(name=name, status='failed', reason=f'Exceeded {timeout}s time limit')
        result.update(seconds=time.perf_counter()-start, log=log.name, command=command)
        report['checks'].append(result); save(out/'verification.json', report)
        if result['status'] != 'passed':
            raise RuntimeError(f'{name} failed; inspect {log}')
    try:
        report['source_checks'] = inspect_sources()
        report['bundle'] = inspect_bundle()
        manifest = json.loads(DEFAULT_MANIFEST.read_text())
        report['data'] = dataset_status(ROOT/'data/CICIoT23_archive_sample_v1', manifest)
        save(out/'verification.json', report)
        step('environment', [sys.executable, '-m', 'tools.check_environment'], 150)
        step('unit_tests', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], 180)
        step('saved_study', [sys.executable, '-m', 'tools.check_study_results', '--output', str(out/'study_integrity.json')], 90)
        if not a.quick:
            step('original_architecture_feedback_smoke', [sys.executable, '-m', 'tools.smoke_training',
                 '--original-model', '--reweight-fixture', '--output', str(out/'synthetic_training')], 300)
            step('authenticated_model_tcp', [sys.executable, '-m', 'tools.run_deployment_demo',
                 '--synthetic', '--output', str(out/'synthetic_tcp')], 240)
            step('crypto_telemetry_benchmark', [sys.executable, '-m', 'tools.benchmark_security',
                 '--iterations', '1000', '--output', str(out/'benchmarks')], 180)
            with __import__('tempfile').TemporaryDirectory(prefix='qarmor-verified-counters-') as temp:
                step('durable_crypto_benchmark', [sys.executable, '-m', 'tools.benchmark_security',
                     '--iterations', '500', '--store', str(Path(temp)/'state.sqlite'),
                     '--output', str(out/'benchmarks_durable')], 180)
        freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, timeout=30)
        if freeze.returncode:
            raise RuntimeError('Could not record dependency versions')
        (out/'requirements-verified.txt').write_text(freeze.stdout)
        report.update(status='passed_runnable_checks', full_verification=not a.quick,
                      new_CICIoT_training_or_quality_results=False,
                      remaining_blockers=[
                          *(['Matching raw archive/split CSVs were deleted and are not in GitHub.'] if report['data']['status'] != 'verified' else []),
                          'Raw packet-to-CICIoT2023 feature parity and live capture remain unverified.',
                          'Repeated training checkpoints and later XSS intervention files were not committed.'],
                      interpretation='Synthetic tests establish wiring only; published study scores remain historical bounded-sample evidence.')
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        save(out/'verification.json', report)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
