"""Real checkpoint + authenticated TCP demo, with explicit synthetic fallback mode."""
import argparse
import hashlib
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from tools.demo_features import write_synthetic_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='results/deployment_tcp_demo')
    p.add_argument('--bundle', default='deployment/archive_sample_seed42')
    inputs = p.add_mutually_exclusive_group()
    inputs.add_argument('--features', help='Existing compatible feature CSV')
    inputs.add_argument('--synthetic', action='store_true',
                        help='Explicit integration-only numeric fixtures, never CICIoT2023 results')
    p.add_argument('--records', type=int, default=128)
    args = p.parse_args()
    if args.records < 1:
        p.error('--records must be positive')
    bundle = Path(args.bundle).expanduser().resolve()
    source = Path(args.features or 'data/CICIoT23_archive_sample_v1/validation/validation.csv').expanduser().resolve()
    if not args.synthetic and not source.is_file():
        p.error(f'Feature CSV missing: {source}. Restore matching data, provide --features, or explicitly use --synthetic for an integration smoke only.')
    out = Path(args.output).expanduser().resolve()
    if out.exists():
        raise FileExistsError('Use a new output directory')
    out.mkdir(parents=True)
    evidence = ('synthetic integration fixture; no dataset-quality evaluation'
                if args.synthetic else 'provided feature CSV; not a new training experiment')
    status = dict(status='running', evidence=evidence, synthetic=args.synthetic)
    try:
        if args.synthetic:
            source = write_synthetic_features(bundle, out/'SYNTHETIC_features.csv', args.records)
        source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(prefix='qarmor-deploy-keys-') as folder:
            subprocess.run([sys.executable, '-m', 'tools.provision_demo_identities', '--directory', folder],
                           cwd=PROJECT_ROOT, check=True, capture_output=True, timeout=30)
            def environment(role, peer):
                return dict(os.environ, QARMOR_IDENTITY=role, QARMOR_PEER_IDENTITY=peer,
                            QARMOR_SIGNING_KEY=str(Path(folder)/(role+'.private.pem')),
                            QARMOR_TRUSTED_PEER_KEY=str(Path(folder)/(peer+'.public.pem')))
            server = subprocess.Popen([sys.executable, '-u', '-m', 'Server.server', '--port', '0',
                                       '--once', '--output', str(out/'receiver')], cwd=PROJECT_ROOT,
                                      env=environment('server', 'agent'), stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True)
            try:
                lines = queue.Queue()
                threading.Thread(target=lambda: lines.put(server.stdout.readline()), daemon=True).start()
                ready = lines.get(timeout=60)
                if not ready.startswith('Authenticated server listening on 127.0.0.1:'):
                    raise RuntimeError('Server did not start: '+ready)
                port = ready.strip().rsplit(':', 1)[1]
                agent = subprocess.run([sys.executable, '-u', '-m', 'Agent.ids_agent', '--port', port,
                                        '--bundle', str(bundle), '--features', str(source),
                                        '--limit', str(args.records)], cwd=PROJECT_ROOT,
                                       env=environment('agent', 'server'), capture_output=True,
                                       text=True, timeout=180)
                (out/'agent.log').write_text(agent.stdout+agent.stderr)
                remaining, _ = server.communicate(timeout=30)
                (out/'server.log').write_text(ready+remaining)
                if agent.returncode or server.returncode:
                    raise RuntimeError('CLI demonstration failed; inspect saved logs')
                metrics = json.loads((out/'receiver/metrics.json').read_text())
                alerts = [json.loads(line) for line in (out/'receiver/alerts.jsonl').read_text().splitlines()] if (out/'receiver/alerts.jsonl').exists() else []
                if metrics['records_processed'] != args.records:
                    raise RuntimeError(f"Expected {args.records} records; processed {metrics['records_processed']}")
                if metrics['alerts_sent'] != len(alerts):
                    raise RuntimeError('Receiver alert count differs from sender')
                if args.synthetic and any(row.get('source') != 'synthetic_smoke' for row in alerts):
                    raise RuntimeError('Synthetic provenance was lost during delivery')
                status.update(status='passed', records=args.records, alerts_received=len(alerts),
                              authenticated_loopback_tcp=True, server_exit=server.returncode,
                              agent_exit=agent.returncode, features_sha256=source_digest,
                              private_keys_removed_on_exit=True)
            finally:
                if server.poll() is None:
                    server.terminate()
                    try:
                        server.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        server.kill(); server.wait()
                if server.stdout:
                    server.stdout.close()
    except BaseException as exc:
        status.update(status='failed', error=repr(exc))
        raise
    finally:
        (out/'verification.json').write_text(json.dumps(status, indent=2)+'\n')
    print(json.dumps(status, indent=2))

if __name__ == '__main__':
    main()
