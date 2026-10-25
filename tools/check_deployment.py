"""Exercise the actual frozen checkpoint, original preprocessing and encrypted alert path."""
import argparse
import json
import socket
import threading
import time
from pathlib import Path
import numpy as np
import pandas as pd
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from Agent.inference import InferenceEngine, file_hash
from Agent.ids_agent import run_stream
from Server.server import handle_connection
from crypto.authenticated_handshake import AuthenticatedHandshake


def load_validation(path, records):
    path = Path(path)
    if records < 1:
        raise ValueError('Record limit must be positive')
    if not path.is_file():
        raise FileNotFoundError(f'Validation CSV unavailable: {path}. Restore the frozen splits with python -m tools.restore_data --archive /path/to/CSV.zip')
    frame = pd.read_csv(path, nrows=records)
    if frame.empty:
        raise ValueError('Validation CSV has no records')
    if '_qarmor_source' in frame and frame['_qarmor_source'].eq('synthetic_smoke').any():
        raise ValueError('Use tools.run_deployment_demo --synthetic for synthetic fixtures')
    return frame


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',default='deployment/archive_sample_seed42')
    p.add_argument('--output',default='results/deployment_v1')
    p.add_argument('--features',type=Path,default=Path('data/CICIoT23_archive_sample_v1/validation/validation.csv'))
    p.add_argument('--records',type=int,default=128)
    p.add_argument('--reference-preprocessing',type=Path,
                   help='Optional trusted local training preprocessing.pkl for independent transform parity')
    a=p.parse_args();out=Path(a.output)
    try:
        frame=load_validation(a.features,a.records)
    except (FileNotFoundError, ValueError) as exc:
        p.error(str(exc))
    if a.reference_preprocessing and not a.reference_preprocessing.is_file():
        raise FileNotFoundError(a.reference_preprocessing)
    if out.exists():raise FileExistsError('Use a new result directory')
    out.mkdir(parents=True)
    start=time.perf_counter();engine=InferenceEngine(a.bundle);load_seconds=time.perf_counter()-start
    records=frame.to_dict('records')
    parity=None
    if a.reference_preprocessing:
        import joblib
        prep=joblib.load(a.reference_preprocessing)  # explicitly supplied trusted local artifact
        if prep['features'] != engine.features:
            raise ValueError('Reference preprocessing feature order differs from deployment bundle')
        raw=frame[prep['features']].replace([np.inf,-np.inf],np.nan).fillna(engine.medians)
        reference=prep['scaler'].transform(raw).astype(np.float32)
        np.testing.assert_array_equal(engine.transform(records),reference)
        parity=True
    timings=[]
    for _ in range(4):
        start=time.perf_counter();predictions=engine.predict(records);timings.append(time.perf_counter()-start)
    expected=sum(x['malicious'] for x in predictions)
    first,second=Ed25519PrivateKey.generate(),Ed25519PrivateKey.generate()
    client,server=socket.socketpair();client.settimeout(30);server.settimeout(30)
    received=[];errors=[]
    def listen():
        try:
            with server:
                received.append(handle_connection(server,AuthenticatedHandshake('server',second,'agent',first.public_key()),out/'receiver'))
        except BaseException as exc:errors.append(str(exc))
    worker=threading.Thread(target=listen);worker.start();start=time.perf_counter()
    with client:
        sent=run_stream(engine,records,client,AuthenticatedHandshake('agent',first,'server',second.public_key()))
    end_to_end=time.perf_counter()-start;worker.join(35)
    if worker.is_alive() or errors:raise RuntimeError(f'Receiver failure: {errors}')
    alerts=[json.loads(l) for l in (out/'receiver/alerts.jsonl').read_text().splitlines()] if expected else []
    expected_predictions=[x for x in predictions if x['malicious']]
    assert sent['alerts_sent']==received[0]['alerts_received']==expected==len(alerts)
    assert [r['label'] for r in alerts]==[r['label'] for r in expected_predictions]
    for delivered,predicted in zip(alerts,expected_predictions):
        assert abs(delivered['confidence']-predicted['confidence']) < 1e-6
    benign=[i for i,x in enumerate(predictions) if not x['malicious']]
    assert not set(benign)&{x['record_index'] for x in alerts}
    result=dict(status='passed',validation_records=len(records),schema_transform_bit_identical=parity,
        reference_preprocessing_status='verified' if parity else 'not_run_no_reference_artifact',
        features_path=str(a.features.resolve()),features_sha256=file_hash(a.features),
        dataset_identity='provided feature CSV; compare hash to frozen data manifest before claiming study-split identity',
        model_load_seconds=load_seconds,inference_batch_seconds=timings,
        steady_batch_median_seconds=float(np.median(timings[1:])),
        encrypted_socketpair_end_to_end_seconds=end_to_end,expected_alerts=expected,
        benign_predictions_suppressed=len(benign),agent=sent,server=received[0],
        bundle_manifest_sha256=file_hash(Path(a.bundle)/'manifest.json'),
        scope='Validation-feature replay through real model and authenticated socketpair; excludes packet extraction and network transit.')
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
