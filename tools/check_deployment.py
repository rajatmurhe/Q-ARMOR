"""Exercise the actual frozen checkpoint, original preprocessing and encrypted alert path."""
import argparse
import json
import socket
import threading
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from Agent.inference import InferenceEngine, file_hash
from Agent.ids_agent import run_stream
from Server.server import handle_connection
from crypto.authenticated_handshake import AuthenticatedHandshake


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',default='deployment/archive_sample_seed42')
    p.add_argument('--output',default='results/deployment_v1')
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('Use a new result directory')
    out.mkdir(parents=True)
    start=time.perf_counter();engine=InferenceEngine(a.bundle);load_seconds=time.perf_counter()-start
    frame=pd.read_csv('data/CICIoT23_archive_sample_v1/validation/validation.csv').iloc[:128]
    records=frame.to_dict('records')
    prep=joblib.load('results/remaining_study_v2/ablation/42/preprocessing.pkl')
    raw=frame[prep['features']].replace([np.inf,-np.inf],np.nan).fillna(engine.medians)
    reference=prep['scaler'].transform(raw).astype(np.float32)
    np.testing.assert_array_equal(engine.transform(records),reference)
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
    result=dict(status='passed',validation_records=len(records),schema_transform_bit_identical=True,
        model_load_seconds=load_seconds,inference_batch_seconds=timings,
        steady_batch_median_seconds=float(np.median(timings[1:])),
        encrypted_socketpair_end_to_end_seconds=end_to_end,expected_alerts=expected,
        benign_predictions_suppressed=len(benign),agent=sent,server=received[0],
        bundle_manifest_sha256=file_hash(Path(a.bundle)/'manifest.json'),
        scope='Validation-feature replay through real model and authenticated socketpair; excludes packet extraction and network transit.')
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
