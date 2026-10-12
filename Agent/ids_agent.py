"""Run checkpoint inference on compatible feature CSV/JSONL and send signed-session alerts."""
import argparse
import csv
import json
import socket
import sys
import time
from itertools import islice
from pathlib import Path
from Agent.inference import InferenceEngine
from crypto.identity_config import handshake_from_environment
from crypto.secure_transport import connect_authenticated, send_frame, receive_frame


def feature_records(path, format='csv'):
    stream = sys.stdin if str(path) == '-' else Path(path).open(newline='')
    try:
        if format == 'csv':
            reader = csv.DictReader(stream)
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError('CSV needs unique column names')
            yield from reader
        else:
            for number, line in enumerate(stream, 1):
                if len(line) > 1024 * 1024:
                    raise ValueError(f'JSONL record {number} exceeds size limit')
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f'JSONL record {number} must be an object')
                    yield value
    finally:
        if stream is not sys.stdin:
            stream.close()


def run_stream(engine, records, sock, handshake, batch_size=128, confidence_threshold=0.0):
    if batch_size < 1 or not 0 <= confidence_threshold <= 1:
        raise ValueError('Invalid batch size or confidence threshold')
    sender, receiver = connect_authenticated(sock, handshake, True)
    counts = dict(records_processed=0, alerts_sent=0, packets_sent=0,
                  encrypt_success=0, encrypt_fail=0, handshake_success=1)
    inference_seconds = 0.0
    records = iter(records)
    while True:
        batch = list(islice(records, batch_size))
        if not batch:
            break
        started = time.perf_counter()
        predictions = engine.predict(batch)
        inference_seconds += time.perf_counter() - started
        for row, prediction in zip(batch, predictions):
            if prediction['malicious'] and prediction['confidence'] >= confidence_threshold:
                payload = dict(type='alert', timestamp=time.time(), label=prediction['label'],
                    confidence=prediction['confidence'], record_index=counts['records_processed'],
                    source='ciciot2023_feature_record')
                for key in ['src_ip', 'dst_ip']:
                    if row.get(key):
                        payload[key] = str(row[key])
                send_frame(sock, sender.encrypt(json.dumps(payload).encode()))
                counts['alerts_sent'] += 1
                counts['packets_sent'] += 1
                counts['encrypt_success'] += 1
            counts['records_processed'] += 1
    counts['inference_seconds'] = inference_seconds
    send_frame(sock, sender.encrypt(json.dumps(dict(type='telemetry',metrics=counts,request_ack=True)).encode()))
    acknowledgement = json.loads(receiver.decrypt(receive_frame(sock)))
    if (acknowledgement.get('type') != 'telemetry'
            or acknowledgement.get('metrics',{}).get('alerts_received') != counts['alerts_sent']):
        raise ValueError('Receiver did not acknowledge all alerts')
    counts['receiver_acknowledged'] = True
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',default='deployment/archive_sample_seed42')
    p.add_argument('--features',required=True,help='Compatible feature CSV/JSONL; - reads standard input')
    p.add_argument('--format',choices=['csv','jsonl'],default='csv')
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=9000)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--limit',type=int);p.add_argument('--confidence-threshold',type=float,default=0.0)
    args = p.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1) or not 0 <= args.confidence_threshold <= 1:
        p.error('Batch size and limit must be positive; threshold must be in [0,1]')
    handshake = handshake_from_environment()
    engine = InferenceEngine(args.bundle)
    records = feature_records(args.features,args.format)
    if args.limit is not None:
        records = islice(records,args.limit)
    with socket.create_connection((args.host,args.port),timeout=30) as sock:
        result = run_stream(engine,records,sock,handshake,args.batch_size,args.confidence_threshold)
    print(json.dumps(result,indent=2))

if __name__ == '__main__':main()
