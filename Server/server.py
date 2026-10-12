"""Authenticated alert receiver with configurable output and acknowledgement."""
import argparse
import json
import math
import os
import socket
from pathlib import Path
from crypto.identity_config import handshake_from_environment
from crypto.secure_transport import connect_authenticated, receive_frame, send_frame, process_telemetry


def handle_connection(conn, handshake, output):
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    sender, receiver = connect_authenticated(conn,handshake,False)
    counts = dict(alerts_received=0,telemetry_received=0,handshake_success=1)
    while True:
        try:
            data = process_telemetry(receiver,receive_frame(conn))
        except EOFError:
            return counts
        if data['type'] == 'alert':
            if (not isinstance(data.get('label'),str) or not data['label']
                    or not isinstance(data.get('confidence'),(int,float))
                    or not math.isfinite(data['confidence']) or not 0 <= data['confidence'] <= 1):
                raise ValueError('Invalid alert schema')
            with (output/'alerts.jsonl').open('a') as stream:
                stream.write(json.dumps(data,allow_nan=False)+'\n')
            counts['alerts_received'] += 1
        else:
            if not isinstance(data.get('metrics'),dict):
                raise ValueError('Invalid telemetry metrics')
            temporary = output/'metrics.json.tmp'
            temporary.write_text(json.dumps(data['metrics'],allow_nan=False,indent=2)+'\n')
            os.replace(temporary,output/'metrics.json')
            counts['telemetry_received'] += 1
            if data.get('request_ack'):
                send_frame(conn,sender.encrypt(json.dumps(dict(type='telemetry',metrics=counts)).encode()))
                return counts


def serve(host='127.0.0.1',port=9000,output='results/live_receiver',once=False):
    handshake_from_environment()
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        sock.bind((host,port));sock.listen(5)
        print(f'Authenticated server listening on {host}:{sock.getsockname()[1]}',flush=True)
        while True:
            conn, addr = sock.accept()
            with conn:
                conn.settimeout(30)
                try:
                    counts = handle_connection(conn,handshake_from_environment(),output)
                    print(json.dumps(counts),flush=True)
                except Exception as exc:
                    print(f'Connection {addr} failed: {exc}',flush=True)
                    if once:
                        raise
            if once:
                return


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=9000)
    p.add_argument('--output',default='results/live_receiver');p.add_argument('--once',action='store_true')
    a=p.parse_args();serve(a.host,a.port,a.output,a.once)

if __name__=='__main__':main()
