"""Local CPU crypto/telemetry processing microbenchmarks, no capture privileges."""
import argparse
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import time
from pathlib import Path
import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from crypto.secure_channel import SecureChannel,CounterStore,canonical
from crypto.authenticated_handshake import AuthenticatedHandshake
from crypto.secure_transport import process_telemetry
from game_theory.ids_training import save_json


def summary(values,payload_bytes):
    values=np.asarray(values,dtype=float)
    return dict(samples=len(values),median_us=float(np.median(values)*1e6),p95_us=float(np.percentile(values,95)*1e6),
                messages_per_second=float(1/values.mean()),MiB_per_second=float(payload_bytes/values.mean()/2**20))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--iterations',type=int,default=1000);p.add_argument('--output',default='results/benchmarks')
    p.add_argument('--sizes',type=int,nargs='+',default=[64,1024,16384])
    p.add_argument('--store',help='Optional SQLite path for durable state benchmark; otherwise fresh in-memory sessions')
    args=p.parse_args()
    if args.iterations<10 or any(s<1 for s in args.sizes):p.error('At least 10 iterations and positive payload sizes required')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows=[];raw={}
    for size in args.sizes:
        # Isolate AES-GCM's counter/AAD effect from envelope/state overhead.
        # Separate keys avoid reusing a GCM nonce across the paired conditions.
        no_aad_aes=AESGCM(os.urandom(32));aad_aes=AESGCM(os.urandom(32))
        aad_metadata=dict(context='QARMOR/telemetry/v1',session=os.urandom(32).hex(),
                          sender='agent',receiver='server')
        no_aad_enc=[];no_aad_dec=[];aad_enc=[];aad_dec=[]
        data=b'x'*size
        for i in range(args.iterations+25):
            counter=i+1
            no_aad_nonce=counter.to_bytes(12,'big')
            aad_nonce=counter.to_bytes(12,'big')
            aad=canonical(dict(aad_metadata,counter=counter))
            start=time.perf_counter();bare=no_aad_aes.encrypt(no_aad_nonce,data,None);mid=time.perf_counter()
            no_aad_aes.decrypt(no_aad_nonce,bare,None);stop=time.perf_counter()
            start2=time.perf_counter();protected=aad_aes.encrypt(aad_nonce,data,aad);mid2=time.perf_counter()
            aad_aes.decrypt(aad_nonce,protected,aad);stop2=time.perf_counter()
            if i>=25:
                no_aad_enc.append(mid-start);no_aad_dec.append(stop-mid)
                aad_enc.append(mid2-start2);aad_dec.append(stop2-mid2)
        bare_wire=size+12+16
        protected_envelope=canonical(dict(aad_metadata,counter=args.iterations+25,
                                           ciphertext=protected.hex()))
        for name,values,wire in [
                ('aesgcm_encrypt_no_counter_aad',no_aad_enc,bare_wire),
                ('aesgcm_decrypt_no_counter_aad',no_aad_dec,bare_wire),
                ('aesgcm_encrypt_counter_aad',aad_enc,len(protected_envelope)),
                ('aesgcm_decrypt_counter_aad',aad_dec,len(protected_envelope))]:
            rows.append(dict(kind=name,payload_bytes=size,wire_bytes=wire,
                             overhead_bytes=wire-size,
                             timing_scope='AESGCM operation only; AAD canonicalization excluded',
                             **summary(values,size)))
            raw[f'{name}_{size}']=values

        key=os.urandom(32);tx=SecureChannel(key,os.urandom(32).hex(),'agent','server',CounterStore(args.store or ':memory:'))
        rx=SecureChannel(key,tx.metadata['session'],'agent','server',CounterStore(args.store or ':memory:'))
        encrypt=[];decrypt=[];end=[]
        for i in range(args.iterations+25):
            start=time.perf_counter();packet=tx.encrypt(data);mid=time.perf_counter();rx.decrypt(packet);stop=time.perf_counter()
            if i>=25:encrypt.append(mid-start);decrypt.append(stop-mid);end.append(stop-start)
        for name,values in [('encrypt',encrypt),('decrypt',decrypt),('roundtrip',end)]:
            rows.append(dict(kind=name,payload_bytes=size,wire_bytes=len(packet),overhead_bytes=len(packet)-size,**summary(values,size)))
            raw[f'{name}_{size}']=values
        tx.store.close();rx.store.close()
    payload={'type':'telemetry','metrics':{'packets_sent':100,'encrypt_success':100,'encrypt_fail':0,'handshake_success':1}}
    key=os.urandom(32);session=os.urandom(32).hex();tx=SecureChannel(key,session,'agent','server',CounterStore(args.store or ':memory:'));rx=SecureChannel(key,session,'agent','server',CounterStore(args.store or ':memory:'))
    plain=[];secure=[]
    for i in range(args.iterations+25):
        start=time.perf_counter();json.loads(canonical(payload));mid=time.perf_counter()
        process_telemetry(rx,tx.encrypt(canonical(payload)));stop=time.perf_counter()
        if i>=25:plain.append(mid-start);secure.append(stop-mid)
    for name,values in [('telemetry_json',plain),('telemetry_authenticated',secure)]:
        rows.append(dict(kind=name,payload_bytes=len(canonical(payload)),wire_bytes=None,**summary(values,len(canonical(payload)))))
        raw[name]=values
    tx.store.close();rx.store.close()
    alice=x25519.X25519PrivateKey.generate();bob=x25519.X25519PrivateKey.generate()
    exchange=[];shared=None
    for i in range(args.iterations+25):
        start=time.perf_counter();shared=alice.exchange(bob.public_key());stop=time.perf_counter()
        if i>=25:exchange.append(stop-start)
    rows.append(dict(kind='x25519_shared_secret',payload_bytes=32,wire_bytes=None,
                     timing_scope='one ephemeral private-key/public-key exchange',**summary(exchange,32)))
    raw['x25519_exchange']=exchange
    transcript_hash=os.urandom(32);hkdf=[];derived=None
    for i in range(args.iterations+25):
        start=time.perf_counter()
        derived=HKDF(algorithm=hashes.SHA256(),length=32,salt=transcript_hash,
                     info=b'QARMOR/X25519-Ed25519/v1').derive(shared)
        stop=time.perf_counter()
        if i>=25:hkdf.append(stop-start)
    rows.append(dict(kind='hkdf_sha256',payload_bytes=32,wire_bytes=None,
                     timing_scope='HKDF-SHA256 from a precomputed X25519 shared secret',**summary(hkdf,32)))
    raw['hkdf_sha256']=hkdf
    ka,kb=Ed25519PrivateKey.generate(),Ed25519PrivateKey.generate();handshakes=[]
    for i in range(min(args.iterations,100)+5):
        start=time.perf_counter();a=AuthenticatedHandshake('a',ka,'b',kb.public_key());b=AuthenticatedHandshake('b',kb,'a',ka.public_key())
        b.accept(a.finish(b.respond(a.offer())));stop=time.perf_counter()
        if i>=5:handshakes.append(stop-start)
    rows.append(dict(kind='authenticated_handshake',payload_bytes=0,wire_bytes=None,**summary(handshakes,0)))
    raw['handshake']=handshakes
    usage=resource.getrusage(resource.RUSAGE_SELF)
    report=dict(measurement='Local CPU crypto/telemetry microbenchmarks. AES-GCM counter/AAD comparison times the AEAD call only; end-to-end telemetry excludes IDS inference, network scheduling and disk alert logging.',
                warmup=25,arguments=vars(args),platform=platform.platform(),machine=platform.machine(),python=platform.python_version(),
                cryptography=importlib.metadata.version('cryptography'),max_rss_native_units=usage.ru_maxrss,
                user_cpu_seconds=usage.ru_utime,system_cpu_seconds=usage.ru_stime,results=rows)
    save_json(out/'benchmark.json',report);save_json(out/'raw_seconds.json',raw)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out/'benchmark.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[r for r in rows if r['kind'] in ['telemetry_json','telemetry_authenticated']]
    fig,ax=plt.subplots();ax.bar([r['kind'] for r in selected],[r['median_us'] for r in selected]);ax.set(ylabel='Median local processing latency (µs)')
    fig.tight_layout();fig.savefig(out/'telemetry_latency.png');plt.close(fig)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
