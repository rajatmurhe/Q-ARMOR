"""Environment audit with bounded isolated imports and authenticated crypto smoke."""
import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
CORE = ['numpy','pandas','sklearn','imblearn','tensorflow','cryptography','joblib','matplotlib','psutil']
EXTRAS = {'core': [], 'notebook': ['seaborn'], 'legacy': ['seaborn','scapy','streamlit']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['core','notebook','legacy'], default='core',
                        help='Core requirements.txt is sufficient for training/inference/transport; legacy extras are optional')
    args = parser.parse_args()
    print('Q-ARMOR ENVIRONMENT CHECK:', args.profile, flush=True)
    print(sys.version,platform.platform(),platform.machine(),flush=True)
    failed=[]
    for name in CORE + EXTRAS[args.profile]:
        code=f'import {name} as m; print(getattr(m,"__version__","unknown"))'
        try:
            r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=45)
            if r.returncode:raise RuntimeError(r.stderr.strip())
            print('OK',name,r.stdout.strip(),flush=True)
        except Exception as exc:failed.append(name);print('FAIL',name,str(exc),flush=True)
    try:
        from game_theory.stackelberg import StackelbergSecurityGame
        result=StackelbergSecurityGame(['A','B'],severity={'A':1.,'B':.8}).solve({'A':.4,'B':.8})
        print('Stackelberg smoke:',result['recommended_defense'],flush=True)
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from crypto.authenticated_handshake import AuthenticatedHandshake
        from crypto.secure_channel import ReplayError
        akey,bkey=Ed25519PrivateKey.generate(),Ed25519PrivateKey.generate()
        a=AuthenticatedHandshake('a',akey,'b',bkey.public_key());b=AuthenticatedHandshake('b',bkey,'a',akey.public_key())
        b.accept(a.finish(b.respond(a.offer())));assert a.key==b.key
        tx,rx=a.channel(True),b.channel(False);packet=tx.encrypt(b'Q-ARMOR authenticated smoke')
        assert rx.decrypt(packet)==b'Q-ARMOR authenticated smoke'
        try:rx.decrypt(packet)
        except ReplayError:pass
        else:raise AssertionError('Replay accepted')
        print('Authenticated handshake/AES/replay smoke: OK',flush=True)
    except Exception as exc:failed.append('core smoke');print('FAIL core smoke',str(exc),flush=True)
    if failed:print('CHECK FAILED:',failed,flush=True);return 1
    import tensorflow as tf
    print('TensorFlow devices:',tf.config.list_physical_devices(),flush=True)
    print('CHECK PASSED',flush=True);return 0

if __name__=='__main__':raise SystemExit(main())
