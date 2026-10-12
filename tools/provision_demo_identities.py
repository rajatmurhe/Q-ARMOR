"""Create LOCAL DEMO Ed25519 identities. Transfer/verify peer public keys out of band."""
import argparse
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);args=p.parse_args()
    folder=Path(args.directory).expanduser();folder.mkdir(parents=True,exist_ok=True,mode=0o700)
    for name in ['agent','server']:
        key=Ed25519PrivateKey.generate()
        private=folder/(name+'.private.pem');public=folder/(name+'.public.pem')
        if private.exists() or public.exists():raise FileExistsError('Refusing to overwrite existing identity')
        # Exclusive creation; private key file is never briefly world-readable.
        import os
        fd=os.open(private,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as stream:
            stream.write(key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
        public.write_bytes(key.public_key().public_bytes(serialization.Encoding.PEM,serialization.PublicFormat.SubjectPublicKeyInfo))
        print(f'{name}: private key {private}; pinned public key {public}')

if __name__=='__main__':main()
