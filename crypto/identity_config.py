"""Load pre-provisioned identity and pinned peer PEM keys, never network keys."""
import os
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from .authenticated_handshake import AuthenticatedHandshake


def handshake_from_environment():
    names=['QARMOR_IDENTITY','QARMOR_PEER_IDENTITY','QARMOR_SIGNING_KEY','QARMOR_TRUSTED_PEER_KEY']
    missing=[name for name in names if not os.environ.get(name)]
    if missing:raise ValueError('Provision authenticated transport settings: '+', '.join(missing))
    private=serialization.load_pem_private_key(Path(os.environ['QARMOR_SIGNING_KEY']).read_bytes(),password=None)
    peer=serialization.load_pem_public_key(Path(os.environ['QARMOR_TRUSTED_PEER_KEY']).read_bytes())
    if not isinstance(private,ed25519.Ed25519PrivateKey) or not isinstance(peer,ed25519.Ed25519PublicKey):
        raise ValueError('Ed25519 identity and trusted peer keys required')
    return AuthenticatedHandshake(os.environ['QARMOR_IDENTITY'],private,
                                  os.environ['QARMOR_PEER_IDENTITY'],peer)
