import os
from cryptography.hazmat.primitives.asymmetric import x25519, ed25519
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class HybridCrypto:

    def __init__(self):
        self.private_key = x25519.X25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()

        self.sign_key = ed25519.Ed25519PrivateKey.generate()
        self.verify_key = self.sign_key.public_key()

        self.session_key = None

    # --------------------------------------------------
    # KEY EXCHANGE (IMPORTANT)
    # --------------------------------------------------
    def export_public_bytes(self):
        return self.public_key.public_bytes_raw()

    def import_public_bytes(self, data):
        return x25519.X25519PublicKey.from_public_bytes(data)

    def derive_shared_key(self, peer_public_bytes):
        peer = self.import_public_bytes(peer_public_bytes)

        shared = self.private_key.exchange(peer)

        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"ids-session"
        )

        self.session_key = hkdf.derive(shared)
        return self.session_key

    # --------------------------------------------------
    # ENCRYPT / DECRYPT
    # --------------------------------------------------
    def encrypt(self, data: bytes):
        nonce = os.urandom(12)
        aes = AESGCM(self.session_key)
        return nonce + aes.encrypt(nonce, data, None)

    def decrypt(self, ciphertext: bytes):
        nonce = ciphertext[:12]
        ct = ciphertext[12:]
        aes = AESGCM(self.session_key)
        return aes.decrypt(nonce, ct, None)

    # --------------------------------------------------
    # SIGNATURE
    # --------------------------------------------------
    def sign(self, data):
        return self.sign_key.sign(data)

    def verify(self, signature, data):
        self.verify_key.verify(signature, data)
