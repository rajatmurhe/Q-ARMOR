from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
import os


class ClassicalCrypto:

    def __init__(self):
        self.key = AESGCM.generate_key(bit_length=256)
        self.aes = AESGCM(self.key)

        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()

    def encrypt(self, data: bytes):
        nonce = os.urandom(12)
        return nonce + self.aes.encrypt(nonce, data, None)

    def decrypt(self, data: bytes):
        nonce, ct = data[:12], data[12:]
        return self.aes.decrypt(nonce, ct, None)

    def sign(self, msg: bytes):
        return self.private_key.sign(
            msg,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )

    def verify(self, msg, sig):
        self.public_key.verify(sig, msg,
                               padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                               salt_length=padding.PSS.MAX_LENGTH),
                               hashes.SHA256())
