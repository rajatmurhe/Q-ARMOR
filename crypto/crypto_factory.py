"""Legacy factory; live transport uses secure_transport."""
CRYPTO_TYPE = "pqc"  # Historical alias for classical X25519, NOT post-quantum.


def get_crypto():
    if CRYPTO_TYPE == "pqc":
        from .pqc_crypto import HybridCrypto
        return HybridCrypto()
    from .classical_crypto import ClassicalCrypto
    return ClassicalCrypto()
