"""Mutual authenticated ephemeral X25519 exchange with pinned Ed25519 identities.

Three messages: offer -> signed full-transcript response -> signed finish.
Neither endpoint accepts a session until verifying the expected peer signature.
New local challenges and one-use state prevent replay across sessions/restarts.
Public-key pinning is provisioned out of band; received keys are never trusted.
"""
import hashlib
import os
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from .secure_channel import canonical, SecureChannel, CounterStore

PROTOCOL='QARMOR/X25519-Ed25519/v1'


def raw_public(key):
    return key.public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)


class AuthenticatedHandshake:
    def __init__(self, identity, signing_key, peer_identity, trusted_peer_key):
        if not identity or not peer_identity or identity==peer_identity:
            raise ValueError('Distinct identities required')
        self.identity=identity;self.signing_key=signing_key
        self.peer_identity=peer_identity;self.trusted_peer_key=trusted_peer_key
        self.ephemeral=x25519.X25519PrivateKey.generate()
        self.challenge=os.urandom(32).hex()
        self.state='new';self.transcript=None;self.key=None
        self._channel_store = CounterStore()
        self._channels = {}

    def offer(self):
        if self.state!='new':raise ValueError('Handshake already used')
        self.state='offered'
        return dict(protocol=PROTOCOL,initiator=self.identity,responder=self.peer_identity,
                    initiator_nonce=self.challenge,initiator_ephemeral=raw_public(self.ephemeral.public_key()).hex())

    @staticmethod
    def _validate_offer(offer):
        fields={'protocol','initiator','responder','initiator_nonce','initiator_ephemeral'}
        if not isinstance(offer,dict) or set(offer)!=fields or offer['protocol']!=PROTOCOL:
            raise ValueError('Invalid offer')
        for key in ['initiator_nonce','initiator_ephemeral']:
            if not isinstance(offer[key],str) or len(offer[key])!=64 or len(bytes.fromhex(offer[key]))!=32:
                raise ValueError('Invalid offer encoding')

    def respond(self, offer):
        if self.state!='new':raise ValueError('Handshake already used')
        self._validate_offer(offer)
        if offer['initiator']!=self.peer_identity or offer['responder']!=self.identity:
            raise ValueError('Wrong handshake identities')
        self.transcript=dict(offer,responder_nonce=self.challenge,
                             responder_ephemeral=raw_public(self.ephemeral.public_key()).hex())
        self.state='responded'
        return dict(transcript=self.transcript.copy(),
                    signature=self.signing_key.sign(b'responder\0'+canonical(self.transcript)).hex())

    def finish(self,response):
        if self.state!='offered':raise ValueError('No outstanding offer or handshake already used')
        if not isinstance(response,dict) or set(response)!={'transcript','signature'}:
            raise ValueError('Invalid response')
        transcript=response['transcript']
        if not isinstance(transcript,dict) or set(transcript)!={'protocol','initiator','responder','initiator_nonce','initiator_ephemeral','responder_nonce','responder_ephemeral'}:
            raise ValueError('Invalid transcript fields')
        expected=dict(protocol=PROTOCOL,initiator=self.identity,responder=self.peer_identity,
                      initiator_nonce=self.challenge,initiator_ephemeral=raw_public(self.ephemeral.public_key()).hex())
        if any(transcript[k]!=v for k,v in expected.items()):raise ValueError('Transcript does not bind outstanding offer')
        for k in ['responder_nonce','responder_ephemeral']:
            if not isinstance(transcript[k],str) or len(transcript[k])!=64 or len(bytes.fromhex(transcript[k]))!=32:
                raise ValueError('Invalid responder encoding')
        self.trusted_peer_key.verify(bytes.fromhex(response['signature']),b'responder\0'+canonical(transcript))
        self.transcript=transcript.copy()
        key=self._derive('responder_ephemeral')
        signature=self.signing_key.sign(b'initiator\0'+canonical(transcript)).hex()
        self.key=key;self.state='accepted'
        return dict(transcript=transcript.copy(),signature=signature)

    def accept(self,finish):
        if self.state!='responded':raise ValueError('No outstanding response or handshake already used')
        if not isinstance(finish,dict) or set(finish)!={'transcript','signature'} or finish['transcript']!=self.transcript:
            raise ValueError('Transcript mismatch')
        self.trusted_peer_key.verify(bytes.fromhex(finish['signature']),b'initiator\0'+canonical(self.transcript))
        key=self._derive('initiator_ephemeral')
        self.key=key;self.state='accepted'
        return key

    def _derive(self,peer_field):
        peer=x25519.X25519PublicKey.from_public_bytes(bytes.fromhex(self.transcript[peer_field]))
        secret=self.ephemeral.exchange(peer)
        digest=hashlib.sha256(canonical(self.transcript)).digest()
        return HKDF(algorithm=hashes.SHA256(),length=32,salt=digest,info=PROTOCOL.encode()).derive(secret)

    @property
    def session_id(self):
        if self.state!='accepted':raise ValueError('Peer has not authenticated')
        return hashlib.sha256(canonical(self.transcript)).hexdigest()

    def channel(self, sending=True, store=None):
        if self.state!='accepted':raise ValueError('Peer has not authenticated')
        if sending in self._channels:
            existing = self._channels[sending]
            if store is not None and store is not existing.store:
                raise ValueError('Cannot change counter store for an active direction')
            return existing
        sender,receiver=(self.identity,self.peer_identity) if sending else (self.peer_identity,self.identity)
        channel = SecureChannel(self.key,self.session_id,sender,receiver,store or self._channel_store)
        self._channels[sending] = channel
        return channel
