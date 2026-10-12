"""Authenticated counter envelopes with atomic SQLite replay/counter state.

Memory state is supported only for a fresh handshake session that is discarded
on restart. Persistent state must be retained without rollback, shared by all
writers of the same key/session, and never restored from stale backups.
"""
import hashlib
import json
import sqlite3
import threading
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

CONTEXT = 'QARMOR/telemetry/v1'
MAX_COUNTER = (1 << 32)-1  # Conservative per-direction GCM message budget; rotate before exhaustion.


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode('ascii')


class ReplayError(ValueError): pass


class CounterStore:
    """Atomic reservations survive crashes (gaps allowed); receiver commits after auth."""
    def __init__(self, path=':memory:'):
        self.db = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS counters (scope TEXT PRIMARY KEY, value INTEGER NOT NULL)')
        self.lock = threading.RLock()

    def _value(self, scope):
        row=self.db.execute('SELECT value FROM counters WHERE scope=?',(scope,)).fetchone()
        return row[0] if row else 0

    def reserve(self, scope):
        with self.lock:
            self.db.execute('BEGIN IMMEDIATE')
            try:
                value=self._value(scope)+1
                if value>MAX_COUNTER: raise OverflowError('Rotate session/key: message budget exhausted')
                self.db.execute('INSERT OR REPLACE INTO counters VALUES (?,?)',(scope,value))
                self.db.execute('COMMIT');return value
            except BaseException:
                self.db.execute('ROLLBACK');raise

    def authenticate(self, scope, counter, operation):
        with self.lock:
            self.db.execute('BEGIN IMMEDIATE')
            try:
                if counter <= self._value(scope): raise ReplayError('Duplicate or stale message counter')
                plaintext=operation()  # InvalidTag exits without advancing trusted replay state.
                self.db.execute('INSERT OR REPLACE INTO counters VALUES (?,?)',(scope,counter))
                self.db.execute('COMMIT');return plaintext
            except BaseException:
                self.db.execute('ROLLBACK');raise

    def close(self): self.db.close()


class SecureChannel:
    """One sender/session/direction. Key and context bound by HKDF and authenticated AAD.

    A single persistent CounterStore must cover every writer using the same
    master key. Do not fork/copy memory stores or resume a memory session.
    Reverse traffic uses a separate sender ID, hence a separate derived key.
    """
    def __init__(self, session_key, session_id, sender_id, receiver_id, store=None):
        if not isinstance(session_key, bytes) or len(session_key)!=32:
            raise ValueError('A 32-byte session key is required')
        if any(not isinstance(v,str) or not v or len(v)>256 for v in [session_id,sender_id,receiver_id]) or sender_id==receiver_id:
            raise ValueError('Nonempty session and distinct sender/receiver identities required')
        self.metadata=dict(context=CONTEXT,session=session_id,sender=sender_id,receiver=receiver_id)
        key=HKDF(algorithm=hashes.SHA256(),length=32,salt=None,
                 info=canonical(self.metadata)).derive(session_key)
        self.aes=AESGCM(key)
        self.scope=hashlib.sha256(key).hexdigest()
        self.store=store or CounterStore()

    def _aad(self, counter):
        return canonical(dict(self.metadata,counter=counter))

    def encrypt(self, data):
        if not isinstance(data,bytes): raise TypeError('Plaintext must be bytes')
        counter=self.store.reserve('send:'+self.scope)
        nonce=counter.to_bytes(12,'big')
        return canonical(dict(self.metadata,counter=counter,
                              ciphertext=self.aes.encrypt(nonce,data,self._aad(counter)).hex()))

    def decrypt(self, message):
        if not isinstance(message,bytes) or len(message)>16*1024*1024:
            raise ValueError('Invalid envelope size/type')
        try:
            envelope=json.loads(message)
            if not isinstance(envelope,dict) or set(envelope)!=set(self.metadata)|{'counter','ciphertext'}:
                raise ValueError('Envelope fields mismatch')
            if canonical(envelope)!=message: raise ValueError('Noncanonical envelope')
            if any(envelope[k]!=v for k,v in self.metadata.items()): raise ValueError('Session or peer mismatch')
            counter=envelope['counter']
            if type(counter) is not int or not 1<=counter<=MAX_COUNTER: raise ValueError('Invalid counter')
            ciphertext=bytes.fromhex(envelope['ciphertext'])
            if len(ciphertext)<16: raise ValueError('Missing authentication tag')
        except (TypeError,KeyError,json.JSONDecodeError,UnicodeError) as exc:
            raise ValueError('Malformed authenticated envelope') from exc
        return self.store.authenticate('receive:'+self.scope,counter,
            lambda:self.aes.decrypt(counter.to_bytes(12,'big'),ciphertext,self._aad(counter)))
