import copy
import json
import os
import socket
import tempfile
import threading
import unittest
from cryptography.exceptions import InvalidTag, InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from crypto.secure_channel import SecureChannel, CounterStore, ReplayError, canonical, MAX_COUNTER
from crypto.authenticated_handshake import AuthenticatedHandshake
from crypto.secure_transport import connect_authenticated, send_frame, receive_frame, process_telemetry


class SecurityTests(unittest.TestCase):
    def channels(self, key=None, session='s', sender='a'):
        key=key or os.urandom(32)
        return SecureChannel(key,session,sender,'b'),SecureChannel(key,session,sender,'b')

    def peers(self, wrong=False):
        a,b=Ed25519PrivateKey.generate(),Ed25519PrivateKey.generate()
        return (AuthenticatedHandshake('a',a,'b',b.public_key()),
                AuthenticatedHandshake('b',b,'a',(Ed25519PrivateKey.generate() if wrong else a).public_key()))

    def test_message_success_and_exact_replay(self):
        a,b=self.channels();packet=a.encrypt(b'hello')
        self.assertEqual(b.decrypt(packet),b'hello')
        with self.assertRaises(ReplayError):b.decrypt(packet)

    def test_old_counter_different_authenticated_message(self):
        a,b=self.channels();packet=a.encrypt(b'first');b.decrypt(packet)
        e=json.loads(packet);counter=e['counter']
        e['ciphertext']=a.aes.encrypt(counter.to_bytes(12,'big'),b'different',a._aad(counter)).hex()
        with self.assertRaises(ReplayError):b.decrypt(canonical(e))

    def test_tampering_does_not_advance_state(self):
        for target in ['ciphertext','tag','counter']:
            a,b=self.channels();packet=a.encrypt(b'hello');e=json.loads(packet)
            if target=='counter':e['counter']+=100
            else:
                ct=bytearray.fromhex(e['ciphertext']);ct[-1 if target=='tag' else 0]^=1;e['ciphertext']=ct.hex()
            with self.assertRaises(InvalidTag):b.decrypt(canonical(e))
            self.assertEqual(b.decrypt(packet),b'hello')

    def test_wrong_key(self):
        a,_=self.channels();_,b=self.channels()
        with self.assertRaises(InvalidTag):b.decrypt(a.encrypt(b'hello'))

    def test_session_sender_isolation_and_aad(self):
        key=os.urandom(32);a,b=self.channels(key);packet=a.encrypt(b'hello')
        for field in ['session','sender','receiver','context']:
            e=json.loads(packet);e[field]='evil'
            with self.assertRaises(ValueError):b.decrypt(canonical(e))
        _,other=self.channels(key,'other')
        self.assertNotEqual(a.scope,other.scope)
        with self.assertRaises(ValueError):other.decrypt(packet)
        c,d=self.channels(key,'s','c');self.assertEqual(d.decrypt(c.encrypt(b'new')),b'new')
        self.assertEqual(b.decrypt(packet),b'hello')
        self.assertNotEqual(a.scope,c.scope)

    def test_invalid_counters_and_envelopes(self):
        a,b=self.channels();packet=a.encrypt(b'x')
        for counter in [0,-1,True,1.5,MAX_COUNTER+1,'1']:
            e=json.loads(packet);e['counter']=counter
            with self.assertRaises(ValueError):b.decrypt(canonical(e))
        for packet in [b'',b'{}',b'[]',b'null',b'bad']:
            with self.assertRaises(ValueError):b.decrypt(packet)

    def test_persistence_and_nonce_uniqueness(self):
        key=os.urandom(32)
        with tempfile.TemporaryDirectory() as folder:
            path=folder+'/state.sqlite'
            store=CounterStore(path);a=SecureChannel(key,'s','a','b',store)
            packets=[a.encrypt(b'x') for _ in range(100)]
            b=SecureChannel(key,'s','a','b',store);b.decrypt(packets[-1]);store.close()
            store=CounterStore(path);a=SecureChannel(key,'s','a','b',store);b=SecureChannel(key,'s','a','b',store)
            packets.append(a.encrypt(b'x'))
            self.assertEqual(len({json.loads(p)['counter'] for p in packets}),101)
            with self.assertRaises(ReplayError):b.decrypt(packets[-2])
            self.assertEqual(b.decrypt(packets[-1]),b'x');store.close()

    def test_concurrent_instances_share_durable_counters(self):
        key=os.urandom(32)
        with tempfile.TemporaryDirectory() as folder:
            path=folder+'/state.sqlite'
            stores=[CounterStore(path) for _ in range(4)]
            channels=[SecureChannel(key,'s','a','b',store) for store in stores]
            packets=[];errors=[]
            def send(channel):
                try:
                    for _ in range(25):packets.append(channel.encrypt(b'x'))
                except Exception as exc:errors.append(exc)
            threads=[threading.Thread(target=send,args=(c,)) for c in channels]
            for thread in threads:thread.start()
            for thread in threads:thread.join(10)
            self.assertEqual(errors,[])
            self.assertEqual(sorted(json.loads(p)['counter'] for p in packets),list(range(1,101)))
            for store in stores:store.close()

    def test_handshake_channel_accessor_reuses_state(self):
        a,b=self.peers();b.accept(a.finish(b.respond(a.offer())))
        first=a.channel(True).encrypt(b'one');second=a.channel(True).encrypt(b'two')
        self.assertEqual(json.loads(first)['counter'],1)
        self.assertEqual(json.loads(second)['counter'],2)
        with self.assertRaises(ValueError):a.channel(True,CounterStore())

    def test_counter_exhaustion(self):
        a,_=self.channels();a.store.db.execute('INSERT INTO counters VALUES (?,?)',('send:'+a.scope,MAX_COUNTER))
        with self.assertRaises(OverflowError):a.encrypt(b'x')

    def test_handshake_success_bidirectional(self):
        a,b=self.peers();offer=a.offer();response=b.respond(offer);finish=a.finish(response)
        self.assertEqual(a.key,b.accept(finish));self.assertEqual(a.session_id,b.session_id)
        self.assertEqual(b.channel(False).decrypt(a.channel(True).encrypt(b'hi')),b'hi')
        self.assertNotEqual(a.channel(True).scope,b.channel(True).scope)

    def test_handshake_invalid_peer_signature(self):
        a,b=self.peers(wrong=True);response=b.respond(a.offer());finish=a.finish(response)
        with self.assertRaises(InvalidSignature):b.accept(finish)
        self.assertIsNone(b.key)
        a,b=self.peers();response=b.respond(a.offer());response['signature']='00'*64
        with self.assertRaises(InvalidSignature):a.finish(response)
        self.assertIsNone(a.key)

    def test_mismatched_pinned_responder_key(self):
        a,b=self.peers()
        a.trusted_peer_key=Ed25519PrivateKey.generate().public_key()
        response=b.respond(a.offer())
        with self.assertRaises(InvalidSignature):a.finish(response)
        self.assertIsNone(a.key)

    def test_wrong_identity_transcript_and_ephemeral(self):
        for field in ['initiator','responder','initiator_nonce','initiator_ephemeral','responder_nonce','responder_ephemeral','protocol']:
            a,b=self.peers();response=b.respond(a.offer())
            response=copy.deepcopy(response);response['transcript'][field]=('00'*32 if 'nonce' in field or 'ephemeral' in field else 'evil')
            with self.assertRaises((ValueError,InvalidSignature)):a.finish(response)
            self.assertIsNone(a.key)
        a,b=self.peers();offer=a.offer();offer['responder']='c'
        with self.assertRaises(ValueError):b.respond(offer)

    def test_handshake_replay(self):
        a,b=self.peers();response=b.respond(a.offer());finish=a.finish(response);b.accept(finish)
        with self.assertRaises(ValueError):b.accept(finish)
        with self.assertRaises(ValueError):a.finish(response)
        a2,b2=self.peers();a2.offer()
        with self.assertRaises(ValueError):a2.finish(response)
        # Replaying an old offer creates a fresh responder challenge; old finish cannot match.
        a3,b3=self.peers();old_offer=dict(response['transcript']);old_offer.pop('responder_nonce');old_offer.pop('responder_ephemeral')
        b3.respond(old_offer)
        with self.assertRaises(ValueError):b3.accept(finish)

    def test_socketpair_telemetry(self):
        a,b=self.peers();left,right=socket.socketpair();left.settimeout(5);right.settimeout(5)
        errors=[];received=[]
        def server():
            try:
                _,rx=connect_authenticated(right,b,False)
                received.append(process_telemetry(rx,receive_frame(right)))
            except Exception as exc:errors.append(exc)
        thread=threading.Thread(target=server);thread.start()
        try:
            tx,_=connect_authenticated(left,a,True)
            send_frame(left,tx.encrypt(canonical({'type':'telemetry','metrics':{'packets_sent':1}})))
            thread.join(5);self.assertFalse(thread.is_alive());self.assertEqual(errors,[])
            self.assertEqual(received[0]['metrics']['packets_sent'],1)
        finally:left.close();right.close()

if __name__=='__main__':unittest.main()
