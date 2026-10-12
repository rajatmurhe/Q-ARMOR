import csv
import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from Agent.feature_extractor import extract_features
from Agent.ids_agent import feature_records, run_stream
from Server.server import handle_connection
from crypto.authenticated_handshake import AuthenticatedHandshake


class DeploymentTests(unittest.TestCase):
    def test_feature_order_imputation_and_missing_schema(self):
        self.assertEqual(extract_features({'b':'3','a':'nan'},['a','b'],{'a':2}),[2,3])
        with self.assertRaisesRegex(ValueError,'Missing required'):
            extract_features({'a':1},['a','b'])
        with self.assertRaisesRegex(ValueError,'must be numeric'):
            extract_features({'a':'garbage'},['a'],{'a':2})
        with self.assertRaises(TypeError):
            extract_features(object(),['a'])

    def test_duplicate_csv_columns_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data.csv';p.write_text('a,a\n1,2\n')
            with self.assertRaisesRegex(ValueError,'unique column'):
                list(feature_records(p))

    def test_stream_benign_suppression_authenticated_delivery_and_ack(self):
        class Engine:
            def predict(self, rows):
                return [dict(label=r['label'],confidence=.8,malicious=r['label']!='BenignTraffic') for r in rows]
        a,b=Ed25519PrivateKey.generate(),Ed25519PrivateKey.generate()
        initiator=AuthenticatedHandshake('agent',a,'server',b.public_key())
        responder=AuthenticatedHandshake('server',b,'agent',a.public_key())
        with tempfile.TemporaryDirectory() as tmp:
            client,server=socket.socketpair();client.settimeout(5);server.settimeout(5)
            errors=[];received=[]
            def listen():
                try:
                    with server:received.append(handle_connection(server,responder,tmp))
                except BaseException as exc:errors.append(exc)
            worker=threading.Thread(target=listen);worker.start()
            with client:
                result=run_stream(Engine(),iter([{'label':'BenignTraffic'},{'label':'XSS'}]),client,initiator,batch_size=1)
            worker.join(6)
            self.assertFalse(worker.is_alive());self.assertFalse(errors)
            self.assertEqual(result['records_processed'],2)
            self.assertEqual(result['alerts_sent'],1);self.assertTrue(result['receiver_acknowledged'])
            alerts=[json.loads(l) for l in (Path(tmp)/'alerts.jsonl').read_text().splitlines()]
            self.assertEqual([a['label'] for a in alerts],['XSS'])
            self.assertEqual(received[0]['alerts_received'],1)

if __name__=='__main__':unittest.main()
