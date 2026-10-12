"""Launch real server and agent CLIs on loopback with temporary demo identities."""
import argparse
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='results/deployment_tcp_demo')
    a=p.parse_args();out=Path(a.output).resolve()
    if out.exists():raise FileExistsError('Use a new output directory')
    out.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix='qarmor-deploy-keys-') as folder:
        subprocess.run([sys.executable,'-m','tools.provision_demo_identities','--directory',folder],check=True,capture_output=True)
        def environment(role,peer):
            return dict(os.environ,QARMOR_IDENTITY=role,QARMOR_PEER_IDENTITY=peer,
                QARMOR_SIGNING_KEY=str(Path(folder)/(role+'.private.pem')),
                QARMOR_TRUSTED_PEER_KEY=str(Path(folder)/(peer+'.public.pem')))
        server=subprocess.Popen([sys.executable,'-u','-m','Server.server','--port','0','--once','--output',str(out/'receiver')],
            env=environment('server','agent'),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            lines=queue.Queue()
            threading.Thread(target=lambda: lines.put(server.stdout.readline()),daemon=True).start()
            ready=lines.get(timeout=60)
            if not ready.startswith('Authenticated server listening on 127.0.0.1:'):
                raise RuntimeError('Server did not start: '+ready)
            port=ready.strip().rsplit(':',1)[1]
            agent=subprocess.run([sys.executable,'-u','-m','Agent.ids_agent','--port',port,
                '--features','data/CICIoT23_archive_sample_v1/validation/validation.csv','--limit','128'],
                env=environment('agent','server'),capture_output=True,text=True,timeout=180)
            (out/'agent.log').write_text(agent.stdout+agent.stderr)
            remaining,_=server.communicate(timeout=30)
            (out/'server.log').write_text(ready+remaining)
            if agent.returncode or server.returncode:
                raise RuntimeError('CLI demonstration failed; inspect saved logs')
            metrics=json.loads((out/'receiver/metrics.json').read_text())
            alerts=(out/'receiver/alerts.jsonl').read_text().splitlines() if (out/'receiver/alerts.jsonl').exists() else []
            assert metrics['records_processed']==128
            assert metrics['alerts_sent']==len(alerts)
            result=dict(status='passed',records=128,alerts_received=len(alerts),
                authenticated_loopback_tcp=True,server_exit=server.returncode,agent_exit=agent.returncode,
                source='Replay of existing validation feature CSV; no raw packet extraction')
            (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
        finally:
            if server.poll() is None:
                server.terminate()
                try:server.wait(timeout=5)
                except subprocess.TimeoutExpired:server.kill();server.wait()

if __name__=='__main__':main()
