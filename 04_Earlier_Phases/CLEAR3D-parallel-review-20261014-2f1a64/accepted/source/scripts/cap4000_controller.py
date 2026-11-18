"""One shell, no retry. No process signalling or automatic repair."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--execute',choices=['build','index']);p.add_argument('--binding-sha256');p.add_argument('--disposition');p.add_argument('--disposition-sha256');p.add_argument('--manifest-sha256');a=p.parse_args(argv)
 if not a.execute:print(json.dumps({'status':'PLAN_ONLY','reference_release':'HOLD','policy_status':'NOT_ADOPTED','torch_imported':'torch' in sys.modules}));return 0
 root=Path(__file__).resolve().parents[2];binding=root/'launcher-binding.json'
 if hashlib.sha256(binding.read_bytes()).hexdigest()!=a.binding_sha256:raise ValueError('Controller binding mismatch')
 b=json.loads(binding.read_text());runs=root/'runs'
 if not runs.is_dir() or any(x.is_symlink() for x in (runs,*runs.parents)):raise ValueError('Real existing runs parent required')
 out=Path(b['outputs'][a.execute]);evidence=runs/(a.execute+'-controller-001')
 if out.exists() or out.is_symlink() or evidence.exists() or evidence.is_symlink():raise ValueError('Output is not fresh')
 actual={str(x.relative_to(root/'work')):hashlib.sha256(x.read_bytes()).hexdigest() for x in (root/'work').rglob('*') if x.is_file() and '__pycache__' not in x.parts and '.pytest_cache' not in x.parts}
 if actual!=b['sources']:raise ValueError('Controller source inventory mismatch')
 if not a.disposition or not a.disposition_sha256:raise ValueError('Adoption required')
 evidence.mkdir();args=['/bin/sh',str(root/'work/scripts/cap4000_command.sh'),'--execute',a.execute,'--binding',str(binding),'--binding-sha256',a.binding_sha256,'--disposition',a.disposition,'--disposition-sha256',a.disposition_sha256,'--acknowledge-sampled-risk']
 if a.manifest_sha256:args+=['--manifest-sha256',a.manifest_sha256]
 (evidence/'invocation.json').write_text(json.dumps({'argv':args,'utc_unix':time.time(),'reference_release':'HOLD'},indent=2));env=dict(os.environ,CAP4000_EVIDENCE=str(evidence));start=time.monotonic()
 with (evidence/'stdout.log').open('xb') as stdout,(evidence/'stderr.log').open('xb') as stderr:
  child=subprocess.Popen(args,env=env,stdout=stdout,stderr=stderr);(evidence/'started.json').write_text(json.dumps({'shell_pid':child.pid,'no_retry':True}));rc=child.wait()
 worker=int((evidence/'worker.exit').read_text()) if (evidence/'worker.exit').exists() else None
 result={'shell_exit':rc,'worker_exit':worker,'elapsed_seconds':time.monotonic()-start,'status':'PASS' if rc==worker==0 else 'FAIL','shell_reaped':True,'reference_release':'HOLD','no_retry':True}
 (evidence/'exits.json').write_text(json.dumps(result,indent=2));return rc if rc else (0 if worker==0 else 1)
if __name__=='__main__':raise SystemExit(main())
