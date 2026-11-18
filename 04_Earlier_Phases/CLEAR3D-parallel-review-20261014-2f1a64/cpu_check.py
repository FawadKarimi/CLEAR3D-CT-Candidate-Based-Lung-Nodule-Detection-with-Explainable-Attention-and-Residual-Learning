import hashlib,json,os,subprocess,sys,time,resource,importlib.metadata
from pathlib import Path
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
b=json.loads((R/'accepted/training-binding.json').read_text())
def bindings():
 assert str(Path(sys.executable))==b['interpreter']
 assert sha(Path(sys.executable))==b['interpreter_sha256']
 assert len(b['pins'])==44
 assert all(importlib.metadata.version(k)==v for k,v in b['pins'])
 assert {p.relative_to(R/'source').as_posix():sha(p) for p in (R/'source').rglob('*') if p.is_file()}==b['sources']
 return {'source_files':len(b['sources']),'pins':b['pins'],'interpreter':sys.executable,'interpreter_sha256':sha(Path(sys.executable))}
out=R/'cpu-results';out.mkdir()
result={'before':bindings(),'runs':[],'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
commands=[[sys.executable,'-B',str(R/'C-tests/test_model_path.py'),'--source',str(R/'source/src')], [sys.executable,'-B',str(R/'C-tests/test_fixed_batch.py'),'--source',str(R/'source/src'),'--runner',str(R/'D-plan/fixed_batch.py'),'--checkpoints','enabled']]
code=0
try:
 for n,cmd in enumerate(commands):
  start=time.monotonic()
  with (out/f'test-{n}.log').open('xb') as log:
   cp=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=180,check=False)
  result['runs'].append({'argv':cmd,'exit':cp.returncode,'seconds':time.monotonic()-start,'cumulative_child_peak_rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss})
  if cp.returncode:code=cp.returncode;break
 result['after']=bindings()
except BaseException as ex:result['error']=repr(ex);code=1
result['exit']=code
(out/'results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
raise SystemExit(code)
