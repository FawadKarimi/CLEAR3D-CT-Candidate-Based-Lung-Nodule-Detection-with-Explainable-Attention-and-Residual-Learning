"""CPU tests: resource/GPU mocks explicitly separated from disposable child checks."""
import importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path
import pytest
S=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(S))
import user_watchdog as w
import watchdog_generated_worker as worker

def limits(**kw):return {**w.GENERATED_LIMITS,**kw}
@pytest.mark.parametrize('field,kwargs,expected',[
 ('total_seconds',{'now':901},'TOTAL_TIME_LIMIT'),('setup_seconds',{'now':181},'SETUP_TIME_LIMIT'),
 ('rss_bytes',{'rss':2*w.GIB+1},'SAMPLED_RSS_LIMIT'),('gpu_bytes',{'gpu':2*w.GIB+1},'SAMPLED_OWNED_VRAM_LIMIT'),
 ('reserve_ram_bytes',{'ram':8*w.GIB-1},'HOST_RAM_HEADROOM')])
def test_budget_limits(field,kwargs,expected):
 args=dict(now=0,phase='setup',rss=0,gpu=0,ram=32*w.GIB);args.update(kwargs)
 assert w.SampleBudget(0,limits()).check(**args)==expected

def test_exact_bounds_and_unknown_gpu():
 b=w.SampleBudget(0,limits());assert b.check(180,'setup',2*w.GIB,None,8*w.GIB) is None

def test_phase_limits_cannot_be_reset():
 b=w.SampleBudget(0,limits(run_seconds=200,validation_seconds=30));b.check(0,'training',0,0,32*w.GIB);b.check(10,'validation',0,0,32*w.GIB)
 assert b.check(41,'validation',0,0,32*w.GIB)=='VALIDATION_TIME_LIMIT'
 with pytest.raises(RuntimeError,match='backward'):b.check(42,'training',0,0,32*w.GIB)
 b=w.SampleBudget(0,limits(run_seconds=20));b.check(0,'training',0,0,32*w.GIB)
 assert b.check(21,'validation',0,0,32*w.GIB)=='RUN_TIME_LIMIT'

def test_plan_does_not_import_torch_or_touch_output(tmp_path):
 code='import sys;sys.path.insert(0,'+repr(str(S))+');import user_watchdog;assert user_watchdog.main([])==0;assert "torch" not in sys.modules'
 p=subprocess.run([sys.executable,'-B','-c',code],cwd=tmp_path,capture_output=True,text=True)
 assert p.returncode==0 and 'PLAN_ONLY' in p.stdout and not list(tmp_path.iterdir())

@pytest.mark.parametrize('args,match',[
 (['--execute-generated'],'sampled-risk'),
 (['--execute-generated','--acknowledge-sampled-risk'],'Bindings'),
 (['--execute-generated','--device','cuda','--acknowledge-sampled-risk'],'shared GPU'),
 (['--execute-generated','--pilot-plan'],'Reference HOLD')])
def test_refusal_before_launch(monkeypatch,args,match):
 monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:pytest.fail('must not launch'))
 with pytest.raises(RuntimeError,match=match):w.main(args)

def test_environment_clears_overrides(monkeypatch):
 for k in ['PYTHONHOME','PYTHONPATH','LD_LIBRARY_PATH','LD_PRELOAD']:monkeypatch.setenv(k,'unsafe')
 env=w.child_env('cpu');assert not any(k in env for k in ['PYTHONHOME','PYTHONPATH','LD_LIBRARY_PATH','LD_PRELOAD'])
 assert env['CUDA_VISIBLE_DEVICES']=='' and env['CUBLAS_WORKSPACE_CONFIG']==':4096:8'

class Proc:
 pid=123
 def poll(self):return None
class Fake:
 def __init__(self,rss=0,kids=None):self.proc=Proc();self.rss=rss;self.kids=kids or [];self.cleaned=0
 def sample(self):return {'pid':123,'start_ticks':55,'rss_bytes':self.rss,'children':self.kids}
 def cleanup(self):self.cleaned+=1;return {'worker_exit':-15,'signals':['SIGTERM'],'scope':'mock only'}
@pytest.mark.parametrize('kind,reason',[
 ('rss','SAMPLED_RSS'),('gpu','SAMPLED_OWNED_VRAM'),('ram','HEADROOM'),('stop','EXTERNAL_TERMINATION'),('child','SINGLE_PROCESS'),('query','observation failed')])
def test_failure_preserves_checkpoint_and_only_owned_cleanup(tmp_path,monkeypatch,kind,reason):
 checkpoint=tmp_path/'committed-checkpoint.fixture';checkpoint.write_bytes(b'last valid bytes');before=w.digest(checkpoint)
 f=Fake(rss=3*w.GIB if kind=='rss' else 0,kids=[999] if kind=='child' else [])
 def observe():
  if kind=='query':raise RuntimeError('observation failed')
  return {'processes':[{'pid':999,'gpu_bytes':99*w.GIB},{'pid':123,'gpu_bytes':3*w.GIB if kind=='gpu' else 0}]}
 monkeypatch.setattr(os,'kill',lambda *a:pytest.fail('arbitrary PID signal forbidden'))
 result=w.supervise(f,tmp_path,'cuda',limits(),observe=observe,ram=lambda:0 if kind=='ram' else 32*w.GIB,stop=lambda:kind=='stop',sleep=lambda _:None)
 assert result['status']=='FAIL' and reason in result['reason'] and f.cleaned==1
 assert w.digest(checkpoint)==before and (tmp_path/'watchdog-result.json').exists()
 if kind=='query':assert result['peak_sampled_rss_bytes'] is None

def test_resident_vram_excluded(tmp_path):
 f=Fake();ticks=[0]
 def sleep(_):ticks[0]+=1
 result=w.supervise(f,tmp_path,'cuda',limits(),observe=lambda:{'processes':[{'pid':999,'gpu_bytes':99*w.GIB},{'pid':123,'gpu_bytes':12}]},ram=lambda:32*w.GIB,stop=lambda:ticks[0]>0,sleep=sleep)
 assert result['peak_sampled_owned_vram_bytes']==12

def test_pid_identity_mismatch_never_signals(monkeypatch):
 owned=object.__new__(w.OwnedWorker);owned.proc=Proc();owned.identity={'start_ticks':55};owned.fd=7
 monkeypatch.setattr(w,'process_identity',lambda pid:{'start_ticks':56});monkeypatch.setattr(signal,'pidfd_send_signal',lambda *a:pytest.fail('must not signal'))
 with pytest.raises(RuntimeError,match='identity'):owned.send(signal.SIGTERM)

@pytest.mark.parametrize('event',['os.fork','os.forkpty','subprocess.Popen','os.posix_spawn','os.system','os.exec'])
def test_worker_python_process_creation_rejected(event):
 with pytest.raises(RuntimeError,match='Single-process'):worker.prohibit_processes(event,())

def test_generated_cpu_worker_boundary_evidence(tmp_path):
 worker.run('cpu',tmp_path);result=w.worker_evidence(tmp_path,'cpu');assert result['actual_updates']==20 and result['skipped_updates']==0
 events=(tmp_path/'boundaries.jsonl').read_text().splitlines();(tmp_path/'boundaries.jsonl').write_text(chr(10).join(events[:-1])+chr(10))
 with pytest.raises(RuntimeError,match='Missing'):w.worker_evidence(tmp_path,'cpu')

def test_corrupt_or_zero_update_result_refused(tmp_path):
 worker.run('cpu',tmp_path);r=json.loads((tmp_path/'worker-result.json').read_text());r['actual_updates']=0;w.save(tmp_path/'worker-result.json',r)
 with pytest.raises(RuntimeError,match='Counter'):w.worker_evidence(tmp_path,'cpu')

@pytest.mark.parametrize('use_pidfd',[True,False])
def test_real_disposable_direct_child_cleanup_and_sentinel(tmp_path,monkeypatch,use_pidfd):
 # Actual CPU process test; no cgroup, GPU, or real-data access.
 if not use_pidfd:monkeypatch.delattr(os,'pidfd_open')
 script=tmp_path/'sleep_child.py';script.write_text("""import argparse,os,time
p=argparse.ArgumentParser();p.add_argument("--gate-fd",type=int);p.add_argument("--parent-pid",type=int);a=p.parse_args();assert os.read(a.gate_fd,1)==b"1";os.close(a.gate_fd);time.sleep(30)
""")
 sentinel=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(30)'])
 try:
  with (tmp_path/'child.log').open('wb') as log:
   owned=w.OwnedWorker([sys.executable,'-B',str(script)],w.child_env('cpu'),log);pid=owned.proc.pid
   assert owned.sample()['pid']==pid
   info=owned.cleanup();assert info['worker_exit'] is not None and sentinel.poll() is None
 finally:sentinel.terminate();sentinel.wait(timeout=5)

def test_binding_hash_refusal(tmp_path):
 p=tmp_path/'binding.json';p.write_text('{}')
 with pytest.raises(RuntimeError,match='hash mismatch'):w.bindings(p,'0'*64)

def test_atomic_evidence_and_no_real_worker_path(tmp_path):
 w.save(tmp_path/'result.json',{'x':1});assert w.load(tmp_path/'result.json')=={'x':1} and not (tmp_path/'result.json.tmp').exists()
 source=(S/'watchdog_generated_worker.py').read_text();assert 'lidc3d' not in source and 'DataLoader' not in source

def test_cleanup_failure_is_failure(tmp_path):
 f=Fake()
 def fail():raise RuntimeError('owned child cleanup timed out')
 f.cleanup=fail
 result=w.supervise(f,tmp_path,'cpu',limits(),stop=lambda:True)
 assert result['status']=='FAIL' and 'timed out' in result['cleanup_error']

def test_between_sample_phase_overrun_rejected(tmp_path):
 worker.run('cpu',tmp_path);r=w.load(tmp_path/'worker-result.json');r['phase_seconds']['training']=181;w.save(tmp_path/'worker-result.json',r)
 with pytest.raises(RuntimeError,match='between samples'):w.worker_evidence(tmp_path,'cpu',limits())

def test_missing_completion_after_zero_exit_is_failure(tmp_path):
 f=Fake();f.proc.poll=lambda:0
 result=w.supervise(f,tmp_path,'cpu',limits())
 assert result['status']=='FAIL' and f.cleaned==1

def test_bindings_exact_inventory_and_pin_tampering(tmp_path,monkeypatch):
 root=tmp_path/'work';(root/'scripts').mkdir(parents=True);script=root/'scripts/user_watchdog.py';script.write_text('fixture')
 monkeypatch.setattr(w,'__file__',str(script));pins=[['package'+str(i),'1'] for i in range(44)];monkeypatch.setattr(w.importlib.metadata,'version',lambda name:'1')
 b={'schema':'lidc3d-user-watchdog-v1','sources':{'scripts/user_watchdog.py':w.digest(script)},'interpreter':sys.executable,'interpreter_sha256':w.digest(sys.executable),'pins':pins};p=tmp_path/'binding.json';w.save(p,b)
 assert w.bindings(p,w.digest(p))==b
 script.write_text('changed')
 with pytest.raises(RuntimeError,match='inventory'):w.bindings(p,w.digest(p))
 script.write_text('fixture');monkeypatch.setattr(w.importlib.metadata,'version',lambda name:'2')
 with pytest.raises(RuntimeError,match='pin changed'):w.bindings(p,w.digest(p))
