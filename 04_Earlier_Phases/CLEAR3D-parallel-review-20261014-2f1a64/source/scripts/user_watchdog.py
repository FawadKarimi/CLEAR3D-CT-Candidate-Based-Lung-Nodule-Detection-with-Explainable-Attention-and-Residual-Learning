"""Unprivileged, sampled, single-worker supervision. NOT cgroup containment."""
import argparse, hashlib, importlib.metadata, json, math, os, signal, subprocess, sys, time
from pathlib import Path
import pilot_supervisor as reviewed
GIB=1024**3
PILOT_LIMITS={'setup_seconds':3600,'run_seconds':28800,'validation_seconds':7200,'total_seconds':32400,'rss_bytes':16*GIB,'allocator_bytes':8*GIB,'gpu_bytes':9*GIB,'sample_seconds':1.0,'min_ram_bytes':32*GIB,'reserve_ram_bytes':8*GIB,'min_gpu_bytes':10*GIB,'min_disk_bytes':20*GIB}
GENERATED_LIMITS={**PILOT_LIMITS,'setup_seconds':180,'run_seconds':180,'validation_seconds':180,'total_seconds':900,'rss_bytes':2*GIB,'allocator_bytes':GIB,'gpu_bytes':2*GIB}
BLOCKED_PILOT='Reference HOLD: exploratory disposition and separately reviewed bounded-data adapter required; no pilot execution in this entry point.'
def require(ok,message):
 if not ok:raise RuntimeError(message)
def digest(path):
 with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else _hash(f)
def _hash(f):
 h=hashlib.sha256()
 for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def load(path):
 require(Path(path).stat().st_size<=2*1024*1024,'Oversized JSON evidence');return json.loads(Path(path).read_text())
def utc():return reviewed.utc()
def save(path,value):
 path=Path(path);tmp=path.with_name(path.name+'.tmp')
 with tmp.open('w') as f:json.dump(value,f,sort_keys=True);f.write(chr(10));f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
 fd=os.open(path.parent,os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def append(path,value):
 with Path(path).open('a') as f:json.dump(value,f,sort_keys=True);f.write(chr(10));f.flush();os.fsync(f.fileno())
def process_identity(pid):
 text=Path('/proc',str(pid),'stat').read_text();fields=text.rsplit(')',1)[1].split()
 return {'pid':pid,'start_ticks':int(fields[19]),'rss_bytes':int(fields[21])*os.sysconf('SC_PAGE_SIZE')}
def children(pid):
 # Read child identifiers only, for every thread. Never signal these identifiers.
 result=set()
 for p in Path('/proc',str(pid),'task').glob('*/children'):
  try:result.update(int(x) for x in p.read_text().split())
  except FileNotFoundError:continue
 return sorted(result)
def available_ram():
 for line in Path('/proc/meminfo').read_text().splitlines():
  if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
 raise RuntimeError('MemAvailable unavailable')
def disk_free(path):
 s=os.statvfs(path);return s.f_bavail*s.f_frsize
class OwnedWorker:
 """Only the unreaped Popen child is a possible signal target, never GPU PIDs."""
 def __init__(self,argv,env,log):
  self.proc=None;self.fd=None;self.identity=None;rd,wr=os.pipe()
  try:
   self.proc=subprocess.Popen(argv+['--gate-fd',str(rd),'--parent-pid',str(os.getpid())],env=env,pass_fds=(rd,),stdout=log,stderr=subprocess.STDOUT,start_new_session=False)
   self.identity=process_identity(self.proc.pid)
   if hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'):
    try:self.fd=os.pidfd_open(self.proc.pid)
    except OSError as e:raise RuntimeError('pidfd unavailable unexpectedly; refusing launch') from e
   os.write(wr,b'1')
  except BaseException:
   if self.proc is not None and self.proc.poll() is None:
    # Unreaped direct child cannot be PID-reused. Gate still unreleased.
    self.proc.kill();self.proc.wait(timeout=5)
   if self.fd is not None:os.close(self.fd);self.fd=None
   raise
  finally:os.close(rd);os.close(wr)
 def sample(self):
  observed=process_identity(self.proc.pid)
  require(observed['start_ticks']==self.identity['start_ticks'],'Owned PID identity changed')
  observed['children']=children(self.proc.pid);return observed
 def send(self,sig):
  if self.proc.poll() is not None:return False
  observed=process_identity(self.proc.pid)
  require(observed['start_ticks']==self.identity['start_ticks'],'Refuse signal: PID identity changed')
  if self.fd is not None:signal.pidfd_send_signal(self.fd,sig)
  else:self.proc.send_signal(sig) # Unreaped Popen child only, no arbitrary PID API.
  return True
 def cleanup(self):
  sent=[]
  try:
   if self.proc.poll() is None:
    if self.send(signal.SIGTERM):sent.append('SIGTERM')
    try:self.proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
     if self.send(signal.SIGKILL):sent.append('SIGKILL')
     self.proc.wait(timeout=5)
   return {'worker_exit':self.proc.returncode,'signals':sent,'scope':'owned direct worker only; no descendant cleanup promise'}
  finally:
   if self.fd is not None:os.close(self.fd);self.fd=None
class SampleBudget:
 def __init__(self,start,limits):self.start=start;self.limits=limits;self.phase=None;self.phase_start=start;self.run_start=None
 def check(self,now,phase,rss,gpu,ram):
  require(phase in ['setup','training','validation','complete'],'Unknown worker phase')
  if phase!=self.phase:
   order={'setup':0,'training':1,'validation':2,'complete':3}
   require(self.phase is None or order[phase]>=order[self.phase],'Phase moved backward')
   self.phase=phase;self.phase_start=now
   if phase in ['training','validation'] and self.run_start is None:self.run_start=now
  l=self.limits
  if now-self.start>l['total_seconds']:return 'TOTAL_TIME_LIMIT'
  if phase=='setup' and now-self.start>l['setup_seconds']:return 'SETUP_TIME_LIMIT'
  if self.run_start is not None and now-self.run_start>l['run_seconds']:return 'RUN_TIME_LIMIT'
  if phase=='validation' and now-self.phase_start>l['validation_seconds']:return 'VALIDATION_TIME_LIMIT'
  if rss>l['rss_bytes']:return 'SAMPLED_RSS_LIMIT'
  if gpu is not None and gpu>l['gpu_bytes']:return 'SAMPLED_OWNED_VRAM_LIMIT'
  if ram<l['reserve_ram_bytes']:return 'HOST_RAM_HEADROOM'
  return None
def worker_evidence(out,expected_device,limits=None):
 result=load(out/'worker-result.json');require(result['status']=='PASS' and result['device']==expected_device,'Worker failed or device mismatch')
 phase=load(out/'phase.json');require(phase['name']=='complete','Missing complete phase')
 events=[]
 with (out/'boundaries.jsonl').open() as f:
  for line in f:
   require(len(events)<64,'Unbounded boundary log');events.append(json.loads(line))
 require(len(events)==20 and all(x['boundary']==i for i,x in enumerate(events)),'Missing or reordered boundaries')
 updates=sum(x['actual_update'] for x in events);skips=sum(x['skipped_update'] for x in events)
 require(updates>0 and updates+skips==20,'Zero-update or inconsistent execution')
 require(result['actual_updates']==updates and result['skipped_updates']==skips,'Counter disagreement')
 require(result['fixture_sha256']==digest(out/'generated-fixture.json'),'Generated fixture altered')
 if limits is not None:
  d=result['phase_seconds'];require(set(d)=={'setup','training','validation'} and all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in d.values()),'Invalid phase measurements')
  require(d['setup']<=limits['setup_seconds'] and d['training']+d['validation']<=limits['run_seconds'] and d['validation']<=limits['validation_seconds'],'Worker exceeded phase budget between samples')
 return result
def supervise(child,out,device,limits,observe=None,ram=available_ram,stop=lambda:False,clock=time.monotonic,sleep=time.sleep,validate_evidence=None,completion_reason="GENERATED_COMPLETE",budget_factory=None):
 start=clock();budget=(budget_factory or SampleBudget)(start,limits);result={'status':'FAIL','reason':'NOT_FINISHED','enforcement':'sampled only; no containment'};n=0;peak_rss=peak_gpu=None
 try:
  while True:
   if stop():raise RuntimeError('EXTERNAL_TERMINATION')
   code=child.proc.poll()
   if code is not None:
    require(code==0,'WORKER_EXIT_'+str(code));evidence=(validate_evidence or worker_evidence)(out,device,limits)
    now=clock();require(now-start<=limits['total_seconds'],'TOTAL_TIME_LIMIT')
    result.update(status='PASS',reason=completion_reason,worker=evidence);break
   own=child.sample();now=clock();phase=load(out/'phase.json')['name'] if (out/'phase.json').exists() else 'setup'
   gpu=observe() if observe else None
   # Recheck identity after slow observation before associating VRAM by PID.
   if gpu is not None:
    if child.proc.poll() is not None:continue
    require(child.sample()['start_ticks']==own['start_ticks'],'Identity changed during GPU query')
   vr=sum(x['gpu_bytes'] for x in gpu['processes'] if x['pid']==own['pid']) if gpu is not None else None
   available=ram();n+=1;peak_rss=max(peak_rss or 0,own['rss_bytes']);peak_gpu=max(peak_gpu or 0,vr) if vr is not None else None
   append(out/'samples.jsonl',{'utc':utc(),'monotonic':clock(),'phase':phase,'owned_worker':own,'gpu':gpu,'owned_gpu_bytes':vr,'available_ram_bytes':available})
   require(not own['children'],'SINGLE_PROCESS_VIOLATION: descendants observed; only worker will be signalled')
   reason=budget.check(clock(),phase,own['rss_bytes'],vr,available)
   if reason:raise RuntimeError(reason)
   require(sum(p.stat().st_size for p in out.iterdir() if p.is_file())<100*1024**2,'EVIDENCE_SIZE_LIMIT')
   sleep(limits['sample_seconds'])
 except BaseException as e:result.update(status='FAIL',reason=type(e).__name__+': '+str(e))
 finally:
  try:result['cleanup']=child.cleanup()
  except BaseException as e:result.update(status='FAIL',cleanup_error=str(e))
  result.update(wall_seconds=clock()-start,samples=n,peak_sampled_rss_bytes=peak_rss,peak_sampled_owned_vram_bytes=peak_gpu,utc=utc(),reference_release='HOLD')
  if observe:
   try:append(out/'gpu-after.jsonl',{'utc':utc(),'gpu':observe()})
   except BaseException as e:result.update(status='FAIL',after_observation_error=str(e))
  save(out/'watchdog-result.json',result)
 return result
def bindings(path,expected):
 require(digest(path)==expected,'Binding hash mismatch');b=load(path);root=Path(__file__).resolve().parents[1]
 require(b['schema']=='lidc3d-user-watchdog-v1','Wrong binding schema')
 actual={str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
 require(actual==b['sources'],'Source inventory changed')
 require(str(Path(sys.executable))==b['interpreter'] and digest(sys.executable)==b['interpreter_sha256'],'Interpreter changed')
 for name,version in b['pins']:require(importlib.metadata.version(name)==version,'Environment pin changed: '+name)
 require(len(b['pins'])==44,'Incomplete pin inventory');return b
def child_env(device):
 env=dict(os.environ)
 for k in ['PYTHONHOME','PYTHONPATH','LD_LIBRARY_PATH','LD_PRELOAD']:env.pop(k,None)
 env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=reviewed.GPU if device=='cuda' else '',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 return env
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--execute-generated',action='store_true');p.add_argument('--pilot-plan',action='store_true');p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--binding');p.add_argument('--binding-sha256');p.add_argument('--out');p.add_argument('--acknowledge-sampled-risk',action='store_true');p.add_argument('--acknowledge-shared-gpu-permission',action='store_true');a=p.parse_args(argv)
 if not a.execute_generated:
  print(json.dumps({'status':'PLAN_ONLY','enforcement':'sampled RSS/VRAM; no containment','single_process':True,'num_workers':0,'generated_limits':GENERATED_LIMITS,'pilot_outer_limits':PILOT_LIMITS,'pilot':BLOCKED_PILOT,'reference_release':'HOLD'}));return 0
 require(not a.pilot_plan,BLOCKED_PILOT)
 require(a.acknowledge_sampled_risk,'Explicit sampled-risk acknowledgement required')
 require(a.device!='cuda' or a.acknowledge_shared_gpu_permission,'Explicit shared GPU permission required')
 require(a.binding and a.binding_sha256 and a.out,'Bindings and fresh output required')
 b=bindings(a.binding,a.binding_sha256);out=Path(a.out).absolute();work=Path(__file__).resolve().parents[1]
 require(out.parent.resolve()==work.parent/'runs' and not out.exists(),'Output must be absent directly under this review runs directory')
 require(not any(x.is_symlink() for x in [out.parent,*out.parents]),'Symlink output parent refused')
 out.mkdir();child=None;handlers={};stopped=[False]
 def stop_handler(sig,frame):stopped[0]=True
 try:
  save(out/'invocation.json',{'argv':sys.argv if argv is None else argv,'binding_sha256':a.binding_sha256,'binding':b,'limits':GENERATED_LIMITS,'device':a.device,'utc':utc(),'reference_release':'HOLD'})
  require(available_ram()>=GENERATED_LIMITS['min_ram_bytes'],'Insufficient host RAM headroom')
  require(disk_free(out)>=GENERATED_LIMITS['min_disk_bytes'],'Insufficient disk headroom')
  observe=reviewed.gpu_snapshot if a.device=='cuda' else None
  if observe:
   before=observe();append(out/'gpu-before.jsonl',{'utc':utc(),'gpu':before});require(before['free_bytes']>=GENERATED_LIMITS['min_gpu_bytes'],'Insufficient GPU headroom')
  for sig in [signal.SIGTERM,signal.SIGINT]:handlers[sig]=signal.signal(sig,stop_handler)
  require(not stopped[0],'EXTERNAL_TERMINATION')
  with (out/'worker.log').open('xb',buffering=0) as log:
   child=OwnedWorker([sys.executable,'-B',str(work/'scripts/watchdog_generated_worker.py'),'--device',a.device,'--out',str(out)],child_env(a.device),log)
   result=supervise(child,out,a.device,GENERATED_LIMITS,observe=observe,stop=lambda:stopped[0]);os.fsync(log.fileno())
  return 0 if result['status']=='PASS' else 1
 except BaseException as e:
  cleanup=None
  if child is not None:
   try:cleanup=child.cleanup()
   except BaseException as ce:cleanup={'error':str(ce)}
  save(out/'watchdog-result.json',{'status':'FAIL','reason':str(e),'cleanup':cleanup,'reference_release':'HOLD'});return 1
 finally:
  for sig,handler in handlers.items():signal.signal(sig,handler)
if __name__=='__main__':raise SystemExit(main())
