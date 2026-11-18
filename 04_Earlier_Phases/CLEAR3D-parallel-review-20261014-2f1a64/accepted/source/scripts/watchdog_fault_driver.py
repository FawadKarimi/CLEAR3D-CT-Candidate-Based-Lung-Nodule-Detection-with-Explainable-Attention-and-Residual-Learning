"""Review-only by default. Serial generated CPU faults against exact reviewed watchdog."""
import argparse,ctypes,hashlib,json,os,resource,signal,sys,time
from pathlib import Path
import user_watchdog as w
from watchdog_generated_worker import prohibit_processes
MIB=1024**2
CASES=('rss','timeout','child_creation','pid_identity','sigint','sigterm','watchdog_death')
CASE_SECONDS=15;SUITE_SECONDS=180;BUFFER_BYTES=32*MIB;RSS_MARGIN=8*MIB
EXPECTED={
 'rss':'SAMPLED_RSS_LIMIT; touched 32 MiB fixture; only worker cleanup',
 'timeout':'RUN_TIME_LIMIT; sleeping worker; only worker cleanup',
 'child_creation':'WORKER_EXIT_42 plus audit-hook rejection, no created child',
 'pid_identity':'PID identity refusal; zero signal calls during mismatch; then owned cleanup',
 'sigint':'EXTERNAL_TERMINATION after controller SIGINT to owned watchdog; worker cleanup',
 'sigterm':'EXTERNAL_TERMINATION after controller SIGTERM to owned watchdog; worker cleanup',
 'watchdog_death':'owned watchdog SIGKILL; worker parent-death SIGKILL; no controller worker signal needed'}

def planned():
 return {'status':'PLAN_ONLY','execution_authorized':False,'reference_release':'HOLD','cases':[{'case':c,'expected':EXPECTED[c],'observed':'NOT_RUN'} for c in CASES],'limits':{'case_seconds':CASE_SECONDS,'suite_seconds':SUITE_SECONDS,'worker_address_space_bytes':256*MIB,'buffer_bytes':BUFFER_BYTES,'sample_seconds':.1,'min_available_ram_bytes':2*1024**3},'GPU':'not queried or initialized','thresholds':'test-only; reviewed execution defaults unchanged'}
def require(ok,message):w.require(ok,message)
def libc():return ctypes.CDLL(None,use_errno=True)
def set_parent_death(parent):
 require(libc().prctl(1,signal.SIGKILL,0,0,0)==0,'PR_SET_PDEATHSIG failed')
 require(os.getppid()==parent,'Unexpected parent before gate')
def internal_gate(a):
 require(a.gate_fd is not None and a.parent_pid is not None,'Internal role requires inherited gate')
 set_parent_death(a.parent_pid);require(os.read(a.gate_fd,1)==b'1','Unreleased internal role');os.close(a.gate_fd)
def proc(pid):
 p=Path('/proc',str(pid),'stat')
 try:
  f=p.read_text().rsplit(')',1)[1].split();return {'pid':pid,'state':f[0],'ppid':int(f[1]),'start_ticks':int(f[19])}
 except FileNotFoundError:return None

def fixture(out):
 p=out/'committed-checkpoint.fixture';p.write_bytes(b'GENERATED CHECKPOINT COMMIT V1'+bytes(range(256)))
 manifest={'kind':'synthetic committed bytes, not a Torch checkpoint','file':p.name,'sha256':w.digest(p)};w.save(out/'committed-manifest.json',manifest)
 return {p.name:w.digest(p),'committed-manifest.json':w.digest(out/'committed-manifest.json')}
def preserved(out,expected):return all((out/n).is_file() and w.digest(out/n)==h for n,h in expected.items())
def test_limits(case,baseline=0):
 require(case in CASES,'Unknown fault case');require(0<=baseline<=64*MIB,'Fixture baseline unexpectedly large')
 return {**w.GENERATED_LIMITS,'setup_seconds':4,'run_seconds':2 if case=='timeout' else 10,'validation_seconds':2,'total_seconds':12,'rss_bytes':baseline+RSS_MARGIN if case=='rss' else 96*MIB,'reserve_ram_bytes':1024**3,'sample_seconds':.1}
def expired(start,now,limit):return now-start>=limit

def worker(a):
 internal_gate(a)
 # Fixture-only virtual-address guard: prevents an accidental large allocation, not RSS containment.
 resource.setrlimit(resource.RLIMIT_AS,(256*MIB,256*MIB));resource.setrlimit(resource.RLIMIT_CPU,(5,5));signal.alarm(10)
 sys.addaudithook(prohibit_processes);out=Path(a.out)
 w.save(out/'phase.json',{'name':'training'})
 identity=w.process_identity(os.getpid());identity['parent_pid']=os.getppid();identity['fixture_case']=a.case
 w.save(out/'worker-ready.json',identity)
 until=time.monotonic()+4
 while not (out/'worker-release.json').exists():
  require(time.monotonic()<until,'Fixture release timeout');time.sleep(.01)
 if a.case=='rss':
  data=bytearray(BUFFER_BYTES)
  for i in range(0,len(data),4096):data[i]=173
  w.save(out/'buffer-touched.json',{'bytes':len(data),'stride':4096,'sample_value':data[0]})
 elif a.case=='child_creation':
  try:
   child=os.fork()
  except RuntimeError as e:
   w.save(out/'child-rejection.json',{'event':'os.fork','error':str(e),'rejected':True});return 42
  # Unexpected audit failure: short-lived generated child only, no detach, report failure.
  if child==0:os._exit(99)
  os.waitpid(child,0);w.save(out/'child-rejection.json',{'rejected':False,'unexpected_child_pid':child});return 43
 time.sleep(10);return 44 # natural timeout is NOT expected commissioning success

def wait_ready(path,child,seconds=3):
 start=time.monotonic()
 while not path.exists():
  require(child.proc.poll() is None,'Owned fixture exited before ready')
  require(not expired(start,time.monotonic(),seconds),'Readiness timeout');time.sleep(.01)
 return w.load(path)
def child_command(role,case,out):
 return [sys.executable,'-B',str(Path(__file__).resolve()),'--role',role,'--case',case,'--out',str(out)]
def case_process(a):
 internal_gate(a);out=Path(a.out);stopped=[False];old={};child=None;events=[]
 for sig in (signal.SIGINT,signal.SIGTERM):old[sig]=signal.signal(sig,lambda s,f:stopped.__setitem__(0,True))
 try:
  with (out/'worker.log').open('xb',buffering=0) as log:
   child=w.OwnedWorker(child_command('worker',a.case,out),w.child_env('cpu'),log)
   ready=wait_ready(out/'worker-ready.json',child);require(ready['pid']==child.proc.pid and ready['start_ticks']==child.identity['start_ticks'],'Worker identity disagreement')
   wait_ready(out/'controller-worker-bound.json',child)
   baseline=child.sample()['rss_bytes'];limits=test_limits(a.case,baseline)
   w.save(out/'test-limits.json',{'baseline_rss_bytes':baseline,'limits':limits,'execution_defaults_unchanged':True})
   if a.case=='pid_identity':
    original=child.identity.copy();child.identity={**original,'start_ticks':original['start_ticks']+1}
    original_signal=signal.pidfd_send_signal;signal_calls=[]
    def spy(*args):signal_calls.append(list(args));return original_signal(*args)
    signal.pidfd_send_signal=spy
    try:
     try:child.send(signal.SIGTERM);error=None
     except RuntimeError as e:error=str(e)
    finally:signal.pidfd_send_signal=original_signal;child.identity=original
    w.save(out/'pid-refusal.json',{'error':error,'signal_calls_during_mismatch':signal_calls,'injected_identity':True,'actual_PID_reuse_test':False})
    require(error and 'identity changed' in error and not signal_calls,'Identity mismatch did not fail safely')
    cleanup=child.cleanup();w.save(out/'case-process-result.json',{'case':a.case,'cleanup':cleanup,'status':'EXPECTED_REFUSAL','worker_identity':original});return 0
   w.save(out/'worker-release.json',{'release':True,'case':a.case})
   if a.case=='rss':wait_ready(out/'buffer-touched.json',child)
   w.save(out/'monitor-ready.json',{'watchdog_pid':os.getpid(),'worker':ready})
   result=w.supervise(child,out,'cpu',limits,observe=None,stop=lambda:stopped[0])
   w.save(out/'case-process-result.json',{'case':a.case,'watchdog_result':result,'status':'OBSERVED','worker_identity':ready});os.fsync(log.fileno());return 0
 except BaseException as e:
  cleanup=None
  if child is not None:
   try:cleanup=child.cleanup()
   except BaseException as ce:cleanup={'error':str(ce)}
  w.save(out/'case-process-failure.json',{'type':type(e).__name__,'error':str(e),'cleanup':cleanup});return 1
 finally:
  for sig,handler in old.items():signal.signal(sig,handler)

def assess(case,out,checkpoint,controller):
 result=w.load(out/'watchdog-result.json') if (out/'watchdog-result.json').exists() else {}
 rejection=w.load(out/'child-rejection.json') if (out/'child-rejection.json').exists() else {}
 refusal=w.load(out/'pid-refusal.json') if (out/'pid-refusal.json').exists() else {}
 case_result=w.load(out/'case-process-result.json') if (out/'case-process-result.json').exists() else {}
 reason=result.get('reason','');checks={'controller_error_absent':controller.get('error') is None,'checkpoint_preserved':preserved(out,checkpoint),'no_surviving_worker':controller['worker_absent'],'no_controller_rescue':not controller['worker_rescue_signals'],'case_deadline_not_exceeded':not controller['deadline_exceeded'],'no_case_failure':not (out/'case-process-failure.json').exists()}
 if case=='watchdog_death':checks.update(watchdog_killed=controller['case_exit']==-signal.SIGKILL,worker_parent_death_signal=controller.get('adopted_worker_signal')==signal.SIGKILL,incomplete_watchdog_evidence=not (out/'watchdog-result.json').exists())
 elif case=='pid_identity':checks.update(case_exit_zero=controller['case_exit']==0,refusal=bool(refusal.get('error')) and 'identity changed' in refusal['error'],no_mismatch_signal=refusal.get('signal_calls_during_mismatch')==[],cleanup_completed=case_result.get('cleanup',{}).get('worker_exit') is not None)
 else:
  wanted={'rss':'SAMPLED_RSS_LIMIT','timeout':'RUN_TIME_LIMIT','child_creation':'WORKER_EXIT_42','sigint':'EXTERNAL_TERMINATION','sigterm':'EXTERNAL_TERMINATION'}[case]
  checks.update(case_exit_zero=controller['case_exit']==0,expected_fault=result.get('status')=='FAIL' and wanted in reason,cleanup_completed=result.get('cleanup',{}).get('worker_exit') is not None and not any('error' in k for k in result))
  if case=='rss':checks['small_buffer_touched']=(out/'buffer-touched.json').exists() and w.load(out/'buffer-touched.json')['bytes']==BUFFER_BYTES
  if case=='child_creation':checks['Python_child_rejected']=rejection.get('rejected') is True
  if case in ['sigint','sigterm']:checks['intended_signal']=controller['case_signals']==[case.upper()]
 samples=[json.loads(x) for x in (out/'samples.jsonl').read_text().splitlines()] if (out/'samples.jsonl').exists() else []
 checks['no_detected_descendants']=all(not x['owned_worker']['children'] for x in samples)
 return {'case':case,'expected':EXPECTED[case],'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'observed_watchdog_reason':reason or None,'controller':controller,'samples':len(samples),'observed':'ACTUAL_EXECUTION_ONLY_WHEN_DRIVER_AUTHORIZED'}

def rescue_allowed(ready,current,fd,parent):
 return bool(ready and current and fd is not None and current['pid']==ready['pid'] and current['start_ticks']==ready['start_ticks'] and current['ppid']==parent)

def controller_case(case,out):
 out.mkdir();checkpoint=fixture(out);w.save(out/'fixture-inventory.json',checkpoint)
 owned=None;worker_fd=None;ready=None;case_signals=[];rescue=[];deadline=False;adopted_signal=None;start=time.monotonic();error=None
 try:
  with (out/'case-process.log').open('xb',buffering=0) as log:
   owned=w.OwnedWorker(child_command('case',case,out),w.child_env('cpu'),log)
   # PID identity case may finish fast; read its durably written worker-ready record too.
   while owned.proc.poll() is None:
    if ready is None and (out/'worker-ready.json').exists():
     candidate=w.load(out/'worker-ready.json');now=proc(candidate['pid'])
     if now is not None and now['state']!='Z':
      require(now['start_ticks']==candidate['start_ticks'] and now['ppid']==owned.proc.pid,'Worker ancestry/identity not established')
      worker_fd=os.pidfd_open(candidate['pid']);ready=candidate
      w.save(out/'controller-worker-bound.json',{'pid':ready['pid'],'start_ticks':ready['start_ticks'],'parent_pid':owned.proc.pid,'pidfd_obtained':True})
    if case in ['sigint','sigterm','watchdog_death'] and not case_signals and ready and (out/'monitor-ready.json').exists():
     sig={'sigint':signal.SIGINT,'sigterm':signal.SIGTERM,'watchdog_death':signal.SIGKILL}[case]
     owned.send(sig);case_signals.append('SIGKILL' if case=='watchdog_death' else case.upper());w.save(out/'controller-signal.json',{'target':owned.identity,'signal':case_signals[-1],'utc':w.utc()})
    if expired(start,time.monotonic(),CASE_SECONDS):deadline=True;break
    time.sleep(.01)
   owned.cleanup();os.fsync(log.fileno())
 except BaseException as e:
  error=type(e).__name__+': '+str(e)
 finally:
  if owned is not None:
   try:owned.cleanup()
   except BaseException as e:error=(error or '')+' cleanup: '+str(e)
  if ready is None and (out/'worker-ready.json').exists():ready=w.load(out/'worker-ready.json')
  if ready:
   # Reap only the recorded fixture worker if adopted by this subreaper.
   end=time.monotonic()+2
   while time.monotonic()<end:
    try:
     pid,status=os.waitpid(ready['pid'],os.WNOHANG)
     if pid and os.WIFSIGNALED(status):adopted_signal=os.WTERMSIG(status)
    except ChildProcessError:pass
    current=proc(ready['pid'])
    if current is None or current['start_ticks']!=ready['start_ticks']:break
    time.sleep(.02)
   current=proc(ready['pid'])
   if current is not None and current['start_ticks']==ready['start_ticks']:
    # Emergency rescue is allowed ONLY for the previously verified owned worker,
    # now adopted by this controller. Its use makes the case fail, never masks it.
    if rescue_allowed(ready,current,worker_fd,os.getpid()):
     signal.pidfd_send_signal(worker_fd,signal.SIGKILL);rescue.append('SIGKILL')
     end=time.monotonic()+2
     while time.monotonic()<end:
      try:os.waitpid(ready['pid'],os.WNOHANG)
      except ChildProcessError:pass
      current=proc(ready['pid'])
      if current is None or current['start_ticks']!=ready['start_ticks']:break
      time.sleep(.02)
   current=proc(ready['pid']);absent=current is None or current['start_ticks']!=ready['start_ticks']
  else:absent=False
  if worker_fd is not None:os.close(worker_fd)
  record={'case_exit':owned.proc.returncode if owned else None,'case_signals':case_signals,'worker_rescue_signals':rescue,'worker_absent':absent,'worker_identity':ready,'adopted_worker_signal':adopted_signal,'deadline_exceeded':deadline,'wall_seconds':time.monotonic()-start,'error':error}
  w.save(out/'controller-result.json',record)
 result=assess(case,out,checkpoint,record)
 if error:result['status']='FAIL'
 w.save(out/'case-assessment.json',result);return result

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--acknowledge-owned-fault-signals',action='store_true');p.add_argument('--binding');p.add_argument('--binding-sha256');p.add_argument('--out');p.add_argument('--role',choices=['controller','case','worker'],default='controller',help=argparse.SUPPRESS);p.add_argument('--case',choices=CASES,help=argparse.SUPPRESS);p.add_argument('--gate-fd',type=int,help=argparse.SUPPRESS);p.add_argument('--parent-pid',type=int,help=argparse.SUPPRESS);a=p.parse_args(argv)
 if a.role!='controller':
  require(a.case and a.out,'Missing internal fixture identity');return worker(a) if a.role=='worker' else case_process(a)
 if not a.execute:print(json.dumps(planned(),indent=2));return 0
 require(a.acknowledge_owned_fault_signals,'Explicit owned-fault-signal acknowledgement required');require(a.binding and a.binding_sha256 and a.out,'Bindings and fresh output required')
 require(hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'),'Controller requires pidfd; no fallback')
 b=w.bindings(a.binding,a.binding_sha256);root=Path(__file__).resolve().parents[2];out=Path(a.out).absolute()
 require(out.parent.resolve()==root/'runs' and not out.exists() and not any(x.is_symlink() for x in [out,*out.parents]),'Fresh nonsymlink review runs output required')
 require(w.available_ram()>=2*1024**3,'Insufficient RAM headroom');require(w.disk_free(root)>=1024**3,'Insufficient disk headroom')
 # Driver-only subreaper: ownership/reaping of its orphan fixture, not a host setting.
 require(libc().prctl(36,1,0,0,0)==0,'PR_SET_CHILD_SUBREAPER unavailable')
 out.mkdir();suite={'status':'RUNNING','cases':{c:{'status':'NOT_RUN','expected':EXPECTED[c]} for c in CASES},'reference_release':'HOLD'}
 w.save(out/'invocation.json',{'argv':sys.argv if argv is None else argv,'binding':b,'binding_sha256':a.binding_sha256,'limits':planned()['limits'],'utc':w.utc()});w.save(out/'suite-result.json',suite)
 started=time.monotonic()
 try:
  for case in CASES:
   require(time.monotonic()-started+CASE_SECONDS+12<=SUITE_SECONDS,'Insufficient suite deadline for another case and cleanup')
   suite['cases'][case]=controller_case(case,out/case);w.save(out/'suite-result.json',suite)
   require(suite['cases'][case]['status']=='PASS','Fault case failed: '+case)
  suite['status']='PASS'
 except BaseException as e:suite.update(status='FAIL',error=type(e).__name__+': '+str(e))
 finally:suite.update(wall_seconds=time.monotonic()-started,utc=w.utc());w.save(out/'suite-result.json',suite)
 return 0 if suite['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
