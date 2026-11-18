"""CPU-only: real disposable subprocesses; mocked cgroups/NVIDIA observations."""
import copy, importlib.util, json, os
from pathlib import Path
import signal, subprocess, sys, time
import pytest
SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import pilot_supervisor as s
import pilot_worker as w

@pytest.mark.parametrize('phase,elapsed,rss,gpu,reason',[
 ('preflight',3601,0,0,'SETUP_TIME_LIMIT'),('training',28801,0,0,'RUN_TIME_LIMIT'),
 ('epoch_validation',7201,0,0,'VALIDATION_TIME_LIMIT'),('final_validation',7201,0,0,'VALIDATION_TIME_LIMIT'),
 ('training',1,16*s.GIB+1,0,'PROCESS_TREE_RSS_LIMIT'),('training',1,0,9*s.GIB+1,'OWNED_GPU_LIMIT'),
 ('preflight',32401,0,0,'TOTAL_TIME_LIMIT')])
def test_budgets(phase,elapsed,rss,gpu,reason):
 b=s.Budget(0);b.check(0,phase,0,0,0)
 assert b.check(elapsed,phase,rss,gpu,0)==reason

def test_exact_limits_allowed_and_phase_clock_preserved():
 b=s.Budget(0);assert b.check(1,'training',16*s.GIB,9*s.GIB,0) is None
 assert b.check(7202,'epoch_validation',0,0,1)=='VALIDATION_TIME_LIMIT'

def test_clean_environment(monkeypatch,tmp_path):
 for k in ('PYTHONHOME','PYTHONPATH','LD_LIBRARY_PATH','LD_PRELOAD'):monkeypatch.setenv(k,'unreviewed')
 e=s.clean_environment(tmp_path)
 assert e['CUDA_VISIBLE_DEVICES']==s.GPU and e['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
 assert e['PYTHONPATH']==str(tmp_path/'src')
 assert all(k not in e for k in ('PYTHONHOME','LD_LIBRARY_PATH','LD_PRELOAD'))

def test_default_plan_no_torch_no_launch(tmp_path):
 code="import runpy,sys;sys.argv=['pilot_supervisor.py'];runpy.run_path(sys.argv[1] if len(sys.argv)>1 else "+repr(str(SCRIPTS/'pilot_supervisor.py'))+",run_name='plan_import');assert 'torch' not in sys.modules"
 q=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True);assert q.returncode==0,q.stderr
 q=subprocess.run([sys.executable,'-B',str(SCRIPTS/'pilot_supervisor.py')],cwd=tmp_path,capture_output=True,text=True)
 assert q.returncode==0 and json.loads(q.stdout)['status']=='PLAN_ONLY'
 assert not list(tmp_path.iterdir())

@pytest.mark.parametrize('args',[['--execute'],['--execute','--acknowledge-pilot-execution'],
 ['--execute','--acknowledge-reference-decision','--acknowledge-pilot-execution','--acknowledge-shared-gpu-permission']])
def test_absent_prerequisites_never_launch(monkeypatch,args):
 monkeypatch.setattr(s.subprocess,'Popen',lambda *a,**k:pytest.fail('unexpected child'))
 monkeypatch.setattr(s,'gpu_snapshot',lambda:pytest.fail('unexpected GPU query'))
 assert s.main(args)==1

def test_existing_output_unchanged(tmp_path,monkeypatch):
 out=tmp_path/'old';out.mkdir();(out/'sentinel').write_text('preserve')
 c=tmp_path/'contract';r=tmp_path/'request'
 c.write_text(json.dumps({'limits':s.LIMITS,'preparation':str(tmp_path/'prep/preparation.json'),'live_checkout':str(tmp_path/'live')}));r.write_text('{}')
 args=['--execute','--acknowledge-reference-decision','--acknowledge-pilot-execution','--acknowledge-shared-gpu-permission',
 '--contract',str(c),'--contract-sha256',s.sha(c),'--request',str(r),'--request-sha256',s.sha(r),'--out',str(out),'--cgroup-parent','/unused']
 assert s.main(args)==1 and [p.name for p in out.iterdir()]==['sentinel']

def fixture(tmp_path,monkeypatch):
 work=tmp_path/'work';(work/'src/lidc3d').mkdir(parents=True)
 (work/'src/lidc3d/a.py').write_text('# synthetic')
 prep=tmp_path/'prep';prep.mkdir();(prep/'preparation.json').write_text('{}')
 (prep/'release-holds.json').write_text(json.dumps({'unresolved':['one','two']}))
 bundle_dir=tmp_path/'bundle';bundle_dir.mkdir()
 cfg={'training':{'max_epochs':1},'candidates':{'threshold':None,'threshold_search':[.01,.02]}}
 frozen=copy.deepcopy(cfg);frozen['training']['max_epochs']=100;frozen['candidates']['threshold']=.01
 pilot=copy.deepcopy(frozen);pilot['training']['max_epochs']=1
 for name,data in [('resolved_config.json',frozen),('candidate_selection.json',{'threshold':.01}),
   ('reference_review.json',{'status':'approved','scans_sha256':'scan',
     'preparation_sha256':s.sha(prep/'preparation.json'),'hold_resolutions':{'one':'synthetic','two':'synthetic'},
     **{k:'SYNTHETIC TEST ONLY' for k in ('reviewer','geometry_audit','annotation_audit','source_exclusions_audit','prior_test_access')}})]:
  (bundle_dir/name).write_text(json.dumps(data))
 (bundle_dir/'scans.jsonl').write_text('synthetic');(bundle_dir/'candidates.jsonl').write_text('synthetic')
 scan=s.sha(bundle_dir/'scans.jsonl');rr=s.read(bundle_dir/'reference_review.json');rr['scans_sha256']=scan
 (bundle_dir/'reference_review.json').write_text(json.dumps(rr))
 bundle={'split_scope':'development','reference_audit_approved':True,'development_config_frozen':True,
         'scans':str(bundle_dir/'scans.jsonl'),'candidates':str(bundle_dir/'candidates.jsonl'),'resolved_config':str(bundle_dir/'resolved_config.json'),
         'files':{str(p.resolve()):s.sha(p) for p in list(bundle_dir.iterdir())+list(prep.iterdir())}}
 (bundle_dir/'bundle.json').write_text(json.dumps(bundle))
 conf=tmp_path/'config.json';conf.write_text(json.dumps(pilot))
 doc=tmp_path/'synthetic-statement.txt';doc.write_text('SYNTHETIC FIXTURE NOT A REAL SCIENTIFIC DECISION')
 item={'path':str(doc),'sha256':s.sha(doc)}
 request={'scope':'one-epoch-development-pilot','seed':17,'resume':None,**{k:item.copy() for k in ('scientific_decision','execution_authorization','exposure_supplement')},
  'bundle':{'path':str(bundle_dir/'bundle.json'),'sha256':s.sha(bundle_dir/'bundle.json')},
  'config':{'path':str(conf),'sha256':s.sha(conf),'canonical_sha256':s.canonical(pilot)}}
 contract={'exposure_supplement_sha256':s.sha(doc),'interpreter':str(Path(sys.executable).absolute()),'interpreter_sha256':s.sha(sys.executable),
 'pins':{str(i):'synthetic' for i in range(44)},'sources':{'src/lidc3d/a.py':s.sha(work/'src/lidc3d/a.py')},
 'code_identity':s.canonical({'a.py':s.sha(work/'src/lidc3d/a.py')}),'preparation':str(prep/'preparation.json'),
 'preparation_sha256':s.sha(prep/'preparation.json'),'holds_sha256':s.sha(prep/'release-holds.json'),'scans_sha256':scan,'pilot_template':cfg}
 monkeypatch.setattr(s.importlib.metadata,'version',lambda n:'synthetic')
 return request,contract,work

def test_bound_synthetic_prerequisites(tmp_path,monkeypatch):
 req,c,work=fixture(tmp_path,monkeypatch);bundle,cfg=s.prerequisites(req,c,work)
 assert cfg['training']['max_epochs']==1

@pytest.mark.parametrize('change',['decision','environment','source','config','reference','outside','cohort','trace','holds','seed','supplement'])
def test_prerequisite_tampering(tmp_path,monkeypatch,change):
 req,c,work=fixture(tmp_path,monkeypatch)
 if change=='decision':del req['scientific_decision']
 elif change=='environment':c['pins']['0']='wrong'
 elif change=='source':(work/'src/lidc3d/a.py').write_text('changed')
 elif change=='config':c['pilot_template']['training']['max_epochs']=2
 elif change=='cohort':c['scans_sha256']='wrong'
 elif change=='seed':req['seed']=42
 elif change=='supplement':c['exposure_supplement_sha256']='wrong'
 else:
  bp=Path(req['bundle']['path']);b=s.read(bp)
  if change=='outside':b['files']['/unrelated/withheld-file']='do not read'
  else:
   rp=bp.parent/('candidate_selection.json' if change=='trace' else 'reference_review.json');r=s.read(rp)
   if change=='trace':r['threshold']=.02
   elif change=='holds':r['hold_resolutions'].pop('one')
   else:r['status']='HOLD'
   rp.write_text(json.dumps(r));b['files'][str(rp.resolve())]=s.sha(rp)
  bp.write_text(json.dumps(b));req['bundle']['sha256']=s.sha(bp)
 with pytest.raises(ValueError):s.prerequisites(req,c,work)

class FakeProc:
 def __init__(self,code=None):self.code=code
 def poll(self):return self.code
class FakeCG:
 def sample(self):return {'processes':[{'pid':10,'rss_bytes':123}],'rss_bytes':123}
class FakeChild:
 def __init__(self,code=None):self.proc=FakeProc(code);self.cg=FakeCG();self.cleaned=False
 def cleanup(self):self.cleaned=True;self.proc.code=self.proc.code if self.proc.code is not None else -9

def fake_gpu():return {'processes':[{'pid':10,'gpu_bytes':5},{'pid':999,'gpu_bytes':99999999999}]}

@pytest.mark.parametrize('mode',['limit','external','nvml-failure','exit-failure','missing-complete','success'])
def test_monitor_preserves_evidence_and_never_signals_resident(tmp_path,monkeypatch,mode):
 (tmp_path/'training/checkpoints/sealed').mkdir(parents=True)
 kept=tmp_path/'training/checkpoints/sealed/commit.json';kept.write_text('unchanged coherent fixture')
 s.append(tmp_path/'training/optimization_boundaries.jsonl',{'counters':{'optimizer_updates':1,'skipped_updates':2}})
 s.save(tmp_path/'phase.json',{'name':'complete' if mode=='success' else 'preflight','started':time.monotonic()})
 if mode=='success':s.save(tmp_path/'worker-result.json',{'status':'PASS'})
 child=FakeChild(0 if mode in ('success','missing-complete') else 2 if mode=='exit-failure' else None)
 limits=dict(s.LIMITS);limits['rss_bytes']=1 if mode=='limit' else limits['rss_bytes']
 monkeypatch.setattr(s.os,'kill',lambda *a:pytest.fail('PID signalling forbidden'))
 def observe():
  if mode=='nvml-failure':raise RuntimeError('mock NVIDIA unavailable')
  return fake_gpu()
 result=s.monitor(child,tmp_path,observe=observe,limits=limits,stop=lambda:mode=='external')
 assert child.cleaned and kept.read_text()=='unchanged coherent fixture'
 assert result['status']==('PASS' if mode=='success' else 'FAIL')
 assert result['update_counters']=={'optimizer_updates':1,'skipped_updates':2}
 assert (tmp_path/'supervisor-result.json').exists()
 if mode=='success':assert result['peak_sampled_owned_gpu_bytes']==5
 if mode=='external':assert result['reason']=='EXTERNAL_TERMINATION'
 if mode=='nvml-failure':
  assert result['complete_resource_samples']==0
  assert result['peak_sampled_rss_bytes'] is None and result['peak_sampled_owned_gpu_bytes'] is None

def test_partial_boundary_tail_not_dropped_or_double_counted(tmp_path):
 p=tmp_path/'log';p.write_text('{"counters":');tail=s.Tail(p);tail.poll();assert tail.events==0
 with p.open('a') as f:f.write('{"optimizer_updates":2}}'+chr(10))
 tail.poll();tail.poll();assert tail.events==1 and tail.last['optimizer_updates']==2

def test_cgroup_refusal_and_kill_target(tmp_path):
 with pytest.raises(ValueError):s.Cgroup(tmp_path,100)
 cg=s.Cgroup.__new__(s.Cgroup);cg.path=tmp_path;cg.attached=True
 (tmp_path/'cgroup.kill').write_text('0');outside=tmp_path.parent/'outside-sentinel';outside.write_text('untouched')
 cg.kill();assert (tmp_path/'cgroup.kill').read_text()=='1' and outside.read_text()=='untouched'

class KernelFixtureGroup:
 """Mocks cgroup membership but uses real pidfds for our own spawned processes."""
 def __init__(self):self.pid=None;self.fds={}
 def attach(self,pid):self.pid=pid;self.fds[pid]=os.pidfd_open(pid)
 def capture(self):
  for p in Path('/proc').iterdir():
   if not p.name.isdigit():continue
   try:
    x=(p/'stat').read_text().rsplit(')',1)[1].split()
    if int(x[1]) in self.fds and int(p.name) not in self.fds:self.fds[int(p.name)]=os.pidfd_open(int(p.name))
   except (FileNotFoundError,ProcessLookupError):pass
 def kill(self):
  self.capture()
  for fd in self.fds.values():
   try:signal.pidfd_send_signal(fd,signal.SIGKILL)
   except ProcessLookupError:pass
 def empty(self):
  self.capture()
  for pid in self.fds:
   try:
    if Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':return False
   except FileNotFoundError:pass
  return True
 def close(self):
  for fd in self.fds.values():os.close(fd)

def test_real_owned_child_and_grandchild_cleanup(tmp_path):
 fixture=tmp_path/'child.py'
 fixture.write_text("import os,sys,time; fd=int(sys.argv[-1]); assert os.read(fd,1)==b'1'; pid=os.fork(); time.sleep(30)")
 outsider=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
 cg=KernelFixtureGroup()
 try:
  with (tmp_path/'log').open('w') as log:
   child=s.OwnedChild([sys.executable,str(fixture)],s.clean_environment(tmp_path,False),cg,log)
   deadline=time.monotonic()+3
   while len(cg.fds)<2 and time.monotonic()<deadline:cg.capture();time.sleep(.02)
   assert len(cg.fds)==2
   child.cleanup();assert cg.empty() and outsider.poll() is None
 finally:
  cg.kill();cg.close();outsider.terminate();outsider.wait(timeout=5)

def test_attach_failure_never_releases_gate(tmp_path):
 class Reject:
  def attach(self,pid):raise ValueError('no containment')
 with (tmp_path/'log').open('w') as log:
  with pytest.raises(ValueError):
   s.OwnedChild([sys.executable,'-c','import time;time.sleep(30)'],s.clean_environment(tmp_path,False),Reject(),log)

def test_phase_probe_locations_exact():
 lines=(SCRIPTS.parent/'src/lidc3d/training.py').read_text().splitlines()
 expected={'bundle_verification':'scans = verify_bundle','candidate_loading':'index = CandidateIndex',
 'labeling':'train_rows, val_rows = index.train','model_setup':'seed_all','training':'weighted_loss = train_epoch',
 'epoch_validation':'predictions, statuses = predict','epoch_metrics':'curves = DetectionCurves',
 'checkpoint_publication':'publish_set','epoch_record':'if new_best','best_checkpoint_load':'final_set = load_set',
 'final_validation':'predictions, statuses = predict','final_selection':'curves = DetectionCurves'}
 assert set(expected)==set(w.LINES.values())
 for line,phase in w.LINES.items():assert expected[phase] in lines[line-1]
 assert 'if not train_rows or not all' in lines[w.COUNTS_LINE-1]
 assert next(k for k,v in w.LINES.items() if v=='epoch_record')>next(k for k,v in w.LINES.items() if v=='checkpoint_publication')

def test_phase_measurements_cpu(tmp_path):
 m=w.Measurements(tmp_path);m.phase_to('preflight');m.phase_to('training');m.phase_to('final_validation');m.phase_to('complete')
 events=[json.loads(x) for x in (tmp_path/'phases.jsonl').read_text().splitlines()]
 assert len(events)==7 and all(x['seconds']>=0 for x in events if x['event']=='end')
 assert s.read(tmp_path/'phase.json')['run_started'] is not None

def test_update_reconciliation_incomplete_group_and_skips(tmp_path):
 p=tmp_path/'updates';counts={'training_nonignored':17};counter={'boundaries':0,'optimizer_updates':0,'skipped_updates':0,'scale_changes':0,'scaler_update_calls':0}
 for i,(status,n) in enumerate([('SKIPPED_AMP_NONFINITE',16),('UPDATED',1)]):
  counter.update(boundaries=i+1,optimizer_updates=i,skipped_updates=1,scale_changes=1,scaler_update_calls=i+1)
  s.append(p,{'kind':'accumulation_boundary','status':status,'optimizer_updates':int(status=='UPDATED'),
   'scaler_update_calls':1,'scale_before':65536 if i==0 else 32768,'scale_after':32768,'microbatches':[{'samples':n}], 'counters':dict(counter)})
 r=w.reconcile(p,counts);assert r['boundaries']==2 and r['skipped_examples']==16 and r['actual_optimizer_updates']==1
 with pytest.raises(ValueError):w.reconcile(p,{'training_nonignored':18})

def test_no_update_run_rejected(tmp_path):
 p=tmp_path/'updates'
 s.append(p,{'kind':'accumulation_boundary','status':'SKIPPED_AMP_NONFINITE','optimizer_updates':0,
  'scaler_update_calls':1,'scale_before':2,'scale_after':1,'microbatches':[{'samples':1}],
  'counters':{'boundaries':1,'optimizer_updates':0,'skipped_updates':1,'scale_changes':1,'scaler_update_calls':1}})
 with pytest.raises(ValueError):w.reconcile(p,{'training_nonignored':1})

def test_real_sigterm_delivery_is_preserved(tmp_path):
 child=FakeChild();stopped=[False]
 old=signal.signal(signal.SIGTERM,lambda signum,frame:stopped.__setitem__(0,True))
 try:
  def observe():
   if not stopped[0]:os.kill(os.getpid(),signal.SIGTERM)
   return fake_gpu()
  result=s.monitor(child,tmp_path,observe=observe,stop=lambda:stopped[0])
 finally:signal.signal(signal.SIGTERM,old)
 assert child.cleaned and result['reason']=='EXTERNAL_TERMINATION'
 assert s.read(tmp_path/'supervisor-result.json')['status']=='FAIL'

def test_recursive_cgroup_membership(tmp_path):
 cg=s.Cgroup.__new__(s.Cgroup);cg.path=tmp_path
 (tmp_path/'cgroup.procs').write_text('10')
 (tmp_path/'nested').mkdir();(tmp_path/'nested/cgroup.procs').write_text('11 12')
 assert cg.pids()==[10,11,12]

@pytest.mark.parametrize('bad',['model','driver','missing','unreadable-memory'])
def test_gpu_observation_fails_closed(monkeypatch,bad):
 class Result:
  def __init__(self,text):self.stdout=text
 def query(cmd,**kw):
  if '--query-gpu=' in cmd[1]:
   return Result('missing, NVIDIA GeForce RTX 2080 Ti, 535.230.02, 11000, 11264' if bad=='missing' else
    s.GPU+', '+('wrong' if bad=='model' else 'NVIDIA GeForce RTX 2080 Ti')+', '+('wrong' if bad=='driver' else '535.230.02')+', 11000, 11264')
  return Result(s.GPU+', 999, [N/A]')
 monkeypatch.setattr(s.subprocess,'run',query)
 with pytest.raises(ValueError):s.gpu_snapshot()

def test_shared_observations_keep_resident_memory(monkeypatch):
 class Result:
  def __init__(self,text):self.stdout=text
 def query(cmd,**kw):
  return Result(s.GPU+', NVIDIA GeForce RTX 2080 Ti, 535.230.02, 10452, 11264') if '--query-gpu=' in cmd[1] else Result(s.GPU+', 999, 308')
 monkeypatch.setattr(s.subprocess,'run',query)
 snap=s.gpu_snapshot();assert snap['processes']==[{'pid':999,'gpu_bytes':308*1024**2}]

@pytest.mark.parametrize('kind',['gpu','ram','disk'])
def test_free_resource_gates(kind):
 values={'gpu':10*s.GIB,'ram':32*s.GIB,'disk':20*s.GIB}
 s.resource_gate(values['gpu'],values['ram'],values['disk'])
 values[kind]-=1
 with pytest.raises(ValueError):s.resource_gate(values['gpu'],values['ram'],values['disk'])

def test_extra_source_file_rejected(tmp_path,monkeypatch):
 req,c,work=fixture(tmp_path,monkeypatch);(work/'unreviewed.py').write_text('')
 with pytest.raises(ValueError,match='inventory'):s.prerequisites(req,c,work)
