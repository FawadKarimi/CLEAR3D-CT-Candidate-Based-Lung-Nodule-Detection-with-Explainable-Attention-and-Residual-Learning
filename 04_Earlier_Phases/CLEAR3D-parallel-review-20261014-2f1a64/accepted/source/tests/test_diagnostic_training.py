"""Generated CPU fixtures; not full-hybrid CUDA instrumentation evidence."""
import copy,hashlib,json,sqlite3,subprocess,sys
from pathlib import Path
import pytest,torch
from lidc3d import diagnostic_training as d,scoped_epochs as e,scoped_epochs_contract as c,scoped_epoch_checkpoints as cp,cap4000_labels as labels
from test_scoped_epochs import Tiny,Small

def fixture(tmp_path,monkeypatch):
 p=tmp_path/'fixture.sqlite';db=sqlite3.connect(p)
 db.execute('CREATE TABLE labels(usable_index INTEGER PRIMARY KEY,label INTEGER,payload TEXT,payload_sha256 TEXT)')
 for i in range(800):
  raw=json.dumps({'id':str(i),'fixture_index':i});db.execute('INSERT INTO labels VALUES(?,?,?,?)',(i,int(i<15),raw,hashlib.sha256(raw.encode()).hexdigest()))
 db.commit();db.close();rows=labels.LabelRows(p,labels.hash_file(p));data=copy.deepcopy(d.DATA);plans=[]
 for epoch in range(3):
  sample=labels.DiskTrainingSample(rows,c.SAMPLING,17);sample.set_epoch(epoch);plan={'indices':sample.indices,'receipt':sample.receipt};plans.append(plan)
  path=tmp_path/f'draw{epoch}.json';path.write_text(json.dumps(plan));data['inputs']['draw'+str(epoch)]={'path':str(path),'sha256':labels.hash_file(path),'bytes':path.stat().st_size}
 rows.draw_plans=plans;monkeypatch.setattr(d,'DATA',data)
 cfg=c.config(10,'generated.pth','generated.json','a'*64,diagnostics=True);cfg['diagnostic_data']=data
 binding=d.identity(cfg,'b'*64,'synthetic','research','c'*64)
 scans=[{'scan_id':f'v{i}'} for i in range(6)];val=[{'id':str(i),'scan_id':f'v{i//64}','label':int(i<15)} for i in range(384)]
 def validate(*args):return ([{'id':v['id'],'scan_id':v['scan_id'],'logits':[.1,.2]} for v in val],[{'scan_id':s['scan_id'],'status':'ok','candidates':64,'predictions':64} for s in scans])
 def run(name,**kw):return e.fit(cfg,binding,tmp_path/name,'cpu',rows,scans,val,Small,kw.pop('validator',validate),Tiny,epochs=3,sampling=c.SAMPLING,sampling_factory=labels.DiskTrainingSample,**kw)
 return rows,cfg,binding,run

def test_three_epoch_integrated_diagnostics_and_resume(tmp_path,monkeypatch):
 rows,cfg,b,run=fixture(tmp_path,monkeypatch)
 try:
  result=run('full');assert result['update_state']['optimizer_updates']==180 and result['update_state']['skipped_updates']==0
  assert result['endpoint_epoch']==2 and [v['validation_rows'] for v in result['epochs']]==[384]*3
  report=d.diagnostic_evidence(tmp_path/'full');assert report['training_draws_by_class']=={0:2160,1:720}
  run('part',stop_after=0);run('resume',resume=tmp_path/'part')
  assert e.compare(tmp_path/'full',tmp_path/'resume',b)['exact']
  assert cp.load(tmp_path/'full',b)['state']['stage']=='TRAINED'
 finally:rows.close()

def test_instrumentation_preserves_optimizer_and_rng(tmp_path,monkeypatch):
 rows,cfg,b,run=fixture(tmp_path,monkeypatch)
 try:
  run('instrumented',stop_after=0)
  observed=cp.load(tmp_path/'instrumented',b)['state']
  cfg.pop('diagnostics');cfg.pop('diagnostic_data');plain=c.identity(cfg,'b'*64,'plain','research','c'*64);plain=labels.checkpoint_identity(plain,*['a'*64]*6)
  scans=[{'scan_id':f'v{i}'} for i in range(6)];val=[{'id':str(i),'scan_id':f'v{i//64}','label':int(i<15)} for i in range(384)]
  def validate(*args):return ([{'id':v['id'],'scan_id':v['scan_id'],'logits':[.1,.2]} for v in val],[{'scan_id':s['scan_id'],'status':'ok','candidates':64,'predictions':64} for s in scans])
  e.fit(cfg,plain,tmp_path/'plain','cpu',rows,scans,val,Small,validate,Tiny,epochs=3,sampling=c.SAMPLING,sampling_factory=labels.DiskTrainingSample,stop_after=0)
  expected=cp.load(tmp_path/'plain',plain)['state']
  def equal(a,b):
   if torch.is_tensor(a):return torch.equal(a,b)
   if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
   if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
   if hasattr(a,'shape'):return (a==b).all()
   return a==b
  for k in ('model','optimizer','scheduler','update_state','scaler','loader_rng','torch_rng','numpy_rng','python_rng'):assert equal(observed[k],expected[k]),k
 finally:rows.close()

@pytest.mark.parametrize('key',['indices','receipt'])
def test_accepted_draw_drift_refused(tmp_path,monkeypatch,key):
 rows,cfg,b,run=fixture(tmp_path,monkeypatch)
 try:
  if key=='indices':rows.draw_plans[0]['indices']=list(reversed(rows.draw_plans[0]['indices']))
  else:rows.draw_plans[0]['receipt']['unique_positive_sources']=16
  with pytest.raises(ValueError,match='draw plan'):labels.DiskTrainingSample(rows,c.SAMPLING,17)
 finally:rows.close()

def test_validation_failure_preserves_checkpoint(tmp_path,monkeypatch):
 rows,cfg,b,run=fixture(tmp_path,monkeypatch)
 try:
  def bad(*args):return [],[{'scan_id':f'v{i}','status':'failed','candidates':64,'predictions':0} for i in range(6)]
  with pytest.raises(ValueError):run('fail',validator=bad)
  assert cp.load(tmp_path/'fail',b)['state']['stage']=='INITIALIZED'
  assert (tmp_path/'fail/epoch-failure.json').exists()
 finally:rows.close()

def adopted():
 from lidc3d.engineering import HOLD_IDS
 x={'status':'ADOPTED','scope':d.SCOPE,'source_binding_sha256':'b','configuration_sha256':{},'diagnostic_data':d.DATA,'sampling':d.SAMPLING,'budget':d.BUDGET,'limits':d.RESEARCH_LIMITS,'reference_release':'HOLD','cap4000_policy':'NOT_ADOPTED','endpoint':'final_epoch_2_loss_diagnostics_only','unique_training_positives':15,'acknowledged_unresolved_holds':HOLD_IDS,'supplied_by':'fixture','instruction':'fixture','date':'fixture'}
 for k in ('label_guided_sampling','prior_validation_exposure','pretraining_overlap_uncertain','sampled_monitoring','not_scientific_approval','fp32_not_amp_coverage','no_positive_coverage_improvement','cuda_instrumentation_unmeasured'):x[k]=True
 return copy.deepcopy(x)
@pytest.mark.parametrize('key',['status','scope','diagnostic_data','configuration_sha256','source_binding_sha256','limits','supplied_by','cuda_instrumentation_unmeasured'])
def test_draft_or_changed_adoption_refused(key):
 x=adopted();assert d.disposition(x,'b',{})==x;x[key]=None
 with pytest.raises(ValueError):d.disposition(x,'b',{})

def test_plan_only_no_torch_or_gpu():
 p=Path(__file__).parents[1]/'scripts/diagnostic_training_watchdog.py'
 r=subprocess.run([sys.executable,'-B',str(p)],capture_output=True,text=True)
 assert r.returncode==0,r.stderr
 assert json.loads(r.stdout)['torch_imported'] is False
 sys.path.insert(0,str(p.parent));import diagnostic_training_watchdog as w
 with pytest.raises(ValueError):w.main(['--execute','--depth','18'])

def test_fixed_numerical_configuration():
 cfg=c.config(18,'fixture.pth','fixture.json','a'*64,diagnostics=True);cfg['diagnostic_data']=d.DATA
 assert d.check_config(cfg)==cfg
 for key in ('amp','clip_grad_norm','accumulation'):
  changed=copy.deepcopy(cfg);changed['training'][key]=None
  with pytest.raises(ValueError):d.check_config(changed)

def test_output_binding_rejects_relocation():
 x=adopted();x['diagnostic_data']['outputs']['18']+='_other'
 with pytest.raises(ValueError):d.disposition(x,'b',{})
