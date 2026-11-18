"""Generated metadata/arrays and tiny CPU models. Mock scaler is NOT CUDA evidence."""
import copy,hashlib,json,sys,shutil
from pathlib import Path
import numpy as np,pytest,torch
from lidc3d import engineering as old,positive_engineering as e,positive_optimization as po,optimization as opt,training,checkpoints
from lidc3d.provenance import sha256,canonical_hash
from test_engineering_adapter import fixture as original_fixture,Tiny,binding,put,ev

def fixture(tmp,count=27):
 root,_,cfg=original_fixture(tmp,count);prep=json.loads((root/'preparation.json').read_text());scans=[json.loads(x) for x in (root/'scans.jsonl').read_text().splitlines()];protocol=json.loads((root/'protocol.json').read_text());roles=protocol['folds'][0]['roles']
 for i in range(6,18):
  s=copy.deepcopy(scans[0]);sid=f's{i}';d=root/sid;d.mkdir();role='train' if i<14 else 'validation';split='train' if i<14 else 'val'
  s.update(scan_id=sid,patient_id=f'p{i}',series_uid=f'u{i}',pixel_fingerprint=f'finger{i}',split=split,patient_grouped_subset=0 if split=='train' else 1,image_path=str(d/'image.npy'),valid_path=str(d/'valid.npy'))
  for n in ['image.npy','valid.npy']:shutil.copyfile(root/'s0'/n,d/n)
  put(d/'record.json',s)
  with (d/'proposals.jsonl').open('w') as f:
   for j in range(count):f.write(json.dumps({'scan_id':sid,'id':f'{sid}:candidate:{j:06d}','zyx':[3,3,3],'center_lps':[[0,0,0],[2,2,2],[6,6,6]][j%3]})+chr(10))
  files={n:ev(d/n) for n in ['record.json','proposals.jsonl','image.npy','valid.npy']};put(d/'checkpoint.json',{'scan_id':sid,'preparation_key':'synthetic','files':files,'proposal_count':count});prep['artifacts'].update({sid+'/'+n:v for n,v in files.items()});prep['artifacts'][sid+'/checkpoint.json']=ev(d/'checkpoint.json');scans.append(s)
  roles[role]['series_uids'].append(s['series_uid']);roles[role]['patient_ids'].append(s['patient_id'])
 with (root/'scans.jsonl').open('w') as f:
  for s in scans:f.write(json.dumps(s)+chr(10))
 prep['files']['scans.jsonl']=sha256(root/'scans.jsonl');prep['derivation']['files']['protocol.json']=put(root/'protocol.json',protocol);h=put(root/'preparation.json',prep)
 return root,h,cfg

def selected(tmp,count=27):
 root,h,cfg=fixture(tmp,count);return root,e.make_selection(root,h),cfg

def test_policy_and_limits_distinct_original_guard(tmp_path):
 root,s,cfg=selected(tmp_path)
 assert s['schema']=='positive-engineering-selection-v1' and s['rule']!=old.RULE and e.LIMITS==old.LIMITS
 with pytest.raises(ValueError,match='rule'):old.verify_selection(s)
 assert len(s['inspection'])==18 and sum(x['selected'] for x in s['inspection'])==6
 assert e.verify_selection(s)==s

def test_fixed_pool_patient_rank_not_reference_outcome(tmp_path):
 root,h,_=fixture(tmp_path);scans=[json.loads(x) for x in (root/'scans.jsonl').read_text().splitlines()];p,skipped=e.inspection_pool(scans)
 want=[]
 for role in ['train','val']:want.extend(sorted((x for x in scans if x['split']==role),key=lambda x:(hashlib.sha256(('LIDC3D-positive-engineering-scan-v1|17|'+x['scan_id']).encode()).hexdigest(),x['scan_id'])))
 assert [x['scan_id'] for x in p]==[x['scan_id'] for x in want]
 for s in scans:s['positive']=[];s['ignored']=[]
 assert [x['scan_id'] for x in e.inspection_pool(scans)[0]]==[x['scan_id'] for x in want]

@pytest.mark.parametrize('what',['patient','fingerprint','fold','test','short_pool'])
def test_identity_and_pool_refusals(tmp_path,what):
 root,h,_=fixture(tmp_path);scans=[json.loads(x) for x in (root/'scans.jsonl').read_text().splitlines()]
 if what=='patient':scans[4]['patient_id']=scans[0]['patient_id']
 if what=='fingerprint':scans[1]['pixel_fingerprint']=scans[0]['pixel_fingerprint']
 if what=='fold':scans[0]['fold']=1
 if what=='test':scans[0]['split']='test'
 if what=='short_pool':scans.pop()
 with pytest.raises(ValueError):e.inspection_pool(scans)

def test_stream_heaps_match_bruteforce_distinct_ordinals(tmp_path,monkeypatch):
 root,h,_=fixture(tmp_path,201);peak=[0];push=e.heapq.heappush
 def bounded(heap,x):push(heap,x);peak[0]=max(peak[0],len(heap));assert len(heap)<=32
 monkeypatch.setattr(e.heapq,'heappush',bounded);monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('selection read arrays'))
 s=e.make_selection(root,h);assert sum(len(x['selected']) for x in s['entries'])==384 and peak[0]==32
 for entry in s['entries']:
  sid=entry['scan']['scan_id'];expected=[]
  for label,mod in [(1,0),(0,2)]:
   candidates=list(range(mod,201,3));key=lambda i:(hashlib.sha256(f'LIDC3D-positive-engineering-row-v1|17|{sid}|{label}|{i}'.encode()).hexdigest(),i)
   expected.extend(sorted(candidates,key=key)[:32])
  assert [x['ordinal'] for x in entry['selected']]==expected and len(set(expected))==64

@pytest.mark.parametrize('damage',['hash','append','truncate','order','coords','line','mask','deadline'])
def test_stream_corruption_refusals(tmp_path,damage):
 root,s,_=selected(tmp_path);entry=s['entries'][0];scan=entry['scan'];p=root/scan['scan_id']/'proposals.jsonl';h=entry['proposal_sha256'];deadline=lambda:None
 if damage=='hash':h='0'*64
 if damage=='append':p.write_bytes(p.read_bytes()+b'{}'+bytes([10]))
 if damage=='truncate':p.write_bytes(p.read_bytes()[:-60])
 if damage in ['order','coords']:
  lines=p.read_text().splitlines();r=json.loads(lines[0]);r['id']='wrong' if damage=='order' else r['id'];r['zyx']=[float('nan'),0,0] if damage=='coords' else r['zyx'];lines[0]=json.dumps(r);p.write_text(chr(10).join(lines)+chr(10))
 if damage=='line':p.write_bytes(b' '*16385)
 if damage=='mask':scan['ignored'][0]['mask_path']='forbidden.npy'
 if damage=='deadline':deadline=lambda:(_ for _ in ()).throw(TimeoutError())
 with pytest.raises((ValueError,TimeoutError,json.JSONDecodeError)):e.select_rows(p,scan,h,entry['proposal_count'],deadline)

def test_caps_before_stream_open(tmp_path,monkeypatch):
 root,h,_=fixture(tmp_path);monkeypatch.setattr(e,'select_rows',lambda *a,**k:pytest.fail('stream opened after cap'))
 monkeypatch.setitem(e.RULE,'proposal_stream_cap_total',100)
 with pytest.raises(ValueError,match='Aggregate'):e.make_selection(root,h)

def test_per_stream_declared_cap_before_file(tmp_path):
 with pytest.raises(ValueError,match='Per-stream'):e.select_rows(tmp_path/'absent',{'scan_id':'x'},'unused',200001)

@pytest.mark.parametrize('count',[1,3,6])
def test_insufficient_coverage_never_replaces(tmp_path,count):
 root,h,_=fixture(tmp_path,count)
 with pytest.raises(ValueError,match='COVERAGE_UNAVAILABLE'):e.make_selection(root,h)

def test_strict_radius_precedence_and_ignore(tmp_path):
 scan={'scan_id':'s','positive':[{'center_lps':[0,0,0],'radius_mm':1,'strict_radius':True}],'ignored':[{'center_lps':[0,0,0],'radius_mm':1}]};p=tmp_path/'rows'
 coords=[[0,0,0],[1,0,0],[1.001,0,0],[0,0,0]]
 p.write_text(''.join(json.dumps({'scan_id':'s','id':f's:candidate:{i:06d}','zyx':[0,0,0],'center_lps':c})+chr(10) for i,c in enumerate(coords)))
 picked,counts=e.select_rows(p,scan,sha256(p),4);assert counts=={'positive':2,'negative':1,'ignored':1};assert {r['ordinal'] for r in picked if r['label']==1}=={0,3}
 scan['positive'][0]['strict_radius']=False;assert e.select_rows(p,scan,sha256(p),4)[1]['positive']==3

def test_order_ignored_exclusion_and_tampering(tmp_path):
 _,s,_=selected(tmp_path);train=[x for x in s['entries'] if x['scan']['split']=='train'];ordered,status=e.labelled_rows(train)
 assert [r['label'] for r in ordered]==[1]*36+[0]*36 and len({r['id'] for r in ordered})==72
 assert all(x['positive']==x['negative']==9 for x in status)
 train[0]['selected'][0]['label']=0
 with pytest.raises(ValueError,match='label'):e.labelled_rows(train)


def test_cpu_positive_update_incomplete_group_checkpoint(tmp_path):
 _,s,cfg=selected(tmp_path);b=binding(s,cfg);out=tmp_path/'run';r=e.run_pass(s,cfg,out,'cpu',b,model_factory=Tiny)
 assert r['positive_coverage']['status']=='PASS_POSITIVE_ONLY_UPDATE' and r['validation_class_losses']['positive']
 events=[json.loads(x) for x in (out/'boundaries.jsonl').read_text().splitlines()];assert events[-1]['incomplete_group']
 assert sum(x['class_counts']['positive'] for x in events)==36 and sum(x['class_counts']['negative'] for x in events)==36
 proof=next(x for x in events if x['positive_only_verified']);assert po.qualifies(proof)
 assert e.load_checkpoint(out,b)['state']['rows_consumed']==72
 for mutate in ['hash','step','norm','label']:
  v=copy.deepcopy(proof)
  if mutate=='hash':v['optimizer_evidence']['parameter_sha256_after']=v['optimizer_evidence']['parameter_sha256_before']
  if mutate=='step':v['optimizer_evidence']['steps_after']=v['optimizer_evidence']['steps_before']
  if mutate=='norm':v['norm_before_clip']=0
  if mutate=='label':v['class_microbatches'][0]['rows'][0]['label']=0
  assert not po.qualifies(v)


def test_validation_failure_keeps_trained_checkpoint(tmp_path,monkeypatch):
 _,s,cfg=selected(tmp_path);b=binding(s,cfg);out=tmp_path/'run'
 def fail(model,scans,*a):return [],[{'scan_id':x['scan_id'],'status':'failed','reason':'synthetic','candidates':18,'predictions':0} for x in scans]
 monkeypatch.setattr(training,'predict',fail)
 with pytest.raises(ValueError,match='Incomplete validation'):e.run_pass(s,cfg,out,'cpu',b,model_factory=Tiny)
 assert e.load_checkpoint(out,b)['state']['stage']=='TRAINED' and not (out/'engineering-result.json').exists()
 assert len((out/'engineering_validation_scan_status.jsonl').read_text().splitlines())==2

@pytest.mark.parametrize('stage',['engineering_intent','engineering_state','engineering_sealed'])
def test_checkpoint_crash_preserves_previous_and_fails(tmp_path,monkeypatch,stage):
 _,s,cfg=selected(tmp_path);b=binding(s,cfg);out=tmp_path/'run';original=e.publish_checkpoint;calls=[0]
 def publish(*args):
  calls[0]+=1
  if calls[0]==2:
   with monkeypatch.context() as m:
    m.setattr(checkpoints,'_boundary',lambda n:(_ for _ in ()).throw(RuntimeError('synthetic crash')) if n==stage else None);return original(*args)
  return original(*args)
 monkeypatch.setattr(e,'publish_checkpoint',publish)
 with pytest.raises(RuntimeError,match='synthetic'):e.run_pass(s,cfg,out,'cpu',b,model_factory=Tiny)
 with pytest.raises(ValueError,match='Incomplete'):e.load_checkpoint(out,b)
 assert not (out/'engineering-result.json').exists()
 sealed=list((out/'engineering-checkpoints').glob('*/commit.json'));assert sealed
 for p in sealed:assert sha256(p.with_name('state.pt'))==json.loads(p.read_text())['state_sha256']

def test_checkpoint_tamper_refuses_before_tensor_load(tmp_path,monkeypatch):
 _,s,cfg=selected(tmp_path);b=binding(s,cfg);out=tmp_path/'run';e.run_pass(s,cfg,out,'cpu',b,model_factory=Tiny);ptr=json.loads((out/'engineering-commit.json').read_text());(out/ptr['generation']/'state.pt').write_bytes(b'corrupt')
 monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('corrupt tensor loaded'))
 with pytest.raises(ValueError,match='Hash changed'):e.load_checkpoint(out,b)

def test_default_plan_and_old_disposition_refusal(tmp_path,monkeypatch,capsys):
 sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'));import positive_engineering_watchdog as ew
 monkeypatch.setattr(ew.w,'OwnedWorker',lambda *a,**k:pytest.fail('worker launched'));assert ew.main([])==0;assert json.loads(capsys.readouterr().out)['status']=='PLAN_ONLY'
 with pytest.raises(RuntimeError):ew.main(['--execute'])
 p=tmp_path/'d';put(p,{'scope':'six_scan_exploratory_engineering','reference_release':'HOLD'})
 with pytest.raises(ValueError,match='scope'):e.scoped_disposition(p,sha256(p),'x','y')

def batches(labels,zero=False):
 rows=[{'id':str(i),'scan_id':'synthetic','label':label} for i,label in enumerate(labels)];data=[]
 for i in range(0,len(labels),2):data.append({'image':torch.zeros((len(labels[i:i+2]),1)) if zero else torch.tensor([[0.2+(j%3)/10] for j in range(i,min(i+2,len(labels)))]),'label':torch.tensor(labels[i:i+2]),'index':torch.arange(i,min(i+2,len(labels)))})
 return rows,data

def test_observer_preserves_exact_cpu_objective_and_updates():
 torch.manual_seed(17);a=torch.nn.Linear(1,2);b=copy.deepcopy(a);rows,data=batches([1]*32+[0]*9);oa=torch.optim.AdamW(a.parameters(),lr=.001);ob=torch.optim.AdamW(b.parameters(),lr=.001);sa=opt.new_update_state();sb=opt.new_update_state();events=[];weights=torch.tensor([1.,3.]);dev=torch.device('cpu')
 la=opt.train_epoch(a,data,oa,torch.amp.GradScaler('cuda',enabled=False),weights,dev,False,8,1.,sa,lambda e:None,0)
 lb=po.train_positive_epoch(b,data,ob,torch.amp.GradScaler('cuda',enabled=False),weights,dev,False,8,1.,sb,events.append,0,ordered_rows=rows)
 assert la==lb and sa==sb and all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))
 assert events[-1]['incomplete_group'] and sum(m['samples'] for e in events for m in e['microbatches'])==41
 assert po.coverage_summary(events)['status']=='PASS_POSITIVE_ONLY_UPDATE'

class SyntheticSkipScaler:
 """CPU-only dispatch mock; injects nonfinite gradients, not CUDA evidence."""
 def __init__(self):self.value=64.;self.calls=0;self.bad=False
 def is_enabled(self):return True
 def get_scale(self):return self.value
 def scale(self,loss):return loss*self.value
 def unscale_(self,optimizer):
  self.calls+=1;self.bad=self.calls==1
  for group in optimizer.param_groups:
   for p in group['params']:
    if p.grad is not None:p.grad.div_(self.value)
  if self.bad:optimizer.param_groups[0]['params'][0].grad.fill_(float('inf'))
 def step(self,optimizer):
  if not self.bad:optimizer.step()
 def update(self):
  if self.bad:self.value/=2

@pytest.mark.parametrize('positives,success',[(32,True),(16,False)])
def test_mock_skip_never_counts_as_positive_update_or_replays(positives,success):
 rows,data=batches([1]*positives+[0]*16);model=torch.nn.Linear(1,2);optimizer=torch.optim.AdamW(model.parameters(),lr=.001);events=[];state=opt.new_update_state()
 def run():return po.train_positive_epoch(model,data,optimizer,SyntheticSkipScaler(),torch.tensor([1.,3.]),torch.device('cpu'),True,8,1.,state,events.append,0,ordered_rows=rows)
 if success:run()
 else:
  with pytest.raises(FloatingPointError,match='positive-only'):run()
 bounds=[e for e in events if e['kind']=='accumulation_boundary'];assert bounds[0]['status']=='SKIPPED_AMP_NONFINITE' and bounds[0]['optimizer_evidence'] is None and not bounds[0]['positive_only_verified']
 assert state['skipped_updates']==1 and state['scale_changes']==1
 assert [m['step'] for e in bounds for m in e['microbatches']]==list(range(len(data)))

@pytest.mark.parametrize('failure',['zero_lr','zero_norm','wrong_index','nonfinite_loss'])
def test_positive_evidence_refusals(failure):
 rows,data=batches([1]*32,zero=failure=='zero_norm');model=torch.nn.Linear(1,2,bias=failure!='zero_norm');optimizer=torch.optim.AdamW(model.parameters(),lr=0. if failure=='zero_lr' else .001);events=[]
 if failure=='wrong_index':data[0]['index']=torch.tensor([1,0])
 if failure=='nonfinite_loss':data[0]['image'].fill_(float('nan'))
 with pytest.raises((FloatingPointError,ValueError)):po.train_positive_epoch(model,data,optimizer,torch.amp.GradScaler('cuda',enabled=False),torch.tensor([1.,3.]),torch.device('cpu'),False,8,1.,opt.new_update_state(),events.append,0,ordered_rows=rows)
 assert events[-1]['kind']=='positive_coverage_failure'

def test_selection_failure_retains_progress_without_manifest(tmp_path,monkeypatch):
 sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'));import positive_engineering_watchdog as ew
 monkeypatch.setattr(ew.w,'bindings',lambda *a:{})
 root=Path(ew.__file__).resolve().parents[2];out=root/'runs'/'synthetic-selection-refusal';out.parent.mkdir(exist_ok=True)
 def fail(*a,**k):k['observe']({'scan_id':'synthetic','complete_stream':True});raise ValueError('synthetic corrupt next stream')
 monkeypatch.setattr(ew.e,'make_selection',fail)
 try:
  assert ew.main(['--select','--binding','synthetic','--binding-sha256','synthetic','--out',str(out),'--prepared','synthetic','--acknowledge-development-metadata'])==1
  assert (out/'selection-failure.json').exists() and (out/'inspection-progress.jsonl').exists() and not (out/'selection.json').exists()
 finally:
  import shutil;shutil.rmtree(out)

def test_rank_ties_use_ordinals_without_deduplication(tmp_path,monkeypatch):
 root,h,_=fixture(tmp_path,201);scan=json.loads((root/'s0/record.json').read_text());p=root/'s0/proposals.jsonl'
 monkeypatch.setattr(e,'rank_row',lambda *a:'0'*64)
 picked,c=e.select_rows(p,scan,sha256(p),201)
 assert [r['ordinal'] for r in picked if r['label']==1]==list(range(0,96,3))
 assert [r['ordinal'] for r in picked if r['label']==0]==list(range(2,98,3))

def test_scan_rank_ties_use_scan_identity(tmp_path,monkeypatch):
 root,h,_=fixture(tmp_path);scans=[json.loads(x) for x in (root/'scans.jsonl').read_text().splitlines()];monkeypatch.setattr(e,'rank_scan',lambda x:'0'*64)
 pool,_=e.inspection_pool(scans)
 assert [x['scan_id'] for x in pool if x['split']=='train']==sorted(x['scan_id'] for x in scans if x['split']=='train')

def test_zero_norm_coverage_failure_preserves_initialized_checkpoint(tmp_path):
 _,s,cfg=selected(tmp_path);b=binding(s,cfg);out=tmp_path/'run'
 class Zero(Tiny):
  def forward(self,x):return super().forward(x)*0
 with pytest.raises(FloatingPointError,match='positive-only'):e.run_pass(s,cfg,out,'cpu',b,model_factory=Zero)
 assert e.load_checkpoint(out,b)['state']['stage']=='INITIALIZED' and not (out/'engineering-result.json').exists()
 assert (out/'engineering-failure.json').exists() and (out/'boundaries.jsonl').exists()

def test_mock_skip_bound_exhaustion_is_failure():
 class AlwaysSkip(SyntheticSkipScaler):
  def unscale_(self,optimizer):self.calls=0;super().unscale_(optimizer)
 rows,data=batches([1]*160);m=torch.nn.Linear(1,2);o=torch.optim.AdamW(m.parameters(),lr=.001);events=[];state=opt.new_update_state()
 with pytest.raises(FloatingPointError,match='exhausted'):po.train_positive_epoch(m,data,o,AlwaysSkip(),torch.tensor([1.,3.]),torch.device('cpu'),True,8,1.,state,events.append,0,ordered_rows=rows)
 assert state['skipped_updates']==8 and state['optimizer_updates']==0 and events[-1]['kind']=='positive_coverage_failure'

def test_complete_all_fixed_streams_even_after_enough_eligible(tmp_path,monkeypatch):
 root,h,_=fixture(tmp_path);seen=[];real=e.select_rows
 def observe(path,*a,**k):seen.append(path.parent.name);return real(path,*a,**k)
 monkeypatch.setattr(e,'select_rows',observe);selection=e.make_selection(root,h)
 assert len(seen)==18 and seen==[x['scan_id'] for x in selection['inspection']]

def test_default_has_no_torch_import_in_fresh_process(tmp_path):
 import subprocess,os
 script=Path(__file__).parents[1]/'scripts/positive_engineering_watchdog.py'
 code="import sys,runpy; sys.path.insert(0,"+repr(str(script.parent))+"); sys.modules['torch']=None; sys.argv=["+repr(str(script))+"]; runpy.run_path("+repr(str(script))+",run_name='__main__')"
 r=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
 assert r.returncode==0,r.stderr
 assert json.loads(r.stdout)['status']=='PLAN_ONLY'
