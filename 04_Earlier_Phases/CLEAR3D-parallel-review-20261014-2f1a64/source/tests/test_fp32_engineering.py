"""Generated fixtures and controlled CPU models only; no CUDA equivalence claim."""
import copy,json,hashlib,sys,subprocess
from pathlib import Path
import numpy as np,pytest,torch
from lidc3d import fp32_engineering as e,fp32_optimization as fo,positive_engineering as parent,positive_optimization as po,optimization as op,training
from lidc3d.provenance import sha256,canonical_hash
from test_positive_engineering import fixture as pool_fixture
from test_engineering_adapter import Tiny,put,ev,binding as old_binding

def fixture(tmp,monkeypatch):
 root,h,cfg=pool_fixture(tmp,36);prep=json.loads((root/'preparation.json').read_text());scans=[json.loads(l) for l in (root/'scans.jsonl').read_text().splitlines()];pool,_=parent.inspection_pool(scans)
 train=[s for s in pool if s['split']=='train'][:4];val=[s for s in pool if s['split']=='val'][:2];counts={s['scan_id']:1 for s in pool};counts[train[3]['scan_id']]=2;counts[val[1]['scan_id']]=3;receipts=[]
 for s in pool:
  sid=s['scan_id'];p=root/sid/'proposals.jsonl';n=counts[sid];rows=[]
  for i in range(n+33):rows.append(dict(scan_id=sid,id=f'{sid}:candidate:{i:06d}',zyx=[3,3,3],center_lps=[0,0,0] if i<n else ([2,2,2] if i==n else [6,6,6])))
  p.write_text(''.join(json.dumps(r)+chr(10) for r in rows));cp=json.loads((root/sid/'checkpoint.json').read_text());cp['files']['proposals.jsonl']=ev(p);cp['proposal_count']=len(rows);put(root/sid/'checkpoint.json',cp);prep['artifacts'][sid+'/proposals.jsonl']=ev(p);prep['artifacts'][sid+'/checkpoint.json']=ev(root/sid/'checkpoint.json')
  receipts.append(dict(scan_id=sid,patient_id=s['patient_id'],role=s['split'],scan_rank=e.rank_scan(sid),proposal_count=len(rows),proposal_sha256=sha256(p),counts={'positive':n,'negative':32,'ignored':1},eligible=True,selected=False,reason='not_selected_rank'))
 contract={'pool':receipts,'fixed':[r for r in receipts if r['scan_id'] in {s['scan_id'] for s in train+val}],'plan_zip_sha256':'synthetic'}
 monkeypatch.setattr(e,'CONTRACT',contract);monkeypatch.setattr(e,'RULE',{**e.RULE,'contract_sha256':canonical_hash(contract)})
 h=put(root/'preparation.json',prep);return root,h,cfg

def selected(tmp,monkeypatch):
 root,h,cfg=fixture(tmp,monkeypatch);s=e.make_selection(root,h);return root,s,s['config']

def binding(s,cfg):return {**old_binding(s,cfg),'amp':False,'parent_config_hash':canonical_hash(s['parent_config']),'engineering_policy_sha256':canonical_hash(e.RULE)}

def test_fixed_selection_rows_hashes_original_guard(tmp_path,monkeypatch):
 root,s,cfg=selected(tmp_path,monkeypatch);assert cfg['training']['amp'] is False and s['parent_config']['training']['amp'] is True
 assert [len(x['selected']) for x in s['entries']]==[1,1,1,2,33,35]
 assert len(s['inspection'])==18 and e.verify_selection(s)==s
 for en in s['entries']:
  raw=(root/en['scan']['scan_id']/'proposals.jsonl').read_bytes().splitlines(keepends=True)
  for r in en['selected']:assert r['raw_sha256']==hashlib.sha256(raw[r['ordinal']]).hexdigest() and r['payload_sha256']==canonical_hash(r['row'])
 with pytest.raises(ValueError):parent.verify_selection(s)
 assert parent.RULE['minimum_selected_train_positives_total']==32
 monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('metadata opened arrays'));assert e.make_selection(root,s['preparation_sha256'])==s

def test_receipts_retained_before_count_refusal(tmp_path,monkeypatch):
 root,h,_=fixture(tmp_path,monkeypatch);e.CONTRACT['pool'][0]['counts']['positive']+=1;events=[]
 with pytest.raises(ValueError,match='drift'):e.make_selection(root,h,observe=events.append)
 assert len(events)==18 and all(x['verified_rows'] and x['proposal_sha256'] for x in events)
 assert all(r['payload_sha256']==canonical_hash(r['row']) and len(r['raw_sha256'])==64 for x in events for r in x['verified_rows'])

@pytest.mark.parametrize('what',['pool','fixed_patient','positive_count','row_count','payload','rank','duplicate','label','config'])
def test_count_identity_precision_drift(tmp_path,monkeypatch,what):
 root,s,cfg=selected(tmp_path,monkeypatch)
 if what=='pool':e.CONTRACT['pool'][0]['patient_id']='alien';call=lambda:e.make_selection(root,s['preparation_sha256'])
 elif what=='fixed_patient':s['entries'][0]['scan']['patient_id']='alien';call=lambda:e.labelled_rows(s['entries'])
 elif what=='positive_count':e.CONTRACT['fixed'][0]['counts']['positive']=9;call=lambda:e.labelled_rows(s['entries'])
 elif what=='config':cfg['training']['amp']=True;call=lambda:e.verify_config(s,cfg)
 else:
  rows=s['entries'][0]['selected']
  if what=='row_count':rows.clear()
  if what=='duplicate':rows.append(copy.deepcopy(rows[0]))
  if what=='payload':rows[0]['row']['zyx'][0]=9
  if what=='rank':rows[0]['rank']='0'*64
  if what=='label':rows[0]['label']=0
  call=lambda:e.labelled_rows(s['entries'])
 with pytest.raises(ValueError):call()

class Linear(torch.nn.Module):
 def __init__(self):super().__init__();self.head=torch.nn.Linear(2,2)
 def forward(self,x):return self.head(x)

def controlled():
 torch.manual_seed(17);m=Linear();x=torch.tensor([[1.,2.],[2.,-1.],[.5,.25],[-1.,2.],[3.,.5]])
 data=[{'image':x[i:i+2].clone(),'label':torch.ones(min(2,5-i),dtype=torch.long),'index':torch.arange(i,min(i+2,5))} for i in range(0,5,2)]
 rows=[{'scan_id':'synthetic','id':f'row{i}','label':1} for i in range(5)];return m,data,rows

def execute(m,data,rows,events,**kwargs):
 opt=torch.optim.AdamW(m.parameters(),lr=.01,weight_decay=.01);state=op.new_update_state();scaler=torch.amp.GradScaler('cuda',enabled=False)
 args=dict(amp=False,accumulation=8,max_norm=1.,epoch=0);args.update(kwargs)
 loss=fo.train_fp32_epoch(m,data,opt,scaler,torch.tensor([1.,3.]),torch.device('cpu'),state=state,emit=events.append,ordered_rows=rows,**args)
 return loss,opt,state

def test_five_221_denominator_update_gradient_equivalence(monkeypatch):
 m,data,rows=controlled();reference=copy.deepcopy(m);refopt=torch.optim.AdamW(reference.parameters(),lr=.01,weight_decay=.01);x=torch.cat([b['image'] for b in data]);y=torch.ones(5,dtype=torch.long)
 loss=torch.nn.functional.cross_entropy(reference(x),y,weight=torch.tensor([1.,3.]),reduction='sum')/15;loss.backward();reference_grads=[p.grad.clone() for p in reference.parameters()];refnorm=torch.nn.utils.clip_grad_norm_(reference.parameters(),1.,error_if_nonfinite=True);refopt.step()
 captured=[];clip=torch.nn.utils.clip_grad_norm_
 def observe_clip(parameters,*args,**kwargs):
  parameters=list(parameters);captured.extend(p.grad.detach().clone() for p in parameters);return clip(parameters,*args,**kwargs)
 monkeypatch.setattr(torch.nn.utils,'clip_grad_norm_',observe_clip);events=[]
 value,opt,state=execute(m,data,rows,events);b=next(e for e in events if e['kind']=='accumulation_boundary')
 assert b['denominator']==15 and b['incomplete_group'] and [x['samples'] for x in b['microbatches']]==[2,2,1]
 assert b['class_counts']=={'positive':5,'negative':0} and b['norm_before_clip']>0 and not b['normalized_gradients']['nonfinite']
 assert state['optimizer_updates']==state['boundaries']==1 and state['skipped_updates']==0
 assert b['optimizer_evidence']['parameters_changed'] and b['optimizer_evidence']['optimizer_step_advanced']
 assert fo.coverage_summary(events)['not_amp_coverage']
 assert value==pytest.approx(float(loss.detach()),rel=1e-6,abs=1e-7);assert b['norm_before_clip']==pytest.approx(float(refnorm),rel=1e-6,abs=1e-7)
 for p,q in zip(m.parameters(),reference.parameters()):torch.testing.assert_close(p,q,rtol=1e-6,atol=1e-7)
 for p in m.parameters():assert int(opt.state[p]['step'])==1
 assert len(captured)==len(reference_grads)
 for a,b in zip(captured,reference_grads):torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)

@pytest.mark.parametrize('what',['amp','double_model','double_input','external_autocast','duplicate','negative','short','wrong_batch','zero_lr','zero_gradient','nan'])
def test_optimizer_refusals_and_evidence(what):
 m,data,rows=controlled();events=[];kwargs={}
 if what=='amp':kwargs['amp']=True
 if what=='double_model':m.double()
 if what=='double_input':data[0]['image']=data[0]['image'].double()
 if what=='duplicate':rows[1]['id']=rows[0]['id']
 if what=='negative':rows[0]['label']=0
 if what=='short':rows.pop()
 if what=='wrong_batch':data[0]['image']=data[0]['image'][:1]
 if what=='nan':data[0]['image'][0,0]=float('nan')
 if what=='zero_gradient':
  for p in m.parameters():p.register_hook(lambda g:torch.zeros_like(g))
 if what=='zero_lr':
  opt=torch.optim.AdamW(m.parameters(),lr=0.);call=lambda:fo.train_fp32_epoch(m,data,opt,torch.amp.GradScaler('cuda',enabled=False),torch.tensor([1.,3.]),torch.device('cpu'),False,8,1.,op.new_update_state(),events.append,0,ordered_rows=rows)
 else:call=lambda:execute(m,data,rows,events,**kwargs)
 with pytest.raises((ValueError,FloatingPointError,RuntimeError)):
  if what=='external_autocast':
   with torch.autocast('cpu',dtype=torch.bfloat16):call()
  else:call()
 assert events[-1]['kind']=='fp32_diagnostic_failure'

def test_run_pass_checkpoint_stages_and_validation_completion(tmp_path,monkeypatch):
 root,s,cfg=selected(tmp_path,monkeypatch);out=tmp_path/'pass';b=binding(s,cfg);result=e.run_pass(s,cfg,out,'cpu',b,model_factory=Tiny)
 assert result['positive_coverage']['actual_updates']==1 and len(result['validation_weighted_loss_sums'])==68
 restored=e.load_checkpoint(out,b);assert restored['state']['stage']=='TRAINED' and restored['state']['rows_consumed']==5
 completion=json.loads((out/'validation-completion.json').read_text());assert completion['completed_rows']==68 and completion['checkpoint_stage']=='TRAINED'
 commits=[json.loads(p.read_text()) for p in (out/'engineering-checkpoints').glob('*/commit.json')];assert len(commits)==2
 assert not (out/'TRAINING_COMPLETE').exists()

def test_validation_failure_preserves_trained_no_completion(tmp_path,monkeypatch):
 root,s,cfg=selected(tmp_path,monkeypatch);out=tmp_path/'pass';b=binding(s,cfg);real=training.predict
 def fail(*args,**kwargs):
  predictions,statuses=real(*args,**kwargs);statuses[0]['status']='failed';statuses[0]['error']='synthetic failure';return predictions,statuses
 monkeypatch.setattr(training,'predict',fail)
 with pytest.raises(Exception):e.run_pass(s,cfg,out,'cpu',b,model_factory=Tiny)
 assert e.load_checkpoint(out,b)['state']['stage']=='TRAINED'
 assert not (out/'validation-completion.json').exists() and not (out/'engineering-result.json').exists()
 assert (out/'engineering-failure.json').exists()

@pytest.mark.parametrize('stage',['intent','state','sealed'])
def test_checkpoint_interruption_preserves_initialized(tmp_path,monkeypatch,stage):
 root,s,cfg=selected(tmp_path,monkeypatch);out=tmp_path/'pass';b=binding(s,cfg);real=e.publish_checkpoint;prior={}
 def publish(path,state,identity):
  if state['stage']=='INITIALIZED':
   ans=real(path,state,identity);prior['files']={str(p.relative_to(path)):sha256(p) for p in (path/'engineering-checkpoints').rglob('*') if p.is_file()};return ans
  def crash(where):
   if where==stage:raise RuntimeError('synthetic crash '+stage)
  from lidc3d import checkpoints
  monkeypatch.setattr(checkpoints,'_boundary',lambda where:crash(where.removeprefix('engineering_')));return real(path,state,identity)
 monkeypatch.setattr(e,'publish_checkpoint',publish)
 with pytest.raises(RuntimeError):e.run_pass(s,cfg,out,'cpu',b,model_factory=Tiny)
 for rel,h in prior['files'].items():assert sha256(out/rel)==h
 assert not (out/'validation-completion.json').exists() and not (out/'engineering-result.json').exists()
 with pytest.raises(Exception):e.load_checkpoint(out,b)

def test_default_plan_no_torch_import():
 script=Path(__file__).parents[1]/'scripts/fp32_engineering_watchdog.py'
 code="import sys,runpy;sys.path.insert(0,"+repr(str(script.parent))+");sys.modules['torch']=None;sys.argv=["+repr(str(script))+"];runpy.run_path("+repr(str(script))+",run_name='__main__')"
 r=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True);assert r.returncode==0 and json.loads(r.stdout)['status']=='PLAN_ONLY'

@pytest.mark.parametrize('what',['hash','truncated','appended','per_stream_cap','aggregate_cap'])
def test_new_stream_verifier_refuses_corruption(tmp_path,monkeypatch,what):
 root,h,cfg=fixture(tmp_path,monkeypatch);first=e.CONTRACT['pool'][0];p=root/first['scan_id']/'proposals.jsonl';events=[]
 if what=='hash':p.write_text(p.read_text().replace('[3, 3, 3]','[4, 3, 3]',1))
 if what=='truncated':p.write_text(''.join(p.read_text().splitlines(keepends=True)[:-1]))
 if what=='appended':p.write_text(p.read_text()+'{}'+chr(10))
 if what in ['per_stream_cap','aggregate_cap']:
  monkeypatch.setattr(e,'RULE',{**e.RULE,'proposal_stream_cap_per_scan':2 if what=='per_stream_cap' else 200000,'proposal_stream_cap_total':1 if what=='aggregate_cap' else 1000000})
 with pytest.raises(ValueError):e.make_selection(root,h,observe=events.append)
 assert not events

def test_metadata_failure_launcher_retains_verified_rows(tmp_path,monkeypatch):
 import importlib.util,shutil,uuid
 script=Path(__file__).parents[1]/'scripts/fp32_engineering_watchdog.py';sys.path.insert(0,str(script.parent));spec=importlib.util.spec_from_file_location('fp32_launch_test',script);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 root,h,cfg=fixture(tmp_path,monkeypatch);e.CONTRACT['pool'][0]['counts']['positive']+=1;monkeypatch.setattr(e,'PREPARATION',h);monkeypatch.setattr(m.w,'bindings',lambda *x:{})
 out=script.parents[2]/'runs'/('synthetic-'+uuid.uuid4().hex)
 try:
  assert m.main(['--select','--binding','synthetic','--binding-sha256','synthetic','--prepared',str(root),'--out',str(out),'--acknowledge-development-metadata'])==1
  events=[json.loads(x) for x in (out/'inspection-progress.jsonl').read_text().splitlines()];assert len(events)==18 and all(x['verified_rows'] for x in events)
  assert (out/'selection-failure.json').exists() and not (out/'selection.json').exists()
 finally:
  if out.exists():shutil.rmtree(out)
