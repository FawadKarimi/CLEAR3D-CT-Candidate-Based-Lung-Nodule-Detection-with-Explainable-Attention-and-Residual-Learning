"""Generated metadata/arrays and tiny CPU models only; no research or GPU evidence."""
import copy,hashlib,json,math,os,subprocess,sys,uuid
from pathlib import Path
import numpy as np
import pytest
import torch
from lidc3d import mixed_engineering as e,mixed_optimization as mo,optimization as op,training,checkpoints,data
from lidc3d import fp32_engineering as old,positive_engineering as parent
from lidc3d.provenance import sha256,canonical_hash
from test_positive_engineering import fixture as pool_fixture
from test_engineering_adapter import Tiny,put,ev,binding as old_binding

TRAIN_POS=[1,1,1,2,0,2,3,3,0,1,1,0]
VAL_POS=[0,1,3,1,9,1]

def fixture(tmp,monkeypatch):
 root,h,cfg=pool_fixture(tmp,36);prep=json.loads((root/'preparation.json').read_text());scans=[json.loads(l) for l in (root/'scans.jsonl').read_text().splitlines()];pool,_=parent.inspection_pool(scans);receipts=[]
 for s,n in zip(pool,TRAIN_POS+VAL_POS):
  sid=s['scan_id'];p=root/sid/'proposals.jsonl';rows=[]
  for i in range(n+65):rows.append(dict(scan_id=sid,id=f'{sid}:candidate:{i:06d}',zyx=[3,3,3],center_lps=[0,0,0] if i<n else ([2,2,2] if i==n else [6,6,6])))
  p.write_text(''.join(json.dumps(r)+chr(10) for r in rows));cp=json.loads((root/sid/'checkpoint.json').read_text());cp['files']['proposals.jsonl']=ev(p);cp['proposal_count']=len(rows);put(root/sid/'checkpoint.json',cp);prep['artifacts'][sid+'/proposals.jsonl']=ev(p);prep['artifacts'][sid+'/checkpoint.json']=ev(root/sid/'checkpoint.json')
  receipts.append(dict(scan_id=sid,patient_id=s['patient_id'],role=s['split'],scan_rank=e.rank_scan(sid),proposal_count=len(rows),proposal_sha256=sha256(p),counts={'positive':n,'negative':64,'ignored':1},planned_positive=n,planned_negative=64-n,planned_rows=64))
 contract={**e.CONTRACT,'pool':receipts,'config':e.effective_config(cfg)}
 monkeypatch.setattr(e,'CONTRACT',contract);monkeypatch.setattr(e,'RULE',{**e.RULE,'contract_sha256':canonical_hash(contract)})
 h=put(root/'preparation.json',prep);return root,h,cfg

def selected(tmp,monkeypatch):
 root,h,cfg=fixture(tmp,monkeypatch);s=e.make_selection(root,h);return root,s,s['config']

def binding(s,cfg):return {**old_binding(s,cfg),'amp':False,'parent_config_hash':canonical_hash(s['parent_config']),'engineering_policy_sha256':canonical_hash(e.RULE)}

def test_all_fixed_pool_zero_positive_streams_heaps_ordinals(tmp_path,monkeypatch):
 root,h,cfg=fixture(tmp_path,monkeypatch);seen=[];real=e.select_rows;peaks=[];push=e.heapq.heappush
 def bounded(heap,item):push(heap,item);peaks.append(len(heap));assert len(heap)<=64
 def record(path,*args,**kw):seen.append(path.parent.name);return real(path,*args,**kw)
 monkeypatch.setattr(e.heapq,'heappush',bounded);monkeypatch.setattr(e,'select_rows',record);monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('metadata accessed arrays'))
 s=e.make_selection(root,h);assert len(seen)==18 and seen==[r['scan_id'] for r in s['inspection']]
 assert len(s['entries'])==18 and all(r['selected'] for r in s['inspection']);assert sum(r['counts']['positive']==0 for r in s['inspection'])==4
 assert max(peaks)==64 and sum(len(x['selected']) for x in s['entries'])==1152
 assert len({x['scan']['patient_id'] for x in s['entries']})==18
 for en,expected in zip(s['entries'],e.CONTRACT['pool']):
  raw=(root/en['scan']['scan_id']/'proposals.jsonl').read_bytes().splitlines(keepends=True);sid=en['scan']['scan_id'];n=expected['planned_positive']
  want=sorted(range(n),key=lambda i:(e.rank_row(sid,1,i),i))+sorted(range(n+1,n+65),key=lambda i:(e.rank_row(sid,0,i),i))[:64-n]
  assert [r['ordinal'] for r in en['selected']]==want and len(set(want))==64
  for r in en['selected']:assert r['raw_sha256']==hashlib.sha256(raw[r['ordinal']]).hexdigest() and r['payload_sha256']==canonical_hash(r['row'])
 assert e.verify_selection(s)==s
 with pytest.raises(ValueError):old.verify_selection(s)
 assert old.RULE['exact_training_positive_rows']==5 and parent.RULE['minimum_selected_train_positives_total']==32


def test_scan_major_not_global_positive_order(tmp_path,monkeypatch):
 _,s,_=selected(tmp_path,monkeypatch);r,status=e.labelled_rows(s['entries'][:12]);assert len(r)==768
 for i,n in enumerate(TRAIN_POS):assert [x['label'] for x in r[i*64:(i+1)*64]]==[1]*n+[0]*(64-n)
 assert sum(x['positive'] for x in status)==15 and sum(x['negative'] for x in status)==753
 v,vs=e.labelled_rows(s['entries'][12:]);assert len(v)==384 and sum(x['label'] for x in v)==15

@pytest.mark.parametrize('damage',['hash','append','truncate','order','coords','line','mask','deadline','per_cap','total_cap'])
def test_stream_refusal_retains_failure_receipt(tmp_path,monkeypatch,damage):
 root,h,cfg=fixture(tmp_path,monkeypatch);entry=e.CONTRACT['pool'][0];p=root/entry['scan_id']/'proposals.jsonl';events=[];deadline=lambda:None
 if damage=='hash':p.write_text(p.read_text().replace('[3, 3, 3]','[4, 3, 3]',1))
 if damage=='append':p.write_bytes(p.read_bytes()+b'{}'+bytes([10]))
 if damage=='truncate':p.write_bytes(p.read_bytes()[:-60])
 if damage in ['order','coords']:
  lines=p.read_text().splitlines();r=json.loads(lines[0]);r['id']='wrong' if damage=='order' else r['id'];r['zyx']=[float('nan'),0,0] if damage=='coords' else r['zyx'];lines[0]=json.dumps(r);p.write_text(chr(10).join(lines)+chr(10))
 if damage=='line':p.write_bytes(b' '*16385)
 if damage=='mask':
  scan=json.loads((root/entry['scan_id']/'record.json').read_text());scan['ignored'][0]['mask_path']='forbidden.npy'
  with pytest.raises(ValueError):e.select_rows(p,scan,entry['proposal_sha256'],entry['proposal_count'])
  return
 if damage=='deadline':deadline=lambda:(_ for _ in ()).throw(TimeoutError())
 if damage=='per_cap':monkeypatch.setitem(e.RULE,'proposal_stream_cap_per_scan',2)
 if damage=='total_cap':monkeypatch.setitem(e.RULE,'proposal_stream_cap_total',1)
 with pytest.raises((ValueError,TimeoutError,json.JSONDecodeError)):e.make_selection(root,h,deadline,events.append)
 if damage not in ['deadline','per_cap','total_cap']:assert events[-1]['status']=='STREAM_FAILED' and events[-1]['partial_stream_acceptance'] is False

@pytest.mark.parametrize('damage',['counts','patient','insufficient'])
def test_all_verified_receipts_before_policy_refusal(tmp_path,monkeypatch,damage):
 root,h,cfg=fixture(tmp_path,monkeypatch);events=[]
 if damage=='counts':e.CONTRACT['pool'][0]['counts']['positive']+=1
 elif damage=='patient':e.CONTRACT['pool'][0]['patient_id']='alien'
 else:
  # Fully bound generated stream with too few negatives; no stream corruption.
  first=e.CONTRACT['pool'][0];p=root/first['scan_id']/'proposals.jsonl';p.write_text(''.join(p.read_text().splitlines(keepends=True)[:8]));prep=json.loads((root/'preparation.json').read_text());cp=json.loads((p.parent/'checkpoint.json').read_text());cp.update(proposal_count=8);cp['files']['proposals.jsonl']=ev(p);put(p.parent/'checkpoint.json',cp);prep['artifacts'][first['scan_id']+'/proposals.jsonl']=ev(p);prep['artifacts'][first['scan_id']+'/checkpoint.json']=ev(p.parent/'checkpoint.json');h=put(root/'preparation.json',prep);first.update(proposal_count=8,proposal_sha256=sha256(p),counts={'positive':1,'negative':6,'ignored':1})
 with pytest.raises(ValueError):e.make_selection(root,h,observe=events.append)
 if damage!='patient':assert len(events)==18 and all(x['verified_rows'] for x in events)
 else:assert not events  # Identity mismatch precedes proposal access.


def test_strict_radius_precedence_duplicates_and_ties(tmp_path,monkeypatch):
 scan={'scan_id':'s','positive':[{'center_lps':[0,0,0],'radius_mm':1,'strict_radius':True}],'ignored':[{'center_lps':[0,0,0],'radius_mm':1}]};p=tmp_path/'rows';coords=[[0,0,0],[1,0,0],[1.001,0,0],[0,0,0]]
 p.write_text(''.join(json.dumps({'scan_id':'s','id':f's:candidate:{i:06d}','zyx':[0,0,0],'center_lps':c})+chr(10) for i,c in enumerate(coords)))
 monkeypatch.setattr(e,'rank_row',lambda *a:'0'*64);picked,counts=e.select_rows(p,scan,sha256(p),4)
 assert counts=={'positive':2,'negative':1,'ignored':1} and [r['ordinal'] for r in picked]==[0,3,2]

@pytest.mark.parametrize('damage',['patient','duplicate_scan','duplicate_row','label','rank','payload','row_count','config','pool_missing'])
def test_selection_binding_drift_refused(tmp_path,monkeypatch,damage):
 _,s,cfg=selected(tmp_path,monkeypatch);entries=s['entries'];call=lambda:e.labelled_rows(entries)
 if damage=='patient':entries[0]['scan']['patient_id']='alien'
 if damage=='duplicate_scan':entries.append(copy.deepcopy(entries[0]))
 if damage=='duplicate_row':entries[0]['selected'][1]=copy.deepcopy(entries[0]['selected'][0])
 if damage=='label':entries[0]['selected'][0]['label']=0
 if damage=='rank':entries[0]['selected'][0]['rank']='0'*64
 if damage=='payload':entries[0]['selected'][0]['row']['zyx'][0]=99
 if damage=='row_count':entries[0]['selected'].pop()
 if damage=='config':cfg['training']['amp']=True;call=lambda:e.verify_config(s,cfg)
 if damage=='pool_missing':s['entries'].pop();call=lambda:e.run_pass(s,cfg,tmp_path/'refuse','cpu',binding(s,cfg),model_factory=Tiny)
 with pytest.raises(ValueError):call()


def controlled():
 torch.manual_seed(17);model=torch.nn.Linear(2,2);rows=[]
 for sid,label in mo.expected_labels():
  ordinal=sum(r['scan_id']==sid for r in rows);rows.append({'scan_id':sid,'id':f'{sid}:candidate:{ordinal:06d}','source_ordinal':ordinal,'label':label})
 x=torch.tensor([[1.+(i%7)/10,(-1.)**i*.3] for i in range(768)])
 batches=[{'image':x[i:i+2].clone(),'label':torch.tensor([r['label'] for r in rows[i:i+2]]),'index':torch.arange(i,i+2)} for i in range(0,768,2)]
 return model,batches,rows


def execute(m,batches,rows,events,**kw):
 opt=torch.optim.AdamW(m.parameters(),lr=kw.pop('lr',.001));state=op.new_update_state();args=dict(amp=False,accumulation=8,max_norm=1.,epoch=0);args.update(kw)
 value=mo.train_mixed_epoch(m,batches,opt,torch.amp.GradScaler('cuda',enabled=False),torch.tensor([1.,3.]),torch.device('cpu'),state=state,emit=events.append,ordered_rows=rows,**args)
 return value,opt,state


def test_768_single_consumption_48_updates_matches_unchanged_core():
 m,batches,rows=controlled();ref=copy.deepcopy(m);ro=torch.optim.AdamW(ref.parameters(),lr=.001);rs=op.new_update_state()
 expected=op.train_epoch(ref,batches,ro,torch.amp.GradScaler('cuda',enabled=False),torch.tensor([1.,3.]),torch.device('cpu'),False,8,1.,rs,lambda x:None,0)
 events=[];loss,opt,state=execute(m,batches,rows,events);assert loss==expected and state==rs
 assert all(torch.equal(a,b) for a,b in zip(m.parameters(),ref.parameters()))
 c=mo.coverage_summary(events);assert c['mixed_boundaries']==9 and c['negative_only_boundaries']==39 and c['denominator']==798 and c['actual_updates']==48 and c['skipped_updates']==0
 assert all(len(x['fp32_evidence']['input_dtypes'])==8 for x in events)
 assert [x['counters']['optimizer_updates'] for x in events]==list(range(1,49)) and all(x['optimizer_updates']==1 for x in events)
 assert sum(len(n['rows']) for x in events for n in x['class_microbatches'])==768
 assert all(int(opt.state[p]['step'])==48 for p in m.parameters())

@pytest.mark.parametrize('count',[5,16,19])
def test_controlled_mixed_gradient_equivalence_incomplete_groups(count,monkeypatch):
 # Deliberately exercises unchanged core with linear fixtures, not hybrid equivalence.
 torch.manual_seed(9);m=torch.nn.Linear(2,2);ref=copy.deepcopy(m);x=torch.randn(count,2);y=torch.tensor([int(i%3==0) for i in range(count)]);weights=torch.tensor([1.,3.]);batches=[{'image':x[i:i+2],'label':y[i:i+2]} for i in range(0,count,2)]
 o=torch.optim.AdamW(m.parameters(),lr=.001);ro=torch.optim.AdamW(ref.parameters(),lr=.001);expected_grads=[]
 for start in range(0,count,16):
  ro.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(ref(x[start:start+16]),y[start:start+16],weight=weights,reduction='sum')/weights[y[start:start+16]].sum();loss.backward();expected_grads.append([p.grad.clone() for p in ref.parameters()]);torch.nn.utils.clip_grad_norm_(ref.parameters(),1.,error_if_nonfinite=True);ro.step()
 got=[];clip=torch.nn.utils.clip_grad_norm_
 def capture(params,*a,**k):
  params=list(params);got.append([p.grad.clone() for p in params]);return clip(params,*a,**k)
 monkeypatch.setattr(torch.nn.utils,'clip_grad_norm_',capture);events=[];state=op.new_update_state();op.train_epoch(m,batches,o,torch.amp.GradScaler('cuda',enabled=False),weights,torch.device('cpu'),False,8,1.,state,events.append,0)
 assert len(got)==len(expected_grads)
 for ga,gb in zip(got,expected_grads):
  for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
 for a,b in zip(m.parameters(),ref.parameters()):torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
 assert events[-1]['incomplete_group']==(count%16!=0)

@pytest.mark.parametrize('damage',['amp','double_model','double_input','autocast','wrong_index','duplicate','negative_count','short','zero_lr','zero_grad','nan_loss','inf_grad','norm_overflow'])
def test_fail_fast_no_replay_or_false_update(damage,monkeypatch):
 m,batches,rows=controlled();events=[];args={}
 if damage=='amp':args['amp']=True
 if damage=='double_model':m.double()
 if damage=='double_input':batches[0]['image']=batches[0]['image'].double()
 if damage=='wrong_index':batches[0]['index']=torch.tensor([1,0])
 if damage=='duplicate':rows[1]=copy.deepcopy(rows[0])
 if damage=='negative_count':rows[0]['label']=0
 if damage=='short':batches.pop()
 if damage=='zero_lr':args['lr']=0.
 if damage=='nan_loss':batches[0]['image'].fill_(float('nan'))
 if damage in ['zero_grad','inf_grad','norm_overflow']:
  value={'zero_grad':0.,'inf_grad':float('inf'),'norm_overflow':1e30}[damage]
  for p in m.parameters():p.register_hook(lambda g:torch.full_like(g,value))
 with pytest.raises((ValueError,FloatingPointError,RuntimeError)):
  if damage=='autocast':
   with torch.autocast('cpu',dtype=torch.bfloat16):execute(m,batches,rows,events,**args)
  else:execute(m,batches,rows,events,**args)
 assert events[-1]['kind']=='mixed_diagnostic_failure' and events[-1]['rows_consumed']<=16
 bounds=[x for x in events if x['kind']=='accumulation_boundary'];assert len(bounds)<=1
 if damage=='norm_overflow':assert bounds[0]['classification']=='NONFINITE_NORM_WITH_FINITE_GRADIENTS' and not bounds[0]['unscaled_gradients']['nonfinite']
 if damage=='inf_grad':assert bounds[0]['classification']=='NONFINITE_INDIVIDUAL_GRADIENTS' and bounds[0]['optimizer_updates']==0

@pytest.mark.parametrize('damage',['counter','delta','skip','norm','parameter','step','label','precision','ordinal'])
def test_coverage_refuses_tampered_evidence(damage):
 m,batches,rows=controlled();events=[];execute(m,batches,rows,events);x=events[1]
 if damage=='counter':x['counters']['optimizer_updates']=1
 if damage=='delta':x['optimizer_updates']=2
 if damage=='skip':x['status']='SKIPPED_AMP_NONFINITE'
 if damage=='norm':x['norm_before_clip']=0
 if damage=='parameter':x['optimizer_evidence']['parameter_sha256_after']=x['optimizer_evidence']['parameter_sha256_before']
 if damage=='step':x['optimizer_evidence']['steps_after']=x['optimizer_evidence']['steps_before']
 if damage=='label':x['class_microbatches'][0]['rows'][0]['label']=1
 if damage=='precision':x['fp32_evidence']['input_dtypes'].append('float32')
 if damage=='ordinal':x['class_microbatches'][0]['rows'][0]['source_ordinal']=0
 with pytest.raises((ValueError,KeyError)):mo.coverage_summary(events)


def test_bounded_mmap_pass_validation_checkpoints(tmp_path,monkeypatch):
 _,s,cfg=selected(tmp_path,monkeypatch);out=tmp_path/'pass';b=binding(s,cfg);sizes=[];real=data.CandidateDataset.__init__
 def init(self,scans,candidates,*a,**k):
  candidates=list(candidates);sizes.append(len(candidates));assert len(candidates)<=768;return real(self,scans,candidates,*a,**k)
 monkeypatch.setattr(data.CandidateDataset,'__init__',init);r=e.run_pass(s,cfg,out,'cpu',b,model_factory=Tiny)
 assert sizes==[768]+[64]*6 and len(r['validation_weighted_loss_sums'])==384
 assert len(r['validation_class_losses']['positive'])==15 and len(r['validation_class_losses']['negative'])==369
 assert r['mixed_coverage']['actual_updates']==48 and len(r['validation_scan_status'])==6
 st=e.load_checkpoint(out,b)['state'];assert st['stage']=='TRAINED' and st['rows_consumed']==768 and st['scaler']=={}
 assert all(t.device.type=='cpu' for t in st['model'].values())
 assert isinstance(data.load_array(s['entries'][0]['scan']['image_path']),np.memmap)
 commits=[json.loads(p.read_text()) for p in (out/'engineering-checkpoints').glob('*/commit.json')];assert {x['stage'] for x in commits}=={'INITIALIZED','TRAINED'}
 assert json.loads((out/'validation-completion.json').read_text())['completed_rows']==384
 assert not (out/'TRAINING_COMPLETE').exists() and not (out/'best.pt').exists()


def test_validation_failure_keeps_six_statuses_and_trained(tmp_path,monkeypatch):
 _,s,cfg=selected(tmp_path,monkeypatch);out=tmp_path/'failed';b=binding(s,cfg);real=training.predict
 def fail(*a,**k):
  p,st=real(*a,**k);st[2].update(status='failed',error='synthetic prediction failure');return p,st
 monkeypatch.setattr(training,'predict',fail)
 with pytest.raises(ValueError,match='Incomplete validation'):e.run_pass(s,cfg,out,'cpu',b,model_factory=Tiny)
 assert len((out/'engineering_validation_scan_status.jsonl').read_text().splitlines())==6 and e.load_checkpoint(out,b)['state']['stage']=='TRAINED'
 assert (out/'engineering-failure.json').exists() and not (out/'validation-completion.json').exists() and not (out/'engineering-result.json').exists()

@pytest.mark.parametrize('stage',['engineering_intent','engineering_state','engineering_sealed'])
def test_checkpoint_crash_retains_last_generation(tmp_path,monkeypatch,stage):
 _,s,cfg=selected(tmp_path,monkeypatch);out=tmp_path/'interrupted';b=binding(s,cfg);real=e.publish_checkpoint;prior={}
 def publish(path,state,identity):
  if state['stage']=='INITIALIZED':
   ans=real(path,state,identity);prior.update({str(p.relative_to(path)):sha256(p) for p in (path/'engineering-checkpoints').rglob('*') if p.is_file()});return ans
  monkeypatch.setattr(checkpoints,'_boundary',lambda where:(_ for _ in ()).throw(RuntimeError('synthetic crash')) if where==stage else None);return real(path,state,identity)
 monkeypatch.setattr(e,'publish_checkpoint',publish)
 with pytest.raises(RuntimeError):e.run_pass(s,cfg,out,'cpu',b,model_factory=Tiny)
 assert prior and all(sha256(out/p)==h for p,h in prior.items())
 with pytest.raises(ValueError,match='Incomplete'):e.load_checkpoint(out,b)
 assert not (out/'engineering-result.json').exists() and not (out/'validation-completion.json').exists()


def test_interrupt_before_update_preserves_initialized(tmp_path,monkeypatch):
 _,s,cfg=selected(tmp_path,monkeypatch);out=tmp_path/'interrupt';b=binding(s,cfg)
 monkeypatch.setattr(op,'train_epoch',lambda *a,**k:(_ for _ in ()).throw(KeyboardInterrupt('synthetic')))
 with pytest.raises(KeyboardInterrupt):e.run_pass(s,cfg,out,'cpu',b,model_factory=Tiny)
 assert e.load_checkpoint(out,b)['state']['stage']=='INITIALIZED' and (out/'boundaries.jsonl').exists() and not (out/'engineering-result.json').exists()


def test_disposition_missing_and_old_scope_refused(tmp_path,monkeypatch,capsys):
 sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'));import mixed_engineering_watchdog as w
 monkeypatch.setattr(w.w,'OwnedWorker',lambda *a,**k:pytest.fail('worker launched'))
 assert w.main([])==0 and json.loads(capsys.readouterr().out)['status']=='PLAN_ONLY'
 with pytest.raises(RuntimeError):w.main(['--execute'])
 p=tmp_path/'decision';put(p,{'scope':'fp32_five_positive_exploratory_engineering','reference_release':'HOLD'})
 with pytest.raises(ValueError,match='scope'):e.scoped_disposition(p,sha256(p),'x','y')
 with pytest.raises(ValueError,match='hash'):e.scoped_disposition(p,'wrong','x','y')


def test_plan_only_without_torch_or_worker():
 script=Path(__file__).parents[1]/'scripts/mixed_engineering_watchdog.py';code="import sys,runpy;sys.path.insert(0,"+repr(str(script.parent))+");sys.modules['torch']=None;sys.argv=["+repr(str(script))+"];runpy.run_path("+repr(str(script))+",run_name='__main__')"
 r=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True);assert r.returncode==0,r.stderr
 assert json.loads(r.stdout)['status']=='PLAN_ONLY'
 worker=script.with_name('mixed_engineering_worker.py');r=subprocess.run([sys.executable,'-B',str(worker)],capture_output=True,text=True);assert r.returncode!=0 and '--gate-fd' in r.stderr


def test_selection_launcher_retains_all_receipts_on_refusal(tmp_path,monkeypatch):
 sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'));import mixed_engineering_watchdog as w
 root,h,cfg=fixture(tmp_path,monkeypatch);e.CONTRACT['pool'][0]['counts']['positive']+=1;monkeypatch.setattr(e,'PREPARATION',h);monkeypatch.setattr(w.w,'bindings',lambda *a:{})
 out=Path(w.__file__).resolve().parents[2]/'runs'/('synthetic-mixed-'+uuid.uuid4().hex)
 try:
  assert w.main(['--select','--binding','synthetic','--binding-sha256','synthetic','--prepared',str(root),'--out',str(out),'--acknowledge-development-metadata'])==1
  events=[json.loads(l) for l in (out/'inspection-progress.jsonl').read_text().splitlines()];assert len(events)==18 and all(x['verified_rows'] for x in events)
  assert (out/'selection-failure.json').exists() and not (out/'selection.json').exists()
 finally:
  import shutil
  if out.exists():shutil.rmtree(out)
