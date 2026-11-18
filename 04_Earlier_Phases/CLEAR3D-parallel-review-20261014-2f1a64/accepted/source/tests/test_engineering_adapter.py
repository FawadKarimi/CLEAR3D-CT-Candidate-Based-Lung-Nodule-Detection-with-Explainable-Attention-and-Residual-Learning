"""Generated arrays and tiny CPU model only. No research, GPU or real approval."""
import copy,json,hashlib,sys,types
from pathlib import Path
import numpy as np,pytest,torch,yaml
from lidc3d import engineering as e,checkpoints as cp,training
from lidc3d.provenance import canonical_hash,sha256

def put(p,v):p.write_text(json.dumps(v));return sha256(p)
def ev(p):return {'bytes':p.stat().st_size,'sha256':sha256(p)}
def fixture(tmp,count=3):
 root=tmp/'prepared';root.mkdir();cfg=yaml.safe_load((Path(__file__).parents[1]/'configs/reference.yaml').read_text());cfg['model']['patch_size']=8
 scans=[];artifacts={};roles={k:{'series_uids':[],'patient_ids':[],'subsets':[i]} for i,k in enumerate(['train','validation','test'])}
 for i in range(6):
  sid=f's{i}';d=root/sid;d.mkdir();split='train' if i<4 else 'val';role='train' if i<4 else 'validation';sub=0 if i<4 else 1
  s={'scan_id':sid,'patient_id':f'p{i}','series_uid':f'u{i}','split':split,'fold':0,'pixel_fingerprint':f'finger{i}','patient_grouped_subset':sub,'status':'ok','positive':[{'center_lps':[0,0,0],'radius_mm':1}],'ignored':[{'center_lps':[2,2,2],'radius_mm':1}],'image_path':str(d/'image.npy'),'valid_path':str(d/'valid.npy')}
  np.save(d/'image.npy',np.full((8,8,8),i/10,dtype=np.float32));np.save(d/'valid.npy',np.ones((8,8,8),bool));put(d/'record.json',s)
  with (d/'proposals.jsonl').open('w') as f:
   for j in range(count):f.write(json.dumps({'scan_id':sid,'id':f'{sid}:candidate:{j:06d}','zyx':[3,3,3],'center_lps':[[0,0,0],[2,2,2],[6,6,6]][j%3]})+chr(10))
  files={n:ev(d/n) for n in ['record.json','proposals.jsonl','image.npy','valid.npy']};put(d/'checkpoint.json',{'scan_id':sid,'preparation_key':'synthetic','files':files,'proposal_count':count})
  artifacts.update({sid+'/'+n:v for n,v in files.items()});artifacts[sid+'/checkpoint.json']=ev(d/'checkpoint.json');scans.append(s);roles[role]['series_uids'].append(s['series_uid']);roles[role]['patient_ids'].append(s['patient_id'])
 with (root/'scans.jsonl').open('w') as f:
  for s in scans:f.write(json.dumps(s)+chr(10))
 ph=put(root/'protocol.json',{'folds':[{'fold':0,'roles':roles}]});hh=put(root/'release-holds.json',{'unresolved':list(e.HOLD_IDS)})
 prep={'status':'PROPOSALS_NOT_YET_FROZEN','preparation_key':'synthetic','files':{'scans.jsonl':sha256(root/'scans.jsonl')},'derivation':{'files':{'protocol.json':ph,'release-holds.json':hh}},'artifacts':artifacts,'config':cfg};h=put(root/'preparation.json',prep)
 selection=e.make_selection(root,h);return root,selection,cfg
class Tiny(torch.nn.Module):
 def __init__(self,cfg):super().__init__();self.head=torch.nn.Linear(1,2)
 def forward(self,x):return self.head(x.mean((2,3,4)))
def binding(selection,cfg):return {'run_id':'synthetic-only','selection_sha256':canonical_hash(selection),'config_hash':canonical_hash(cfg),'seed':17,'source_binding_sha256':'synthetic-source','preparation_sha256':selection['preparation_sha256']}

def test_stream_selection_matches_bruteforce_duplicates_and_cap(tmp_path,monkeypatch):
 root,s,cfg=fixture(tmp_path,140);assert sum(len(x['selected']) for x in s['entries'])==384
 for x in s['entries']:
  sid=x['scan']['scan_id'];want=sorted(range(140),key=lambda i:(e.rank_row(sid,i),i))[:64];assert [r['ordinal'] for r in x['selected']]==want
 monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('selection opened pixels'))
 assert e.make_selection(root,s['preparation_sha256'])==s

def test_memory_is_heap_bounded_and_end_hash_checked(tmp_path):
 root,s,_=fixture(tmp_path,1000);x=s['entries'][0];p=root/x['scan']['scan_id']/'proposals.jsonl'
 with p.open('a') as f:f.write('{}'+chr(10))
 with pytest.raises(ValueError):e.select_rows(p,x['scan']['scan_id'],x['proposal_sha256'],1000)

def test_row_hash_mismatch_and_deadline(tmp_path):
 root,s,_=fixture(tmp_path);x=s['entries'][0];p=root/x['scan']['scan_id']/'proposals.jsonl'
 with pytest.raises(ValueError,match='stream'):e.select_rows(p,x['scan']['scan_id'],'0'*64,3)
 with pytest.raises(TimeoutError):e.select_rows(p,x['scan']['scan_id'],x['proposal_sha256'],3,lambda:(_ for _ in ()).throw(TimeoutError()))

@pytest.mark.parametrize('mutation',['patient','fingerprint','test','fold'])
def test_cohort_conflicts_fail_before_selection(tmp_path,mutation):
 _,s,_=fixture(tmp_path);scans=[x['scan'] for x in s['entries']]
 if mutation=='patient':scans[4]['patient_id']=scans[0]['patient_id']
 if mutation=='fingerprint':scans[1]['pixel_fingerprint']=scans[0]['pixel_fingerprint']
 if mutation=='test':scans[0]['split']='test'
 if mutation=='fold':scans[0]['fold']=1
 with pytest.raises(ValueError):e.select_scans(scans)

def test_scan_rule_independent_of_outcomes_and_no_replacement(tmp_path):
 root,s,_=fixture(tmp_path);scans=[x['scan'] for x in s['entries']];a=e.select_scans(scans)
 for x in scans:x['positive']=[];x['status']='failed'
 assert e.select_scans(scans)==a
 # Metadata already selected: a missing chosen stream stops, never substitutes.
 (root/a[0][0]['scan_id']/'proposals.jsonl').unlink()
 with pytest.raises(ValueError):e.make_selection(root,s['preparation_sha256'])

def test_ignored_rows_not_replaced_and_mmap_loading(tmp_path):
 _,s,cfg=fixture(tmp_path);rows,status=e.labelled_rows(s['entries'][:4]);assert len(rows)==8 and all(len(x['ignored_ordinals'])==1 for x in status)
 from lidc3d.data import load_array,CandidateDataset
 assert isinstance(load_array(s['entries'][0]['scan']['image_path']),np.memmap)
 ds=CandidateDataset([x['scan'] for x in s['entries'][:4]],rows,8,False);assert ds[0]['image'].shape==(1,8,8,8)

def test_one_pass_incomplete_group_counts_and_no_selection(tmp_path):
 _,s,cfg=fixture(tmp_path);out=tmp_path/'run';b=binding(s,cfg);r=e.run_pass(s,cfg,out,'cpu',b,model_factory=Tiny)
 assert r['update_state']['optimizer_updates']==1 and r['update_state']['skipped_updates']==0
 event=json.loads((out/'boundaries.jsonl').read_text());assert event['incomplete_group'] and sum(x['samples'] for x in event['microbatches'])==8
 assert len(r['validation_scan_status'])==2 and not (out/'selection.json').exists() and not (out/'best.pt').exists()
 st=e.load_checkpoint(out,b)['state'];assert st['stage']=='TRAINED' and st['scaler']=={} and st['rows_consumed']==8
 assert all(t.device.type=='cpu' for t in st['model'].values())

def test_validation_failure_preserves_trained_commit_and_cohort(tmp_path,monkeypatch):
 _,s,cfg=fixture(tmp_path);out=tmp_path/'run';b=binding(s,cfg)
 def fail(model,scans,*args):return [],[{'scan_id':x['scan_id'],'status':'failed','reason':'synthetic','candidates':3,'predictions':0} for x in scans]
 monkeypatch.setattr(training,'predict',fail)
 with pytest.raises(ValueError,match='Incomplete validation'):e.run_pass(s,cfg,out,'cpu',b,model_factory=Tiny)
 assert not (out/'engineering-result.json').exists();assert e.load_checkpoint(out,b)['state']['stage']=='TRAINED'
 assert len((out/'engineering_validation_scan_status.jsonl').read_text().splitlines())==2
 assert (out/'engineering-failure.json').exists()

def test_zero_candidate_validation_success(tmp_path):
 _,s,cfg=fixture(tmp_path)
 for x in s['entries'][4:]:x['selected']=[]
 r=e.run_pass(s,cfg,tmp_path/'run','cpu',binding(s,cfg),model_factory=Tiny)
 assert all(x['status']=='ok' and x['predictions']==0 for x in r['validation_scan_status'])

@pytest.mark.parametrize('stage',['engineering_intent','engineering_state','engineering_sealed'])
def test_interruption_preserves_prior_generation_and_refuses_incomplete(tmp_path,monkeypatch,stage):
 _,s,cfg=fixture(tmp_path);b=binding(s,cfg);out=tmp_path/'run'
 def crash(name):
  if name==stage and (out/'engineering-checkpoints').exists():raise RuntimeError('synthetic publication interruption')
 # Interrupt second publication only, retaining initial generation.
 calls=[0];real=e.publish_checkpoint
 def publish(*args):
  calls[0]+=1
  if calls[0]==2:
   with monkeypatch.context() as m:
    m.setattr(cp,'_boundary',lambda name:(_ for _ in ()).throw(RuntimeError('synthetic publication interruption')) if name==stage else None);return real(*args)
  return real(*args)
 monkeypatch.setattr(e,'publish_checkpoint',publish)
 with pytest.raises(RuntimeError):e.run_pass(s,cfg,out,'cpu',b,model_factory=Tiny)
 assert not (out/'engineering-result.json').exists()
 with pytest.raises(ValueError,match='Incomplete'):e.load_checkpoint(out,b)
 sealed=[p for p in (out/'engineering-checkpoints').iterdir() if (p/'commit.json').exists()];assert sealed
 for g in sealed:assert sha256(g/'state.pt')==json.loads((g/'commit.json').read_text())['state_sha256']

def test_checkpoint_tamper_rejected_before_load(tmp_path,monkeypatch):
 _,s,cfg=fixture(tmp_path);b=binding(s,cfg);out=tmp_path/'run';e.run_pass(s,cfg,out,'cpu',b,model_factory=Tiny)
 pointer=json.loads((out/'engineering-commit.json').read_text());(out/pointer['generation']/'state.pt').write_bytes(b'corrupt')
 monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('corrupt checkpoint loaded'))
 with pytest.raises(ValueError,match='Hash changed'):e.load_checkpoint(out,b)

def test_nonfinite_and_zero_update_fail_without_completion(tmp_path):
 _,s,cfg=fixture(tmp_path)
 class Bad(Tiny):
  def forward(self,x):return super().forward(x)*float('nan')
 with pytest.raises(FloatingPointError):e.run_pass(s,cfg,tmp_path/'bad','cpu',binding(s,cfg),model_factory=Bad)
 assert not (tmp_path/'bad/engineering-result.json').exists()

def test_disposition_is_explicit_and_bound(tmp_path):
 p=tmp_path/'decision.json';put(p,{'scope':'six_scan_exploratory_engineering','reference_release':'HOLD'})
 with pytest.raises(ValueError,match='binding'):e.scoped_disposition(p,sha256(p),'selection','source')
 with pytest.raises(ValueError,match='hash'):e.scoped_disposition(p,'wrong','selection','source')

def test_plan_only_and_missing_prerequisites_no_worker(monkeypatch,capsys):
 sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'));import engineering_watchdog as ew
 monkeypatch.setattr(ew.w,'OwnedWorker',lambda *a,**k:pytest.fail('worker launched'))
 assert ew.main([])==0;assert json.loads(capsys.readouterr().out)['status']=='PLAN_ONLY'
 with pytest.raises(RuntimeError):ew.main(['--execute'])

def test_watchdog_callback_explicit_mock_only(tmp_path):
 sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'));import user_watchdog as w
 child=types.SimpleNamespace(proc=types.SimpleNamespace(poll=lambda:0),cleanup=lambda:{'worker_exit':0,'signals':[]})
 r=w.supervise(child,tmp_path,'cpu',w.GENERATED_LIMITS,validate_evidence=lambda *a:{'synthetic':True},completion_reason='ENGINEERING_TEST')
 assert r['status']=='PASS' and r['reason']=='ENGINEERING_TEST'

def test_heap_never_exceeds_64(tmp_path,monkeypatch):
 root,s,_=fixture(tmp_path,100);x=s['entries'][0];maximum=[0];real=e.heapq.heappush
 def push(heap,item):real(heap,item);maximum[0]=max(maximum[0],len(heap));assert len(heap)<=64
 monkeypatch.setattr(e.heapq,'heappush',push)
 e.select_rows(root/x['scan']['scan_id']/'proposals.jsonl',x['scan']['scan_id'],x['proposal_sha256'],100);assert maximum[0]==64

def test_path_escape_and_selected_row_tamper(tmp_path):
 root,s,_=fixture(tmp_path)
 with pytest.raises(ValueError,match='escapes'):e.checked(root,'../outside','unused')
 s['entries'][0]['selected'][0]['ordinal']+=1
 with pytest.raises(ValueError,match='prespecified'):e.verify_selection(s)

def test_single_class_not_rebalanced(tmp_path):
 _,s,cfg=fixture(tmp_path)
 for entry in s['entries']:
  entry['scan']['positive']=[];entry['scan']['ignored']=[]
 r=e.run_pass(s,cfg,tmp_path/'single','cpu',binding(s,cfg),model_factory=Tiny)
 assert r['update_state']['optimizer_updates']>0
 status=json.loads((tmp_path/'single/training-scan-status.json').read_text());assert sum(x['usable'] for x in status)==12

def test_interruption_during_pass_keeps_initial_checkpoint(tmp_path,monkeypatch):
 from lidc3d import optimization
 _,s,cfg=fixture(tmp_path);b=binding(s,cfg);out=tmp_path/'interrupted'
 monkeypatch.setattr(optimization,'train_epoch',lambda *a,**k:(_ for _ in ()).throw(KeyboardInterrupt('synthetic interrupt')))
 with pytest.raises(KeyboardInterrupt):e.run_pass(s,cfg,out,'cpu',b,model_factory=Tiny)
 assert e.load_checkpoint(out,b)['state']['stage']=='INITIALIZED' and not (out/'engineering-result.json').exists()

def test_checkpoint_binding_mismatch(tmp_path):
 _,s,cfg=fixture(tmp_path);b=binding(s,cfg);out=tmp_path/'run';e.run_pass(s,cfg,out,'cpu',b,model_factory=Tiny)
 with pytest.raises(ValueError,match='identity'):e.load_checkpoint(out,{**b,'run_id':'unrelated'})
