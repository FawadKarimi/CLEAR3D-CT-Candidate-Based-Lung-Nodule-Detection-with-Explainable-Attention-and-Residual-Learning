"""Generated CPU integration; model doubles explicitly identified, no research inputs."""
import copy,json,hashlib,random,resource,sys,time
from pathlib import Path
import pytest,numpy as np,torch
from lidc3d.streaming_selection import top_rows,verified_stream
from lidc3d.indexed import CandidateIndex,lines
from lidc3d import training,checkpoints
from lidc3d.provenance import sha256,write_jsonl,write_json,read_json,read_jsonl
from lidc3d.epoch_evidence import Observer
from test_ingest_and_guards import make_bundle
from test_checkpoint_sets import assert_same
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import cohort_watchdog as cw

@pytest.mark.parametrize('cap',[0,1,7,400])
@pytest.mark.parametrize('threshold',[0,.5,1])
def test_stable_heap_exact(cap,threshold):
    rng=random.Random(17);rows=[{'scan_id':'s','id':str(i%43),'log_response':rng.choice([0,.5,1]),'ordinal':i} for i in range(5000)]
    expected=sorted([r for r in rows if r['log_response']>=threshold],key=lambda r:(-r['log_response'],r['id']))[:cap]
    assert top_rows(iter(rows),'s',threshold,cap)==expected

@pytest.mark.parametrize('mode',['ok','hash','count','last','oversize'])
def test_complete_stream_even_zero_cap(tmp_path,mode):
    p=tmp_path/'pool';rows=[{'scan_id':'s','id':f's:candidate:{i:06d}','log_response':1} for i in range(9)];write_jsonl(p,rows)
    expected={'sha256':sha256(p),'bytes':p.stat().st_size};count=9
    if mode=='hash':expected['sha256']='0'*64
    if mode=='count':count=8
    if mode=='last':p.write_bytes(p.read_bytes()+b'{}'+bytes([10]))
    if mode=='oversize':p.write_bytes(b' '*16385)
    if mode=='ok':assert top_rows(verified_stream(p,'s',expected,count),'s',0,0)==[]
    else:
        with pytest.raises(ValueError):top_rows(verified_stream(p,'s',expected,count),'s',0,0)

def synthetic_scans():
    return [{'scan_id':f's{i}','patient_id':f'p{i}','split':'train' if i<708 else 'val','pixel_fingerprint':f'fp{i}','positive':[{'id':f'r{i}','center_lps':[0,0,0],'radius_mm':1,'strict_radius':True}],'ignored':[{'center_lps':[2,2,2],'radius_mm':1,'strict_radius':True}]} for i in range(797)]
def row(sid,j):return {'id':f'{sid}:{j}','scan_id':sid,'zyx':[j%3]*3,'center_lps':[0,0,0] if j==0 else [5,5,5],'log_response':.5}

def test_worst_cap_disk_index(tmp_path):
    start=time.monotonic();scans=synthetic_scans();p=tmp_path/'rows.jsonl'
    write_jsonl(p,(row(s['scan_id'],j) for s in scans for j in range(400)))
    index=CandidateIndex(p,scans,tmp_path)
    assert len(index.train)==283200 and len(index.val)==35600
    assert index.train[0]['label']==1 and index.train[283199]['label']==0
    assert len(index.val.for_scan('s796'))==400
    assert not isinstance(index.train,list)
    assert sum(1 for _ in index.train)==283200
    assert sum(1 for _ in index.val)==35600
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    report={'generated_scans':797,'training_rows':283200,'validation_rows':35600,'elapsed_seconds':time.monotonic()-start,'process_high_water_RSS_bytes':peak,'index_bytes':index.path.stat().st_size,'model':'NOT_INVOKED'}
    (tmp_path/'scale-report.json').write_text(json.dumps(report));print('SCALE_REPORT='+json.dumps(report))
    assert peak<8*1024**3;index.close()

@pytest.mark.parametrize('mode',['cap','duplicate','outside','oversized','invalid_coordinate','test'])
def test_index_refuses_admission(tmp_path,mode):
    scans=synthetic_scans()[:1];rows=[row('s0',0),row('s0',1)]
    if mode=='cap':rows=[row('s0',i) for i in range(401)]
    if mode=='duplicate':rows[1]['id']=rows[0]['id']
    if mode=='outside':rows[1]['scan_id']='outside'
    if mode=='invalid_coordinate':rows[1]['zyx']=[float('inf')]*3
    if mode=='test':scans[0]['split']='test'
    p=tmp_path/'rows';p.write_text(''.join(json.dumps(r)+chr(10) for r in rows))
    if mode=='oversized':p.write_text('x'*16385)
    with pytest.raises((ValueError, __import__('sqlite3').IntegrityError)):CandidateIndex(p,scans,tmp_path)

def test_ignored_precedence_and_empty(tmp_path):
    scans=synthetic_scans()[:2];scans[0]['ignored'].append({'center_lps':[0,0,0],'radius_mm':1,'strict_radius':True})
    rows=[row('s0',0),{**row('s0',1),'center_lps':[2,2,2]},row('s0',2)]
    p=tmp_path/'rows';write_jsonl(p,rows);idx=CandidateIndex(p,scans,tmp_path)
    assert idx.receipt['training_labels']=={'0':1,'1':1,'-1':1};assert len(idx.train.for_scan('s1'))==0;idx.close()

class Tiny(torch.nn.Module):
    """Synthetic model double, NOT hybrid-model coverage."""
    def __init__(self,cfg):super().__init__();self.head=torch.nn.Linear(1,2)
    def forward(self,x):return self.head(x.mean(tuple(range(1,x.ndim))).reshape(-1,1))

def cohort_bundle(tmp_path):
    bundle,cfg,original,_=make_bundle(tmp_path);scans=synthetic_scans()
    for s in scans:s.update(image_path=original[0]['image_path'],valid_path=original[0]['valid_path'],status='ok')
    # Zero-candidate training and validation scans remain cohort members.
    rows=(row(s['scan_id'],j) for s in scans if s['scan_id'] not in ('s707','s796') for j in range(2))
    Path(bundle['scans']).write_text(''.join(json.dumps(s)+chr(10) for s in scans));Path(bundle['candidates']).unlink();write_jsonl(bundle['candidates'],rows)
    cfg['training'].update(max_epochs=2,warmup_epochs=5,microbatch=2,accumulation=8,amp=False)
    # Preserve production hybrid configuration as metadata; Tiny replaces forward only.
    cfg['model'].update(kind='hybrid',width=16,patch_size=48,embed_dim=128,heads=4,layers=2,ff_dim=256,dropout=.1,head_dropout=.3,scales=[0,1,2],positional_encoding=True,fusion='hierarchical')
    Path(bundle['resolved_config']).write_text(json.dumps(cfg))
    for name in bundle['files']:bundle['files'][name]=sha256(name)
    return bundle,cfg,scans

def fast_model(monkeypatch):
    monkeypatch.setattr(training,'DetectorClassifier',Tiny);monkeypatch.setattr(training,'signature',lambda *a,**k:{'model_double':'Tiny_mean_linear_CPU'})
    # Avoid enormous 48^3 generated patch copies: dataset returns a generated scalar
    # while exercising the indexed row access, labels, epoch shuffle and actual optimizer.
    seen=[]
    from lidc3d.data import CandidateDataset
    def get(self,i):
        r=self.rows[i]
        if self.augment:seen.append((self.epoch,r['id']))
        return {'image':torch.tensor([[[[float(r['zyx'][0])/10]]]]),'label':int(r.get('label',-1)),'index':i}
    monkeypatch.setattr(CandidateDataset,'__getitem__',get)
    return seen

def test_two_epoch_cohort_resume_and_supervisor_evidence(tmp_path,monkeypatch):
    bundle,cfg,scans=cohort_bundle(tmp_path);seen=fast_model(monkeypatch)
    outer=tmp_path/'full';outer.mkdir();full=outer/'run';observer=Observer(outer);observer('setup')
    training.train(bundle,cfg,full,17,'cpu',observer=observer)
    first=list(seen);seen.clear();N=1414
    assert len(first)==2*N
    for epoch in (0,1):
        ids=[i for e,i in first if e==epoch];assert len(ids)==len(set(ids))==N
        stats=read_jsonl(full/f'validation_epoch_{epoch:03d}_scan_status.jsonl');assert len(stats)==89 and stats[-1]['candidates']==0 and stats[-1]['status']=='ok'
        ep=read_json(full/f'epoch_{epoch:03d}.json');assert ep['optimizer_updates_this_epoch']==89
        assert ep['learning_rates'][0]==pytest.approx(cfg['training']['lr_head']*(epoch+1)/5)
    broken=tmp_path/'broken';real=training.publish_set
    def stop(*a,**k):
        real(*a,**k)
        if a[1]['epoch']==0:raise RuntimeError('explicit synthetic stop')
    with monkeypatch.context() as m:
        m.setattr(training,'publish_set',stop)
        with pytest.raises(RuntimeError,match='explicit synthetic stop'):training.train(bundle,cfg,broken,17,'cpu')
    seen.clear();resumed=tmp_path/'resumed';training.train(bundle,cfg,resumed,17,'cpu',broken/'last.pt')
    assert seen==[x for x in first if x[0]==1]
    left=checkpoints.load_set(full/'last.pt',{});right=checkpoints.load_set(resumed/'last.pt',{})
    for part in ('last','best'):
        for key in ('model','optimizer','scaler','update_state','scheduler','loader_rng','torch_rng','numpy_rng','python_rng','cuda_rng','epoch','best_cpm','stopping_best','stale'):assert_same(left[part][key],right[part][key])
    (outer/'request.json').write_text(json.dumps({'validation_ids':[s['scan_id'] for s in scans if s['split']=='val'],'config':cfg}))
    (outer/'worker-result.json').write_text(json.dumps({'status':'PASS','device':'cpu','epochs':[0,1],'start_epoch':0,'end_epoch':1,'max_allocated':0}))
    result=cw.evidence(outer,'cpu',cw.LIMITS);assert result['validation_scans']==89
    stats=read_jsonl(full/'validation_scan_status.jsonl');stats[0]['status']='failed';(full/'validation_scan_status.jsonl').write_text(''.join(json.dumps(s)+chr(10) for s in stats))
    with pytest.raises(RuntimeError,match='Failed validation'):cw.evidence(outer,'cpu',cw.LIMITS)

@pytest.mark.parametrize('failure_pass',[1,3])
def test_cohort_validation_failure_before_selection(tmp_path,monkeypatch,failure_pass):
    bundle,cfg,scans=cohort_bundle(tmp_path);fast_model(monkeypatch);real=training.predict;calls=[0];out=tmp_path/'run'
    def failing(*a,**k):
        result,status=real(*a,**k);calls[0]+=1
        if calls[0]==failure_pass:status[0].update(status='failed',reason='synthetic',predictions=0)
        return result,status
    monkeypatch.setattr(training,'predict',failing)
    with pytest.raises(ValueError,match='Incomplete validation'):training.train(bundle,cfg,out,17,'cpu')
    assert not (out/'selection.json').exists();assert (out/'failure.json').exists()
    if failure_pass==1:assert not (out/'checkpoint_commit.json').exists()
    else:assert read_json(out/'checkpoint_commit.json')['status']=='COMMITTED'

def test_multi_epoch_budget_and_plan_only(tmp_path,capsys):
    b=cw.EpochBudget(0,cw.LIMITS,tmp_path)
    assert b.check(1,'setup',0,None,100*1024**3) is None
    for epoch in range(2):
        (tmp_path/'phase-detail.json').write_text(json.dumps({'name':'validation','monotonic':10+epoch*20}))
        assert b.check(11+epoch*20,'training',0,None,100*1024**3) is None
    assert b.check(8000,'training',0,None,100*1024**3)=='VALIDATION_TIME_LIMIT'
    assert b.check(50,'training',9*1024**3,None,100*1024**3)=='SAMPLED_RSS_LIMIT'
    assert cw.main([])==0;assert 'PLAN_ONLY' in capsys.readouterr().out
    with pytest.raises(RuntimeError,match='Missing prerequisites'):cw.main(['--execute'])

def test_tracked_order_and_duplicate_refusal(tmp_path):
    from lidc3d.indexed import TrackedLoader
    batch={'index':torch.tensor([1,0]),'label':torch.tensor([0,1])}
    list(TrackedLoader([batch],tmp_path/'good.json',2))
    assert read_json(tmp_path/'good.json')['classes']=={'0':1,'1':1}
    with pytest.raises(ValueError,match='Repeated'):
        list(TrackedLoader([batch,batch],tmp_path/'bad.json',2))
    assert read_json(tmp_path/'bad-failure.json')['status']=='INCOMPLETE'

def test_validation_budget_includes_metrics(tmp_path):
    b=cw.EpochBudget(0,cw.LIMITS,tmp_path)
    (tmp_path/'phase-detail.json').write_text(json.dumps({'name':'metrics','monotonic':7199,'validation_started':0}))
    assert b.check(7201,'training',0,None,100*1024**3)=='VALIDATION_TIME_LIMIT'

def test_scan_admission_before_parse(tmp_path):
    from lidc3d.provenance import verify_bundle
    p=tmp_path/'scans';p.write_bytes(b'x'*(1024*1024+1))
    with pytest.raises(ValueError,match='Scan metadata admission'):verify_bundle({'scans':str(p),'files':{str(p):sha256(p)}})

def test_completed_resume_finalization_is_explicit(tmp_path,monkeypatch):
    bundle,cfg,_,_=make_bundle(tmp_path);fast_model(monkeypatch)
    first=tmp_path/'first';training.train(bundle,cfg,first,17,'cpu')
    outer=tmp_path/'continued';outer.mkdir();observer=Observer(outer);observer('setup')
    training.train(bundle,cfg,outer/'run',17,'cpu',first/'last.pt',observer=observer)
    (outer/'request.json').write_text(json.dumps({'validation_ids':['s1'],'config':cfg,'resume':str(first/'last.pt')}))
    (outer/'worker-result.json').write_text(json.dumps({'status':'PASS','device':'cpu','epochs':[],'start_epoch':2,'end_epoch':1,'max_allocated':0}))
    assert cw.evidence(outer,'cpu',cw.LIMITS)['epochs']==[]
    assert not (outer/'run/optimization_boundaries.jsonl').exists()
