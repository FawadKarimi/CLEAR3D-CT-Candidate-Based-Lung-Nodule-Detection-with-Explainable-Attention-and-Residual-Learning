"""Generated CPU coverage; Tiny model doubles are not CUDA/full-hybrid evidence."""
import copy,json,random,subprocess,sys
from pathlib import Path
import numpy as np
import pytest,torch
from lidc3d import scoped_epochs as e,scoped_epochs_contract as c,scoped_epoch_checkpoints as cp
from lidc3d import checkpoints,training
from lidc3d.provenance import sha256

class Tiny(torch.nn.Module):
    def __init__(self,*args):
        super().__init__();self.head=torch.nn.Linear(1,2);self.drop=torch.nn.Dropout(.2)
    def forward(self,x):
        z=x.mean((2,3,4));return self.head(self.drop(z)+.001*(random.random()+float(np.random.random())))

class Small(e.GeneratedDataset):
    def __getitem__(self,i):
        r=self.rows[i];g=torch.Generator().manual_seed(717+r['fixture_index']+100*self.epoch)
        return {'image':torch.rand(1,2,2,2,generator=g),'label':r['label'],'index':i}

def setup(depth=10):
    cfg=c.config(depth,'generated-only.pth','generated-only.json','a'*64)
    binding=c.identity(cfg,'b'*64,'synthetic-epoch-test','generated')
    rows,scans,val=e.generated_data()
    return cfg,binding,rows,scans,val

def run(out,depth=10,**kw):
    cfg,binding,rows,scans,val=setup(depth)
    return e.fit(cfg,binding,out,'cpu',rows,scans,val,Small,e.generated_validate,Tiny,
                 epochs=2,sampling=c.GENERATED_SAMPLING,**kw)

@pytest.mark.parametrize('depth',[18,10])
def test_separate_model_restart_exact_all_state(tmp_path,depth):
    run(tmp_path/'full',depth);r=run(tmp_path/'interrupted',depth,stop_after=0)
    assert r['status']=='BOUNDARY_INTERRUPTED'
    preserved={str(p):sha256(p) for p in (tmp_path/'interrupted').rglob('*') if p.is_file()}
    run(tmp_path/'resumed',depth,resume=tmp_path/'interrupted')
    result=e.compare(tmp_path/'full',tmp_path/'resumed',setup(depth)[1]);assert result['exact']
    assert {'model','optimizer','scheduler','update_state','loader_rng','torch_rng','numpy_rng','python_rng','cuda_rng'}<=result['checks'].keys()
    assert all(sha256(Path(p))==h for p,h in preserved.items())
    state=cp.load(tmp_path/'resumed',setup(depth)[1])['state']
    assert state['update_state']['optimizer_updates']==2 and state['update_state']['skipped_updates']==0
    events=[json.loads(x) for x in (tmp_path/'full/boundaries.jsonl').read_text().splitlines()]
    assert len(events)==2 and all(x['incomplete_group'] and x['denominator']==11 for x in events)
    assert all([v['samples'] for v in x['microbatches']]==[2,2,1] for x in events)
    assert not (tmp_path/'full/best.pt').exists() and not (tmp_path/'full/selection.json').exists()

def test_exact_research_sampling_three_epochs_final_endpoint(tmp_path):
    cfg,b,_,_,_=setup();b=c.identity(cfg,'b'*64,'research-fixture','research','c'*64)
    rows=e.Rows({'id':f't{i}','label':int(i<15),'fixture_index':i} for i in range(768))
    scans=[{'scan_id':f'v{i}'} for i in range(6)]
    val=[{'id':f'vrow{i}','scan_id':scans[i//64]['scan_id'],'label':int(i<15),'fixture_index':1000+i} for i in range(384)]
    def validation(model,scans,rows,device,cfg):
        # Synthetic predictions; actual completeness/loss checks remain in fit.
        return ([{'id':r['id'],'scan_id':r['scan_id'],'logits':[.1,.2]} for r in rows],
                [{'scan_id':s['scan_id'],'status':'ok','candidates':64,'predictions':64} for s in scans])
    result=e.fit(cfg,b,tmp_path/'run','cpu',rows,scans,val,Small,validation,Tiny,epochs=3,sampling=c.SAMPLING)
    assert result['endpoint_epoch']==2 and result['update_state']['optimizer_updates']==180 and result['update_state']['skipped_updates']==0
    assert [x['validation_rows'] for x in result['epochs']]==[384]*3
    for epoch in range(3):
        r=json.loads((tmp_path/f'run/sampling-{epoch}.json').read_text());ids=r['source_indices']
        assert len(ids)==960 and sum(rows[i]['label'] for i in ids)==240
        assert len({i for i in ids if rows[i]['label']==0})==720
        assert len({i for i in ids if rows[i]['label']==1})==15

@pytest.mark.parametrize('where',['status','rows','loss'])
def test_failed_validation_retains_prior_commit(tmp_path,where):
    cfg,b,rows,scans,val=setup()
    def bad(*args):
        p,s=e.generated_validate(*args)
        if where=='status':s[0]['status']='failed'
        elif where=='rows':p.pop()
        else:p[0]['logits']=[float('nan'),0]
        return p,s
    with pytest.raises(ValueError):e.fit(cfg,b,tmp_path/'bad','cpu',rows,scans,val,Small,bad,Tiny,epochs=2,sampling=c.GENERATED_SAMPLING)
    assert cp.load(tmp_path/'bad',b)['state']['epoch']==-1
    assert (tmp_path/'bad/epoch-failure.json').exists() and (tmp_path/'bad/validation-0_scan_status.jsonl').exists()
    assert not (tmp_path/'bad/epoch-result.json').exists()

@pytest.mark.parametrize('field',['scope','depth','config_hash','source_binding_sha256','selection_sha256','sampling_sha256','checkpoint_sha256','provenance_sha256'])
def test_resume_binding_change_refused(tmp_path,field):
    run(tmp_path/'prior',stop_after=0);b=setup()[1];b[field]='changed'
    with pytest.raises(ValueError,match='binding'):cp.load(tmp_path/'prior',b)

@pytest.mark.parametrize('stage',['scoped_intent','scoped_state','scoped_sealed'])
def test_incomplete_publication_retains_sealed_bytes(tmp_path,monkeypatch,stage):
    run(tmp_path/'prior',stop_after=0);b=setup()[1];old=cp.load(tmp_path/'prior',b)
    old_files={str(p):sha256(p) for p in Path(old['generation']).rglob('*') if p.is_file()}
    s=copy.deepcopy(old['state']);s['epoch']=1;s['scheduler']['epoch']=1
    monkeypatch.setattr(cp,'_boundary',lambda name:(_ for _ in ()).throw(RuntimeError('fixture crash')) if name==stage else None)
    with pytest.raises(RuntimeError):cp.publish(tmp_path/'prior',s,b)
    with pytest.raises(ValueError,match='Incomplete'):cp.load(tmp_path/'prior',b)
    assert all(sha256(Path(p))==h for p,h in old_files.items())
    assert list((tmp_path/'prior').glob('epoch-publication-failure-*.json'))

def test_keyboard_interrupt_preserves_initialized(tmp_path,monkeypatch):
    monkeypatch.setattr(e,'train_epoch',lambda *a,**k:(_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):run(tmp_path/'interrupt')
    assert cp.load(tmp_path/'interrupt',setup()[1])['state']['epoch']==-1
    assert (tmp_path/'interrupt/epoch-failure.json').exists()

def test_production_approval_cannot_be_satisfied_by_scoped_disposition(tmp_path):
    from test_ingest_and_guards import make_bundle
    bundle,cfg,_,_=make_bundle(tmp_path);bundle['reference_audit_approved']=False
    bundle['scoped_disposition']={'status':'ADOPTED','scope':c.SCOPE}
    with pytest.raises(ValueError,match='Reference audit'):training.train(bundle,cfg,tmp_path/'production',17)
    assert not (tmp_path/'production').exists()

@pytest.mark.parametrize('key',['status','scope','sampling','selection_sha256','source_binding_sha256','budget','supplied_by'])
def test_disposition_and_drift(key):
    from lidc3d.engineering import HOLD_IDS
    d={'status':'ADOPTED','scope':c.SCOPE,'source_binding_sha256':'b','configuration_sha256':{},
       'selection_sha256':c.base.SELECTION,'preparation_sha256':c.base.PREPARATION,'sampling':c.SAMPLING,
       'budget':c.BUDGET,'limits':c.RESEARCH_LIMITS,'reference_release':'HOLD','endpoint':'final_epoch_2_loss_diagnostics_only',
       'acknowledged_unresolved_holds':HOLD_IDS,'supplied_by':'synthetic fixture','instruction':'synthetic only','date':'fixture'}
    for k in ('label_guided_sampling','prior_validation_exposure','pretraining_overlap_uncertain','sampled_monitoring','not_scientific_approval','fp32_not_amp_coverage'):d[k]=True
    assert c.disposition(d,'b',{})==d
    d[key]=None
    with pytest.raises(ValueError):c.disposition(d,'b',{})

def test_default_plan_no_torch_and_worker_gate():
    root=Path(__file__).parents[1];script=root/'scripts/scoped_epochs_watchdog.py'
    p=subprocess.run([sys.executable,'-B',str(script)],capture_output=True,text=True)
    assert p.returncode==0 and json.loads(p.stdout)['torch_imported'] is False
    assert json.loads(p.stdout)['budget']['total_seconds']==1800
    assert subprocess.run([sys.executable,'-B',str(root/'scripts/scoped_epochs_worker.py')],capture_output=True).returncode!=0

def launcher():
    sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
    import scoped_epochs_watchdog
    return scoped_epochs_watchdog

def test_absent_prerequisites_never_launch(monkeypatch):
    m=launcher();monkeypatch.setattr(m.w,'OwnedWorker',lambda *a:pytest.fail('unexpected launch'))
    with pytest.raises(ValueError):m.main(['--execute','--mode','research','--depth','18'])

def test_controller_failure_stops_later_legs_and_only_owns_worker(tmp_path,monkeypatch):
    m=launcher();(tmp_path/'runs').mkdir();monkeypatch.setattr(m,'__file__',str(tmp_path/'work/scripts/scoped_epochs_watchdog.py'))
    monkeypatch.setattr(m,'prerequisites',lambda r:{})
    monkeypatch.setattr(m.w,'available_ram',lambda:100*c.base.GIB);monkeypatch.setattr(m.w,'disk_free',lambda p:100*c.base.GIB)
    monkeypatch.setattr(m.w.reviewed,'gpu_snapshot',lambda:{'free_bytes':11*c.base.GIB,'processes':[{'pid':777,'gpu_bytes':128}]})
    calls=[]
    class Owned:
        identity={'pid':123,'start_ticks':456}
        def __init__(self,*args):calls.append('launch')
        def cleanup(self):calls.append('cleanup_owned');return {'worker_exit':1}
    monkeypatch.setattr(m.w,'OwnedWorker',Owned)
    monkeypatch.setattr(m.w,'supervise',lambda *a,**k:{'status':'FAIL','cleanup':{'worker_exit':1}})
    out=tmp_path/'runs/generated-resume18-001'
    assert m.main(['--execute','--depth','18','--binding','fixture','--binding-sha256','b','--out',str(out)])==1
    report=json.loads((out/'acceptance.json').read_text())
    assert calls==['launch','cleanup_owned'] and report['status']=='FAIL'
    assert report['stages']['full']=='FAIL' and all(report['stages'][k]=='NOT_RUN' for k in ('interrupted','resumed','compare'))
    assert not (out/'interrupted').exists()

def test_fixed_budget_arithmetic_and_no_resize():
    assert 2*c.QUAL_LIMITS['total_seconds']+2*c.RESEARCH_LIMITS['total_seconds']==1800
    assert c.BUDGET['no_budget_transfer'] is True
    for limits in (c.QUAL_LIMITS,c.RESEARCH_LIMITS):
        assert limits['rss_bytes']==8*c.base.GIB and limits['allocator_bytes']==4*c.base.GIB
        assert limits['gpu_bytes']==6*c.base.GIB and limits['min_gpu_bytes']==10*c.base.GIB

def test_epoch_validation_limit_during_repeated_training_phases(tmp_path):
    m=launcher();p=tmp_path/'epoch-phase.json';limits={**c.RESEARCH_LIMITS,'validation_seconds':2}
    b=m.EpochBudget(10,limits,p)
    m.w.save(p,{'phase':'validation','epoch':0,'monotonic':11})
    assert b.check(12,'training',1,1,40*c.base.GIB) is None
    m.w.save(p,{'phase':'training','epoch':1,'monotonic':12.5})
    assert b.check(12.5,'training',1,1,40*c.base.GIB) is None
    m.w.save(p,{'phase':'validation','epoch':1,'monotonic':13})
    assert b.check(14,'training',1,1,40*c.base.GIB)=='EPOCH_VALIDATION_TIME_LIMIT'

def test_epoch_budget_keeps_owned_memory_limits(tmp_path):
    m=launcher();b=m.EpochBudget(0,c.RESEARCH_LIMITS,tmp_path/'absent')
    assert b.check(1,'training',9*c.base.GIB,0,40*c.base.GIB)=='SAMPLED_RSS_LIMIT'
