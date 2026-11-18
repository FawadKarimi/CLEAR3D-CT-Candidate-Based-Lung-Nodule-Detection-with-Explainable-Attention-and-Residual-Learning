"""Serial restart qualification or scoped epochs. Default is plan only, no Torch."""
import argparse,json,math,signal,sys,time,uuid
from pathlib import Path
sys.path[:0]=['/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-diagnostic-training-review-20261013-6e91ab/work/src','/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-diagnostic-training-review-20261013-6e91ab/work/scripts']
import user_watchdog as w
from lidc3d import diagnostic_training as c

class EpochBudget(w.SampleBudget):
    """Keep commissioned outer limits; additionally watch the current validation interval."""
    def __init__(self,start,limits,path):
        super().__init__(start,limits);self.path=Path(path);self.last=None;self.validation_elapsed=0.
    def check(self,now,phase,rss,gpu,ram):
        reason=super().check(now,phase,rss,gpu,ram)
        if reason:return reason
        if not self.path.exists():return None
        v=w.load(self.path)
        c.require(v['phase'] in ('training','validation') and type(v['epoch']) is int and 0<=v['epoch']<=2,
                  'Invalid epoch phase')
        c.require(math.isfinite(v['monotonic']) and self.start<=v['monotonic']<=now,'Invalid epoch phase clock')
        if self.last and v!=self.last:
            c.require(v['monotonic']>=self.last['monotonic'],'Epoch phase clock reversed')
            c.require(v['epoch']>=self.last['epoch'],'Epoch sequence reversed')
            if self.last['phase']=='validation':self.validation_elapsed+=v['monotonic']-self.last['monotonic']
        self.last=v
        active=now-v['monotonic'] if v['phase']=='validation' else 0.
        if self.validation_elapsed+active>self.limits['validation_seconds']:return 'EPOCH_VALIDATION_TIME_LIMIT'
        return None

def prerequisites(r):
    c.require(r.get('depth') in (18,10) and r.get('mode')=='research','Mode/depth absent')
    c.require(r.get('binding') and r.get('binding_sha256') and r.get('out'),'Missing binding/output')
    b=w.bindings(r['binding'],r['binding_sha256'])
    c.require(b.get('epoch_budget')==c.BUDGET,'Combined budget changed')
    c.require(r.get('acknowledge_shared_gpu_permission') and r.get('acknowledge_sampled_risk'),'Shared/sampled acknowledgement required')
    for depth in (18,10):
        spec=b['epoch_configs'][str(depth)];cfg=c.read(spec['path'],spec['sha256']);c.check_config(cfg)
        m=cfg['medicalnet'];c.require(m['depth']==depth and c.digest(m['checkpoint'])==c.base.WEIGHTS[depth],'Weights/depth mismatch')
        c.base.provenance(c.read(m['provenance'],m['provenance_sha256']))
    if r['depth']==10:
        c.require(r.get('previous') and r.get('previous_sha256'),'ResNet18 must pass first')
        c.accepted(c.read(r['previous'],r['previous_sha256']),r['mode'],18,r['binding_sha256'])
    if r['mode']=='research':
        c.require(r.get('disposition') and r.get('disposition_sha256'),'Separate adoption absent')
        c.disposition(c.read(r['disposition'],r['disposition_sha256']),r['binding_sha256'],
                      {d:v['sha256'] for d,v in b['epoch_configs'].items()})
        c.verify_inputs()
        for depth in (18,10):
            spec=b['reused_qualifications'][str(depth)]
            c.old.accepted(c.read(spec['path'],spec['sha256']),'generated',depth,spec['source_binding_sha256'])
    return b

def validate_worker(out,device,limits):
    r=w.load(out/'request.json');v=w.load(out/'worker-result.json')
    expected='BOUNDARY_INTERRUPTED' if r['leg']=='interrupted' else 'PASS'
    c.require(v['status']==expected and v['device']==device and v['depth']==r['depth']
              and v['source_binding_sha256']==r['binding_sha256'],'Worker identity/outcome mismatch')
    c.require(w.load(out/'phase.json')['name']=='complete','Missing completion phase')
    b=w.load(r['binding']);spec=b['epoch_configs'][str(r['depth'])];cfg=c.read(spec['path'],spec['sha256'])
    identity=c.identity(cfg,r['binding_sha256'],r['run_id'],r['mode'],r.get('disposition_sha256'))
    c.require(v['identity']==identity,'Worker identity changed')
    d=v['phase_seconds'];c.require(set(d)=={'setup','training','validation'} and all(math.isfinite(x) and x>=0 for x in d.values()),'Invalid phase durations')
    c.require(d['setup']<=limits['setup_seconds'] and d['training']+d['validation']<=limits['run_seconds']
              and d['validation']<=limits['validation_seconds'] and v['max_allocated']<=limits['allocator_bytes'],'Worker phase/allocator limit')
    if r['leg']=='compare':c.require(v.get('exact') is True and all(v['checks'].values()),'Exact resume mismatch')
    else:
        updates=1 if r['leg']=='interrupted' else 2 if r['mode']=='generated' else 180
        c.require(v['update_state']['optimizer_updates']==updates and v['update_state']['skipped_updates']==0,'Wrong update count')
        pointer=c.read(out/'state/epoch-commit.json')
        c.require(pointer['status']=='COMMITTED' and pointer['commit_sha256']==v['checkpoint_commit_sha256'],'Final commit mismatch')
        rel=Path(pointer['generation']);c.require(len(rel.parts)==2 and rel.parts[0]=='epoch-checkpoints','Unsafe checkpoint generation')
        gen=out/'state'/rel;record=c.read(gen/'commit.json',pointer['commit_sha256'])
        c.require(record['binding']==identity and c.digest(gen/'state.pt')==record['state_sha256'],'Checkpoint inventory mismatch')
        expected_rows=4 if r['mode']=='generated' else 384
        c.require(all(x['validation_rows']==expected_rows for x in v['epochs']) and v['epochs'],'Incomplete checkpoint validation')
        if r['mode']=='research':c.require(v['endpoint_epoch']==2 and len(v['epochs'])==3,'Incomplete three-epoch run')
    c.require(json.loads(json.dumps(v['diagnostics']))==json.loads(json.dumps(c.diagnostic_evidence(out/'state'))),'Diagnostic evidence mismatch')
    return v


OUT1800='/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64/runs/resnet18-constlr-001'
def apply_budget1800():
    c.RESEARCH_LIMITS={**c.RESEARCH_LIMITS,'total_seconds':1800,'run_seconds':1800}
def main(argv=None):
    apply_budget1800()
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--mode',choices=['research'],default='research');p.add_argument('--depth',type=int,choices=[18,10])
    for name in ('binding','binding-sha256','out','disposition','disposition-sha256','previous','previous-sha256',
                 'qualification18','qualification18-sha256','qualification10','qualification10-sha256'):p.add_argument('--'+name)
    p.add_argument('--acknowledge-shared-gpu-permission',action='store_true');p.add_argument('--acknowledge-sampled-risk',action='store_true');a=p.parse_args(argv);r=vars(a)
    if not a.execute:
        print(json.dumps({'status':'PLAN_ONLY','budget':c.BUDGET,'reference_release':'HOLD','cap4000':'NOT_ADOPTED','torch_imported':'torch' in sys.modules}));return 0
    prerequisites(r);root=Path('/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-diagnostic-training-review-20261013-6e91ab');out=Path(a.out).absolute()
    expected='resnet18-constlr-001'
    c.require(a.depth==18,'ResNet18 extended budget only')
    c.require(str(out)==OUT1800, "Output not bound to extended run")
    c.require(out.parent==Path('/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64/runs') and out.name==expected and out.parent.is_dir() and not out.exists()
              and not any(x.is_symlink() for x in [out,*out.parents]),'Unsafe/existing destination')
    base=c.QUAL_LIMITS if a.mode=='generated' else c.RESEARCH_LIMITS
    c.require(w.available_ram()>=base['min_ram_bytes'] and w.disk_free(out.parent)>=base['min_disk_bytes'],'Resource headroom')
    out.mkdir();started=time.monotonic();deadline=started+base['total_seconds'];handlers={};stop=[False];child=None
    legs=['full','interrupted','resumed','compare'] if a.mode=='generated' else ['research']
    report={'status':'FAIL','mode':a.mode,'depth':a.depth,'source_binding_sha256':a.binding_sha256,'budget':c.BUDGET,
            'reference_release':'HOLD','diagnostic_scope':c.SCOPE,'stages':{k:'NOT_RUN' for k in legs},'throughput_scope':'shared GPU engineering only'}
    try:
        r={**r,'run_id':uuid.uuid4().hex};w.save(out/'invocation.json',{'argv':sys.argv if argv is None else argv,'request':r})
        for s in (signal.SIGINT,signal.SIGTERM):handlers[s]=signal.signal(s,lambda *args:stop.__setitem__(0,True))
        results=[];gpu_samples=[]
        for leg in legs:
            c.require(not stop[0],'External termination');remaining=deadline-time.monotonic();c.require(remaining>0,'Fixed total budget exhausted')
            report['stages'][leg]='RUNNING'
            limits={**base,'total_seconds':remaining,'setup_seconds':min(base['setup_seconds'],remaining),
                    'run_seconds':min(base['run_seconds'],remaining),'validation_seconds':min(base['validation_seconds'],remaining)}
            stage=out/leg;stage.mkdir();snap=w.reviewed.gpu_snapshot();w.append(stage/'gpu-before.jsonl',snap)
            c.require(snap['free_bytes']>=base['min_gpu_bytes'],'GPU headroom')
            req={**r,'out':str(stage),'leg':leg,'limits':limits,'resume':str(out/'interrupted') if leg=='resumed' else None,
                 'full':str(out/'full'),'resumed':str(out/'resumed')};w.save(stage/'request.json',req)
            device='cpu' if leg=='compare' else 'cuda'
            with (stage/'worker.log').open('xb',buffering=0) as log:
                child=w.OwnedWorker([sys.executable,'-B','/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64/D-plan/constlr_worker.py','--request',str(stage/'request.json')],w.child_env(device),log)
                w.save(stage/'owned-worker.json',child.identity)
                observed=w.supervise(child,stage,device,limits,observe=w.reviewed.gpu_snapshot,stop=lambda:stop[0] or time.monotonic()>deadline,
                                     validate_evidence=validate_worker,completion_reason='SCOPED_EPOCHS_ONLY',
                                     budget_factory=lambda start,lim:EpochBudget(start,lim,stage/'epoch-phase.json'))
            c.require(observed['status']=='PASS' and observed['cleanup']['worker_exit']==0,'Stage failed; later stages NOT_RUN')
            usable=False
            if (stage/'samples.jsonl').exists():
                with (stage/'samples.jsonl').open() as f:
                    for i,line in enumerate(f):
                        c.require(i<2000 and len(line)<1048576,'Sample admission bound');usable|=bool(json.loads(line).get('owned_gpu_bytes'))
            if device=='cuda':c.require(usable,'Computation complete but no usable GPU monitoring sample');gpu_samples.append(usable)
            report['stages'][leg]='PASS';results.append(observed);child=None
        c.require(time.monotonic()<=deadline,'Fixed budget exceeded including comparison/cleanup')
        report.update(status='PASS',exact_resume=a.mode=='generated',all_owned_workers_reaped=True,
                      usable_owned_gpu_samples=all(gpu_samples),results=results)
        return 0
    except BaseException as error:
        report['error']=repr(error)
        for leg,status in report['stages'].items():
            if status=='RUNNING':report['stages'][leg]='FAIL'
        if child is not None:report['cleanup']=child.cleanup()
        return 1
    finally:
        report['wall_seconds']=time.monotonic()-started;w.save(out/'acceptance.json',report)
        for s,old in handlers.items():signal.signal(s,old)
if __name__=='__main__':raise SystemExit(main())
