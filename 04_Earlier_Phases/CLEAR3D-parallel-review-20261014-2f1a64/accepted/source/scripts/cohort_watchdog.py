"""Production epoch evidence, sampled monitoring; plan-only unless explicitly executed."""
import argparse,json,math,os,signal,sys,time
from pathlib import Path
import user_watchdog as w
LIMITS={**w.PILOT_LIMITS,'setup_seconds':3600,'run_seconds':54000,'validation_seconds':7200,'total_seconds':57600,'rss_bytes':8*w.GIB,'allocator_bytes':4*w.GIB,'gpu_bytes':6*w.GIB}
def records(path,maximum,line_bytes=1048576):
    with Path(path).open('rb') as f:
        count=0
        while True:
            raw=f.readline(line_bytes+1)
            if not raw:break
            w.require(count<maximum and len(raw)<=line_bytes and raw.strip(),'Evidence admission limit');count+=1;yield json.loads(raw)
class EpochBudget(w.SampleBudget):
    def __init__(self,start,limits,out):super().__init__(start,limits);self.out=Path(out)
    def check(self,now,phase,rss,gpu,ram):
        reason=super().check(now,phase,rss,gpu,ram)
        detail=self.out/'phase-detail.json'
        if detail.exists():
            d=w.load(detail)
            if d['name'] in ('validation','metrics','final_metrics') and now-d.get('validation_started',d['monotonic'])>self.limits['validation_seconds']:return 'VALIDATION_TIME_LIMIT'
        return reason

def evidence(out,device,limits):
    out=Path(out);run=out/'run';request=w.load(out/'request.json');summary=w.load(out/'worker-result.json')
    w.require(summary['status']=='PASS' and summary['device']==device,'Incomplete worker')
    w.require(w.load(out/'phase.json')['name']=='complete','Missing complete phase')
    index=w.load(run/'candidate-index.json');counts=index['per_scan'];validation=request['validation_ids']
    w.require(len(validation)==len(set(validation)) and set(validation).issubset(counts),'Invalid validation cohort')
    completed=summary['epochs'];w.require(completed==list(range(summary['start_epoch'],summary['end_epoch']+1)),'Incomplete epoch sequence')
    cfg=request['config']['training'];batches=(index['roles']['train']+cfg['microbatch']-1)//cfg['microbatch'];expected=(batches+cfg['accumulation']-1)//cfg['accumulation']
    totals={e:[0,0,0,0.] for e in completed}
    w.require(bool(completed) or request.get('resume'),'Empty execution without explicit resume')
    for event in (records(run/'optimization_boundaries.jsonl',max(1,len(completed)*expected*2)) if completed else []):
        w.require(event['kind']=='accumulation_boundary' and event['epoch'] in totals,'Unexpected optimization evidence')
        w.require(event['status'] in ('UPDATED','SKIPPED_AMP_NONFINITE'),'Failed boundary')
        t=totals[event['epoch']];t[0]+=1;t[1]+=event['optimizer_updates'];t[2]+=event['status']=='SKIPPED_AMP_NONFINITE'
        for m in event['microbatches']:
            w.require(math.isfinite(m['weighted_loss_sum']) and m['weight_sum']>0,'Nonfinite loss evidence');t[3]+=m['samples']
    for epoch in completed:
        count,updates,skips,rows=totals[epoch]
        w.require(count==expected and updates+skips==count and updates>0 and rows==index['roles']['train'],'Epoch accounting mismatch')
        if not cfg['amp']:w.require(updates==expected and skips==0,'FP32 skipped update')
        consumed=w.load(run/f'consumption_epoch_{epoch:03d}.json');w.require(consumed['status']=='COMPLETE' and consumed['rows']==index['roles']['train'] and consumed['classes']=={k:index['training_labels'][k] for k in ('0','1')},'Consumption mismatch')
        e=w.load(run/f'epoch_{epoch:03d}.json');w.require(e['optimizer_updates_this_epoch']==updates,'Epoch update mismatch')
    for prefix in [f'validation_epoch_{e:03d}' for e in completed]+['validation']:
        statuses=list(records(run/(prefix+'_scan_status.jsonl'),89))
        w.require(len(statuses)==len(validation) and {s['scan_id'] for s in statuses}==set(validation),'Incomplete validation cohort')
        w.require(all(s['status']=='ok' and s['candidates']==s['predictions']==counts[s['scan_id']] for s in statuses),'Failed validation scan')
    selection=w.load(run/'selection.json');w.require(selection['status']=='TRAINING_COMPLETE','Incomplete training')
    pointer=w.load(run/'checkpoint_commit.json');w.require(pointer['status']=='COMMITTED','Incomplete publication')
    gen=run/pointer['generation'];w.require(gen.resolve().parent==(run/'checkpoints').resolve(),'Checkpoint escape')
    w.require(w.digest(gen/'commit.json')==pointer['commit_sha256'],'Checkpoint commit changed')
    commit=w.load(gen/'commit.json')
    w.require(commit['epoch']==summary['end_epoch'],'Final epoch mismatch')
    if 'bundle' in request:
        import hashlib
        def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
        w.require(w.digest(request['bundle'])==request['bundle_sha256'],'Bundle binding changed')
        expected={'config_hash':canonical(request['config']),'bundle_hash':canonical(w.load(request['bundle'])),'seed':request['seed'],'device_type':device,'amp':bool(cfg['amp'] and device=='cuda')}
        w.require(all(commit['binding'].get(k)==v for k,v in expected.items()),'Checkpoint request binding mismatch')
    for name,digest in commit['files'].items():w.require(w.digest(gen/name)==w.digest(run/name)==digest,'Checkpoint file changed')
    w.require(selection['checkpoint_sha256']==commit['files']['best.pt'],'Selected best changed')
    w.require(summary['max_allocated']<=limits['allocator_bytes'],'Allocator limit')
    for event in records(out/'phases.jsonl',10000):
        if 'completed_validation_seconds' in event:w.require(event['completed_validation_seconds']<=limits['validation_seconds'],'Completed validation time limit')
    return {'status':'PASS','epochs':completed,'per_epoch':totals,'checkpoint_commit':pointer['commit_sha256'],'validation_scans':len(validation)}

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--binding');p.add_argument('--binding-sha256');p.add_argument('--bundle');p.add_argument('--config');p.add_argument('--out');p.add_argument('--seed',type=int,default=17);p.add_argument('--resume');p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--acknowledge-sampled-risk',action='store_true');p.add_argument('--acknowledge-shared-gpu-permission',action='store_true');a=p.parse_args(argv)
    if not a.execute:print(json.dumps({'status':'PLAN_ONLY','limits':LIMITS,'reference_release':'HOLD','requires':'genuine approved frozen development bundle; this entry cannot adopt exploratory disposition'}));return 0
    w.require(a.binding and a.binding_sha256 and a.bundle and a.config and a.out and a.acknowledge_sampled_risk,'Missing prerequisites')
    w.bindings(a.binding,a.binding_sha256)
    bundle=w.load(a.bundle);w.require(bundle.get('reference_audit_approved') is True and bundle.get('development_config_frozen') is True,'Scientific approval/freeze required')
    import yaml
    config=yaml.safe_load(Path(a.config).read_text());w.require(config['training']['max_epochs']==2 and config['training']['microbatch']==2 and config['training']['accumulation']==8 and config['training']['amp'] is False and config['training']['warmup_epochs']==5 and a.seed==17,'Unreviewed training configuration')
    scans=list(records(bundle['scans'],797));w.require(sum(s['split']=='train' for s in scans)==708 and sum(s['split']=='val' for s in scans)==89,'708/89 development required')
    out=Path(a.out).absolute();w.require(not out.exists() and out.parent.is_dir() and not any(x.is_symlink() for x in [out,*out.parents]),'Unsafe/existing output')
    w.require(w.available_ram()>=LIMITS['min_ram_bytes'] and w.disk_free(out.parent)>=LIMITS['min_disk_bytes'],'Insufficient headroom')
    if a.device=='cuda':w.require(a.acknowledge_shared_gpu_permission,'Shared permission acknowledgement required')
    out.mkdir();child=None;handlers={};stop=[False]
    try:
        observe=w.reviewed.gpu_snapshot if a.device=='cuda' else None
        if observe:
            snap=observe();w.append(out/'gpu-before.jsonl',snap);w.require(snap['free_bytes']>=LIMITS['min_gpu_bytes'],'GPU free memory gate')
        request={'bundle':str(Path(a.bundle).absolute()),'bundle_sha256':w.digest(a.bundle),'config':config,'config_file_sha256':w.digest(a.config),'seed':a.seed,'resume':a.resume,'device':a.device,'validation_ids':[s['scan_id'] for s in scans if s['split']=='val'],'limits':LIMITS,'out':str(out)}
        w.save(out/'request.json',request);w.save(out/'invocation.json',{'argv':sys.argv if argv is None else argv,'binding_sha256':a.binding_sha256,'reference_release':'HOLD'})
        for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,lambda s,f:stop.__setitem__(0,True))
        with (out/'worker.log').open('xb',buffering=0) as log:
            child=w.OwnedWorker([sys.executable,'-B',str(Path(__file__).with_name('cohort_worker.py')),'--request',str(out/'request.json')],w.child_env(a.device),log);w.save(out/'owned-worker.json',child.identity)
            result=w.supervise(child,out,a.device,LIMITS,observe=observe,stop=lambda:stop[0],validate_evidence=evidence,budget_factory=lambda start,limits:EpochBudget(start,limits,out),completion_reason='DEVELOPMENT_TRAINING_TECHNICAL_COMPLETION')
        w.require(result['samples']>0,'No resource samples')
        if a.device=='cuda':w.require(any(x.get('owned_gpu_bytes',0) for x in records(out/'samples.jsonl',60000)),'No owned GPU sample')
        return 0 if result['status']=='PASS' else 1
    except BaseException as ex:
        cleanup=child.cleanup() if child else None;w.save(out/'launcher-failure.json',{'error':repr(ex),'cleanup':cleanup,'reference_release':'HOLD'});return 1
    finally:
        for sig,old in handlers.items():signal.signal(sig,old)
if __name__=='__main__':raise SystemExit(main())
