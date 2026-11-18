"""Current-split fit-only adaptation. Default is plan-only, no Torch import.

No production source edits, resume, validation, policy adoption or automatic retry.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

SCOPE = 'clear3d-fixed16-current-fit-long-v1'
POLICY = dict(scope=SCOPE, depth=18, seed=17, positives=4, negatives=12,
              updates=120, microbatch=2, accumulation=8, precision='FP32',
              augmentation=False, order='positive-first-fixed',
              learning_rate='constant-parent-epoch0-group-rates',
              validation=False, resume=False, reference_release='HOLD',
              cap4000_policy='NOT_ADOPTED')
LIMITS = dict(total_seconds=900, setup_seconds=120, run_seconds=900,
              validation_seconds=60, rss_bytes=8*1024**3, allocator_bytes=4*1024**3,
              gpu_bytes=6*1024**3, min_ram_bytes=32*1024**3, reserve_ram_bytes=8*1024**3,
              min_gpu_bytes=10*1024**3, min_disk_bytes=20*1024**3, sample_seconds=1.)

def require(ok, msg):
    if not ok: raise ValueError(msg)

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()

def read(path, expected=None):
    p = Path(path)
    require(p.stat().st_size <= 8*1024**2, 'JSON admission')
    if expected: require(digest(p) == expected, 'Hash mismatch: '+str(p))
    return json.loads(p.read_text())

def choose(indices):
    require(len(indices) == 960 and all(type(i) is int and i >= 0 for i in indices), 'Draw admission')
    positives = list(dict.fromkeys(indices[:240]))[:4]
    negatives = list(dict.fromkeys(indices[240:]))[:12]
    require(len(positives)==4 and len(negatives)==12 and len(set(positives+negatives))==16, 'Coverage unavailable')
    return positives+negatives

def adoption(value, binding_hash):
    require(value.get('status')=='ADOPTED' and value.get('scope')==SCOPE, 'Separate adoption required')
    require(value.get('binding_sha256')==binding_hash and value.get('policy')==POLICY and value.get('limits')==LIMITS, 'Disposition binding')
    require(value.get('reference_release')=='HOLD' and value.get('cap4000_policy')=='NOT_ADOPTED', 'Status changed')
    require(all(value.get(k) is True for k in ('label_guided','prior_exposure_uncertain','pretraining_overlap_uncertain','sampled_monitoring','no_scientific_approval','train_only')), 'Acknowledgements absent')
    require(all(isinstance(value.get(k),str) and value[k].strip() for k in ('supplied_by','instruction','date')), 'Attribution absent')
    expected={'reference_origin','excluded_reference_outliers','geometry_and_source_scope','protocol_evaluation_disclosure','prior_test_access','independent_reference_review'}
    require(set(value.get('holds',[]))==expected, 'Six holds required')

def gates(request):
    b=read(request['binding'],request['binding_sha256']);root=Path(b['root'])
    require(b['policy']==POLICY and b['limits']==LIMITS, 'Policy changed')
    require(set(b['new_sources']) == {'D-long/fixed_batch_fit.py','D-long/fit_selection.py'}, 'Both fit-only code files must be bound')
    for rel,h in b['new_sources'].items(): require(digest(root/rel)==h,'New source mismatch')
    require(Path(__file__).resolve()==(root/'D-long/fixed_batch_fit.py').resolve(),'Runner path mismatch')
    from fit_selection import select_fit
    require(Path(select_fit.__code__.co_filename).resolve()==(root/'D-long/fit_selection.py').resolve(),'Selector path mismatch')
    work=root/'source';sys.path[:0]=[str(work/'src'),str(work/'scripts')]
    import user_watchdog as w
    from lidc3d import diagnostic_training as parent
    require(Path(w.__file__).resolve()==work/'scripts/user_watchdog.py','Import location')
    old=w.bindings(b['parent_binding'],b['parent_binding_sha256'])
    cfg=read(b['configuration']['path'],b['configuration']['sha256']);parent.check_config(cfg)
    require(cfg['medicalnet']['depth']==18, 'ResNet18 only')
    require(digest(cfg['medicalnet']['checkpoint'])==parent.base.WEIGHTS[18],'Weights')
    parent.base.provenance(read(cfg['medicalnet']['provenance'],cfg['medicalnet']['provenance_sha256']))
    for key in ('preparation','sidecar','parent_selection','label_selection','draw0'):
        spec=old['diagnostic_data']['inputs'][key]
        require(Path(spec['path']).stat().st_size==spec['bytes'] and digest(spec['path'])==spec['sha256'],'Input '+key)
    plan=read(old['diagnostic_data']['inputs']['draw0']['path'])
    split_spec=b['current_split']
    require(type(split_spec['bytes']) is int and Path(split_spec['path']).stat().st_size==split_spec['bytes'],'Current split size changed')
    split=read(split_spec['path'],split_spec['sha256'])
    selection=read(old['diagnostic_data']['inputs']['parent_selection']['path'])
    from lidc3d.cap4000_labels import LabelRows
    rows=LabelRows(old['diagnostic_data']['inputs']['sidecar']['path'],old['diagnostic_data']['inputs']['sidecar']['sha256'])
    try: selected=select_fit(plan['indices'],rows,selection['entries'],split)
    finally: rows.close()
    require(selected==b['indices'],'Current fit-only indices changed')
    adoption(read(request['disposition'],request['disposition_sha256']),request['binding_sha256'])
    require(request.get('acknowledge_shared_gpu_permission') and request.get('acknowledge_sampled_risk'),'Shared/sampled acknowledgement')
    return b,old,cfg,w

def train_fixed(model, batches, cfg, device, out, identity, checkpoint=True):
    """Generated CPU seam; production entry supplies exactly 16 pinned patches."""
    import torch
    from lidc3d.training import parameter_groups,epoch_lr
    from lidc3d.optimization import train_epoch,new_update_state,append_diagnostic
    from lidc3d.diagnostics import Diagnostics
    from lidc3d.engineering import publish_checkpoint,load_checkpoint
    require(len(batches)==8 and sum(len(z['label']) for z in batches)==16,'Fixed batch size')
    require([int(y) for z in batches for y in z['label']]==[1]*4+[0]*12,'Fixed labels')
    groups=parameter_groups(model,cfg['training'])
    for group in groups:group['lr']=epoch_lr(group['lr'],0,cfg['training'])
    opt=torch.optim.AdamW(groups)
    scaler=torch.amp.GradScaler('cuda',enabled=False);state=new_update_state()
    weights=torch.tensor(cfg['training']['class_weights'],device=device,dtype=torch.float32)
    require(weights.tolist()==[1.,3.] and cfg['training']['clip_grad_norm']==1,'Objective/clipping')
    def evaluate():
        model.eval();result=[]
        with torch.no_grad():
            for batch in batches:
                logits=model(batch['image'].to(device));require(bool(torch.isfinite(logits).all()),'Nonfinite logits')
                losses=torch.nn.functional.cross_entropy(logits,batch['label'].to(device),weight=weights,reduction='none')
                require(bool(torch.isfinite(losses).all()),'Nonfinite diagnostic loss')
                result.extend(dict(label=int(y),weighted_loss=float(loss),probability=float(prob)) for y,loss,prob in zip(batch['label'],losses.cpu(),logits.softmax(1)[:,1].cpu()))
        return {'rows':result,'weighted_mean':sum(z['weighted_loss'] for z in result)/24,'scope':'same training fixtures only'}
    def snapshot(stage):
        return dict(stage=stage,model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),update_state=state,config=cfg,selection_sha256=identity['selection_sha256'],torch_rng=torch.get_rng_state(),scheduler={'rule':POLICY['learning_rate']},validation_complete=False)
    before=evaluate()
    if checkpoint: publish_checkpoint(out,snapshot('INITIALIZED'),identity)
    diagnostics=Diagnostics(out/'training-diagnostics.jsonl')
    boundaries=[]
    def emit(v):
        boundaries.append(v);append_diagnostic(out/'boundaries.jsonl',v)
    for step in range(POLICY['updates']):
        model.train()
        loss=train_epoch(model,batches,opt,scaler,weights,torch.device(device),False,8,1,state,emit,step,diagnostics=diagnostics)
        require(state['optimizer_updates']==step+1 and state['skipped_updates']==0,'Update mismatch')
        require(boundaries[-1]['norm_before_clip']>0 and boundaries[-1]['denominator']==24,'Finite nonzero normalized gradient')
    after=evaluate()
    require(all(float(z['step'])==120 for z in opt.state.values()),'AdamW step mismatch')
    if checkpoint:
        publish_checkpoint(out,snapshot('TRAINED'),identity)
        load_checkpoint(out,identity) # CPU, hash-bound, no resume.
    return dict(status='PASS',updates=120,skips=0,draws=1920,before=before,after=after,
                memorization_indicator=after['weighted_mean']<=.8*before['weighted_mean'],
                interpretation='20% reduction is a descriptive prespecified indicator, not quality acceptance; absence is inconclusive at 120 updates.')

def worker(request):
    start=time.monotonic();b,old,cfg,w=gates(request);out=Path(request['out'])
    w.save(out/'phase.json',{'name':'setup'})
    import torch
    from lidc3d.provenance import seed_all,canonical_hash
    from lidc3d.cap4000_labels import LabelRows
    from lidc3d.data import CandidateDataset
    from lidc3d.engineering import prepare_arrays
    from lidc3d.medicalnet_finetuning import build
    require(torch.cuda.device_count()==1,'Selected GPU only');seed_all(17)
    torch.cuda.set_per_process_memory_fraction(LIMITS['allocator_bytes']/torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats()
    inputs=old['diagnostic_data']['inputs'];selection=read(inputs['parent_selection']['path'])
    with_rows=LabelRows(inputs['sidecar']['path'],inputs['sidecar']['sha256'])
    try: rows=[with_rows[i] for i in b['indices']]
    finally: with_rows.close()
    require([z['label'] for z in rows]==[1]*4+[0]*12 and len({z['id'] for z in rows})==16,'Identity/label drift')
    scans={z['scan']['scan_id']:z for z in selection['entries'] if z['scan']['split']=='train'}
    require(all(z['scan_id'] in scans for z in rows),'Training-only boundary')
    entries=[scans[k] for k in sorted({z['scan_id'] for z in rows})]
    train_patients={z['scan']['patient_id'] for z in entries}
    require(not train_patients & {z['scan']['patient_id'] for z in selection['entries'] if z['scan']['split']=='val'},'Patient separation')
    w.save(out/'row-receipt.json',{'indices':b['indices'],'rows':rows,'payload_hashes':[canonical_hash(z) for z in rows],'scan_ids':[z['scan']['scan_id'] for z in entries]})
    prepare_arrays({**selection,'entries':entries})
    ds=CandidateDataset([z['scan'] for z in entries],rows,48,False,17)
    cached=[ds[i] for i in range(16)]
    batches=[{'image':torch.stack([z['image'] for z in cached[i:i+2]]),'label':torch.tensor([z['label'] for z in cached[i:i+2]])} for i in range(0,16,2)]
    # Bind cached values; only generated or separately authorized research worker reaches here.
    w.save(out/'cached-patches.json',{'sha256':[hashlib.sha256(z['image'].numpy().tobytes()).hexdigest() for z in cached],'augment':False})
    model=build(cfg,17).to('cuda');identity={'scope':SCOPE,'binding_sha256':request['binding_sha256'],'disposition_sha256':request['disposition_sha256'],'selection_sha256':canonical_hash(b['indices']),'config_hash':canonical_hash(cfg),'run_output':str(out),'reference_release':'HOLD'}
    state=out/'state';state.mkdir();w.save(out/'identity.json',identity)
    setup=time.monotonic()-start;w.save(out/'phase.json',{'name':'training'})
    result=train_fixed(model,batches,cfg,'cuda',state,identity)
    torch.cuda.synchronize();result.update(setup_seconds=setup,wall_seconds=time.monotonic()-start,max_allocated=torch.cuda.max_memory_allocated(),max_reserved=torch.cuda.max_memory_reserved(),identity=identity)
    require(result['max_allocated']<=LIMITS['allocator_bytes'],'Allocator limit')
    result['compute_and_checkpoint_seconds']=result['wall_seconds']
    gates(request);result['wall_seconds']=time.monotonic()-start
    w.save(out/'worker-result.json',result);w.save(out/'phase.json',{'name':'complete'})

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--worker',action='store_true')
    for k in ('binding','binding-sha256','disposition','disposition-sha256','out','request'):p.add_argument('--'+k)
    p.add_argument('--gate-fd',type=int);p.add_argument('--parent-pid',type=int)
    p.add_argument('--acknowledge-shared-gpu-permission',action='store_true');p.add_argument('--acknowledge-sampled-risk',action='store_true');a=p.parse_args(argv)
    if not a.execute and not a.worker: print(json.dumps({'status':'PLAN_ONLY','policy':POLICY,'limits':LIMITS,'torch_imported':'torch' in sys.modules}));return 0
    if a.worker:
        import ctypes,signal
        require(a.gate_fd is not None and a.parent_pid is not None,'Owned worker gate')
        require(ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGKILL,0,0,0)==0,'Parent-death guard')
        require(os.getppid()==a.parent_pid and os.read(a.gate_fd,1)==b'1','Parent gate');os.close(a.gate_fd)
        request=read(a.request);gates(request)
        from watchdog_generated_worker import prohibit_processes
        sys.addaudithook(prohibit_processes)
        try: worker(request)
        except BaseException as ex:
            (Path(request['out'])/'worker-failure.json').write_text(json.dumps({'error':repr(ex),'no_retry':True}));raise
        return 0
    request=vars(a);b,old,cfg,w=gates(request);out=Path(a.out)
    require(str(out)==b['output'] and out.parent.is_dir() and not out.exists() and not any(z.is_symlink() for z in [out,*out.parents]),'Fresh bound output')
    require(w.available_ram()>=LIMITS['min_ram_bytes'] and w.disk_free(out.parent)>=LIMITS['min_disk_bytes'],'Resource headroom')
    gpu=w.reviewed.gpu_snapshot();require(gpu['free_bytes']>=LIMITS['min_gpu_bytes'],'GPU headroom')
    out.mkdir();w.save(out/'request.json',request);w.append(out/'gpu-before.jsonl',gpu)
    import signal
    stop=[False];handlers={s:signal.signal(s,lambda *args:stop.__setitem__(0,True)) for s in (signal.SIGINT,signal.SIGTERM)}
    def validate(path,device,limits):
        v=read(path/'worker-result.json');require(v['status']=='PASS' and v['updates']==120 and v['skips']==0 and v['draws']==1920,'Incomplete worker')
        require(read(path/'phase.json')['name']=='complete' and v['wall_seconds']<=900 and v['setup_seconds']<=120 and v['max_allocated']<=limits['allocator_bytes'],'Worker limits')
        import math
        rows=[]
        with (path/'state/boundaries.jsonl').open() as f:
            for line in f:
                require(len(rows)<120 and len(line)<=1048576,'Boundary admission');rows.append(json.loads(line))
        require(len(rows)==120 and all(z['status']=='UPDATED' and z['denominator']==24 and z['optimizer_updates']==1 and z['counters']['optimizer_updates']==i+1 and z['counters']['skipped_updates']==0 and math.isfinite(z['norm_before_clip']) and z['norm_before_clip']>0 for i,z in enumerate(rows)),'Boundary reconciliation')
        counts={0:0,1:0};events=0
        with (path/'state/training-diagnostics.jsonl').open() as f:
            for n,line in enumerate(f):
                require(n<20000 and len(line)<=1048576,'Diagnostic admission');z=json.loads(line)
                if z['kind']=='class_losses':
                    events+=1
                    for label in (0,1):
                        q=z['classes'][str(label)];counts[label]+=q['count'];require(math.isfinite(q['weighted_loss_sum']),'Finite class loss')
        require(events==960 and counts=={0:1440,1:480},'Consumption mismatch')
        from lidc3d.engineering import load_checkpoint
        ck=load_checkpoint(path/'state',v['identity'])
        require(ck['state']['stage']=='TRAINED' and ck['state']['update_state']['optimizer_updates']==120 and ck['state']['validation_complete'] is False,'Checkpoint counters/scope')
        require(ck['record']['parent_commit'] is not None,'Missing initial ancestry')
        parents=[p for p in (path/'state/engineering-checkpoints').glob('*/commit.json') if digest(p)==ck['record']['parent_commit']]
        require(len(parents)==1,'Initial commit identity')
        initial=read(parents[0]);require(initial['stage']=='INITIALIZED' and initial['binding']==v['identity'] and initial['parent_commit'] is None and digest(parents[0].parent/'state.pt')==initial['state_sha256'],'Initial ancestry integrity')
        v['independent_counts']={'class_loss_events':events,'labels':counts,'boundaries':len(rows),'checkpoint_commit_sha256':ck['commit_sha256']}
        return v
    child=None
    try:
        with (out/'worker.log').open('xb',buffering=0) as log:
            child=w.OwnedWorker([sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--request',str(out/'request.json')],w.child_env('cuda'),log)
            w.save(out/'owned-worker.json',child.identity)
            result=w.supervise(child,out,'cuda',LIMITS,observe=w.reviewed.gpu_snapshot,stop=lambda:stop[0],validate_evidence=validate,completion_reason='TRAIN_ONLY_DIAGNOSTIC')
        usable=(out/'samples.jsonl').exists() and any(json.loads(z).get('owned_gpu_bytes',0)>0 for z in (out/'samples.jsonl').read_text().splitlines())
        result['usable_gpu_sample']=usable
        if not usable:result['status']='FAIL';result['monitoring_limitation']='No usable owned GPU sample'
        w.save(out/'acceptance.json',result);return 0 if result['status']=='PASS' else 1
    except BaseException as error:
        result={'status':'FAIL','error':repr(error),'reference_release':'HOLD'}
        if child is not None:result['cleanup']=child.cleanup()
        w.save(out/'acceptance.json',result);return 1
    finally:
        if child is not None and child.proc.poll() is None:child.cleanup()
        for s,h in handlers.items():signal.signal(s,h)

if __name__=='__main__':raise SystemExit(main())
