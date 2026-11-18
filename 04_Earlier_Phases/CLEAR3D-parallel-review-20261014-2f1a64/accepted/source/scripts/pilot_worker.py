"""Internal gated worker. All real entry paths revalidate pinned prerequisites."""
from __future__ import annotations
import argparse, ctypes, json, math, os
from pathlib import Path
import signal, sys, time
from pilot_supervisor import (LIMITS, append, canonical, prerequisites, read, require, save, sha, utc)

# Source-bound semantic anchors; refuse ambiguous instrumentation instead of stale line numbers.
ANCHORS={'bundle_verification':'    scans = verify_bundle(bundle)',
 'candidate_loading':'        index = CandidateIndex(', 'labeling':'        train_rows, val_rows = index.train',
 'model_setup':'        seed_all(seed)', 'training':'            weighted_loss = train_epoch(',
 'epoch_validation':'            predictions, statuses = predict(', 'epoch_metrics':'            curves = DetectionCurves(',
 'checkpoint_publication':'            publish_set(output, state, new_best, binding)', 'epoch_record':'            if new_best:',
 'best_checkpoint_load':'        final_set = load_set(', 'final_validation':'        predictions, statuses = predict(',
 'final_selection':'        curves = DetectionCurves('}
def probe_lines():
    source=(Path(__file__).resolve().parents[1]/'src/lidc3d/training.py').read_text().splitlines()
    result={}
    for phase,anchor in ANCHORS.items():
        found=[i+1 for i,line in enumerate(source) if line.startswith(anchor)]
        # epoch_record follows checkpoint publication, not earlier best-state update.
        if phase=='epoch_record':found=[i for i in found if i>next(k for k,v in result.items() if v=='checkpoint_publication')]
        require(len(found)==1,'Ambiguous training phase anchor: '+phase);result[found[0]]=phase
    counts=[i+1 for i,line in enumerate(source) if line.startswith('        if not train_rows or not all(')]
    require(len(counts)==1,'Ambiguous count anchor');return result,counts[0]
LINES,COUNTS_LINE=probe_lines()

class Measurements:
    def __init__(self,out,torch=None):
        self.out=Path(out);self.torch=torch;self.phase=None;self.started=None;self.run_started=None
        self.frames={};self.counts=None
    def gpu(self):
        if self.torch is None:return {'kind':'synthetic CPU test; no hardware observation'}
        t=self.torch
        t.cuda.synchronize()
        return {'allocated':t.cuda.memory_allocated(),'reserved':t.cuda.memory_reserved(),
                'peak_allocated':t.cuda.max_memory_allocated(),'peak_reserved':t.cuda.max_memory_reserved()}
    def phase_to(self,name):
        if name==self.phase:return
        now=time.monotonic()
        if self.phase is not None:
            append(self.out/'phases.jsonl',{'event':'end','phase':self.phase,'utc':utc(),'monotonic':now,
                   'seconds':now-self.started,'torch_memory':self.gpu()})
        self.phase=name;self.started=time.monotonic()
        if name=='training' and self.run_started is None:self.run_started=self.started
        value={'name':name,'started':self.started,'run_started':self.run_started,'utc':utc()}
        append(self.out/'phases.jsonl',{'event':'begin',**value})
        temporary=self.out/'phase.next'
        with temporary.open('w') as f:
            json.dump(value,f);f.flush();os.fsync(f.fileno())
        os.replace(temporary,self.out/'phase.json')
    def tracer(self,frame,event,arg):
        if frame.f_code is not self.code:return None
        if event=='line':
            if frame.f_lineno==COUNTS_LINE:
                receipt=frame.f_locals['index'].receipt
                self.counts={'training_nonignored':receipt['roles']['train'],
                    'training_positive':receipt['training_labels']['1'],
                    'training_negative':receipt['training_labels']['0'],
                    'training_ignored':receipt['training_labels']['-1'],
                    'validation_candidates':receipt['roles']['val'],
                    'frozen_candidates':sum(receipt['per_scan'].values()),
                    'training_scans':len(frame.f_locals['train_scans']),
                    'validation_scans':len(frame.f_locals['val_scans'])}
                save(self.out/'cohort-counts.json',self.counts)
                require(self.counts['training_scans']==708 and self.counts['validation_scans']==89,'Pilot cohort counts changed')
            if frame.f_lineno in LINES:self.phase_to(LINES[frame.f_lineno])
        return self.tracer

def reconcile(path,counts):
    n=counts['training_nonignored'];expected=math.ceil(math.ceil(n/2)/8)
    boundaries=updates=skips=scales=examples=updated_examples=0;last=None
    with Path(path).open() as f:
        for line in f:
            event=json.loads(line)
            require(event.get('kind')=='accumulation_boundary','Unexpected optimization failure event')
            require(event['status'] in ('UPDATED','SKIPPED_AMP_NONFINITE'),'Failed update boundary')
            boundaries+=1;updates+=event['optimizer_updates'];skips+=event['status']=='SKIPPED_AMP_NONFINITE'
            require(event['optimizer_updates']==(1 if event['status']=='UPDATED' else 0),'Update dispatch mismatch')
            require(event['scaler_update_calls']==1,'Missing scaler update')
            scales+=event['scale_before']!=event['scale_after']
            samples=sum(x['samples'] for x in event['microbatches']);examples+=samples
            updated_examples+=samples if event['status']=='UPDATED' else 0
            last=event['counters']
    require(boundaries==expected and examples==n and updates>0,'Boundary/epoch coverage mismatch')
    require(last is not None and last['boundaries']==boundaries and last['optimizer_updates']==updates
            and last['skipped_updates']==skips and last['scale_changes']==scales
            and last['scaler_update_calls']==boundaries,'Counters disagree with complete diagnostics')
    return {'boundaries':boundaries,'actual_optimizer_updates':updates,'skipped_updates':skips,
            'scale_changes':scales,'processed_examples':examples,'update_contributing_examples':updated_examples,
            'skipped_examples':examples-updated_examples,'checkpoint_update_state':last}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('gate-fd','parent-pid','contract','contract-sha256','request','request-sha256','out'):p.add_argument('--'+k,required=True)
    a=p.parse_args();fd=int(a.gate_fd)
    # Kernel parent-death signal guards the direct child if its supervisor dies.
    libc=ctypes.CDLL(None,use_errno=True)
    require(libc.prctl(1,signal.SIGKILL,0,0,0)==0,'Parent-death guard unavailable')
    require(os.getppid()==int(a.parent_pid),'Supervisor disappeared before worker gate')
    require(os.read(fd,1)==b'1','No supervisor launch gate');os.close(fd)
    out=Path(a.out);measure=Measurements(out);measure.phase_to('preflight')
    try:
        require(sha(a.contract)==a.contract_sha256 and sha(a.request)==a.request_sha256,'Worker input race/binding mismatch')
        contract,request=read(a.contract),read(a.request);work=Path(__file__).resolve().parents[1]
        bundle,cfg=prerequisites(request,contract,work)
        from lidc3d.derivative import require_derivative_review
        prep=Path(contract['preparation'])
        require_derivative_review(prep.parent,read(prep),read(Path(request['bundle']['path']).parent/'reference_review.json'))
        import torch
        from lidc3d import training
        from lidc3d.checkpoints import code_identity,load_set
        require(code_identity()['sha256']==contract['code_identity'],'Wrong imported source identity')
        require(Path(training.__file__).resolve()==work/'src/lidc3d/training.py','Foreign training module')
        require(sha(training.__file__)==contract['sources']['src/lidc3d/training.py'],'Trace source changed')
        require(torch.cuda.device_count()==1,'Single selected GPU visibility required')
        total=torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(LIMITS['allocator_bytes']/total,0)
        torch.cuda.reset_peak_memory_stats();measure.torch=torch;measure.code=training.train.__code__
        save(out/'worker-bindings.json',{'code_identity':code_identity(),'config_sha256':canonical(cfg),
            'bundle_sha256':canonical(bundle),'seed':17,'allocator_bytes':LIMITS['allocator_bytes'],
            'environment':{k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','LD_PRELOAD']}})
        require(sys.gettrace() is None,'An existing trace hook would invalidate instrumentation')
        sys.settrace(measure.tracer)
        try:training.train(bundle,cfg,out/'training',17,device='cuda:0')
        finally:sys.settrace(None)
        result=reconcile(out/'training/optimization_boundaries.jsonl',measure.counts)
        identity=read(out/'training/identity.json');state=load_set(out/'training/last.pt',identity['checkpoint_binding'])
        require(state['last']['update_state']==result['checkpoint_update_state'],'Checkpoint counter mismatch')
        require(read(out/'training/selection.json')['status']=='TRAINING_COMPLETE','Training completion absent')
        result.update(status='PASS',torch_memory=measure.gpu(),counts=measure.counts,
                      seconds_since_training_start=time.monotonic()-measure.run_started,
                      throughput_scope='shared GPU; includes final validation/checkpoint audit')
        result['processed_examples_per_second']=result['processed_examples']/result['seconds_since_training_start']
        result['actual_updates_per_second']=result['actual_optimizer_updates']/result['seconds_since_training_start']
        save(out/'worker-result.json',result);measure.phase_to('complete')
        return 0
    except BaseException as error:
        save(out/'worker-failure.json',{'status':'FAIL','type':type(error).__name__,'reason':str(error),'phase':measure.phase,'utc':utc()})
        try:measure.phase_to('failed')
        except BaseException:pass
        raise

if __name__=='__main__':raise SystemExit(main())
