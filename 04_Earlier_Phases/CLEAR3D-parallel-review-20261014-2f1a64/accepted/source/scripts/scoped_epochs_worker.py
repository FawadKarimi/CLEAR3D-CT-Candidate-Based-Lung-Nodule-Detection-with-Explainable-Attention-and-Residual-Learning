"""Owned, gated worker. No standalone or production approval bypass."""
import argparse,ctypes,os,signal,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import user_watchdog as w
from watchdog_generated_worker import prohibit_processes
from lidc3d import scoped_epochs_contract as c

def execute(r):
    out=Path(r['out']);started=time.monotonic();w.save(out/'phase.json',{'name':'setup'})
    b=w.bindings(r['binding'],r['binding_sha256'])
    spec=b['epoch_configs'][str(r['depth'])];cfg=c.read(spec['path'],spec['sha256']);c.check_config(cfg)
    m=cfg['medicalnet'];c.require(c.digest(m['checkpoint'])==c.base.WEIGHTS[r['depth']],'Weights changed')
    c.base.provenance(c.read(m['provenance'],m['provenance_sha256']))
    if r['mode']=='research':
        c.disposition(c.read(r['disposition'],r['disposition_sha256']),r['binding_sha256'],
                      {d:v['sha256'] for d,v in b['epoch_configs'].items()})
        from scoped_epochs_watchdog import prerequisites
        prerequisites(r)
        from lidc3d import mixed_engineering as e
        selection=c.read(b['epoch_selection'],c.base.SELECTION);e.verify_selection(selection)
    import torch
    from lidc3d import scoped_epochs as engine
    from lidc3d.medicalnet_finetuning import build
    identity=c.identity(cfg,r['binding_sha256'],r['run_id'],r['mode'],r.get('disposition_sha256'))
    w.save(out/'identity.json',identity)
    if r['leg']=='compare':
        result=engine.compare(Path(r['full'])/'state',Path(r['resumed'])/'state',identity)
        result.update(device='cpu',mode=r['mode'],depth=r['depth'],phase_seconds={'setup':time.monotonic()-started,'training':0,'validation':0},max_allocated=0)
    else:
        c.require(torch.cuda.device_count()==1,'Single selected GPU required')
        from lidc3d.provenance import seed_all
        seed_all(17);torch.cuda.set_per_process_memory_fraction(r['limits']['allocator_bytes']/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        if r['mode']=='generated':
            rows,scans,val=engine.generated_data();factory=engine.GeneratedDataset;validator=engine.generated_validate
            sampling=c.GENERATED_SAMPLING;epochs=2
        else:
            rows,scans,val,factory=engine.real_data(selection);validator=engine.real_validate;sampling=c.SAMPLING;epochs=3
        setup=time.monotonic()-started;w.save(out/'phase.json',{'name':'training'})
        def epoch_phase(v):
            event={**v,'monotonic':time.monotonic()}
            w.save(out/'epoch-phase.json',event);w.append(out/'epoch-phases.jsonl',event)
        result=engine.fit(cfg,identity,out/'state','cuda',rows,scans,val,factory,validator,
            lambda _:build(cfg,17),epochs=epochs,sampling=sampling,
            resume=Path(r['resume'])/'state' if r.get('resume') else None,
            stop_after=0 if r['leg']=='interrupted' else None,
            phase=epoch_phase)
        torch.cuda.synchronize();result.update(device='cuda',max_allocated=torch.cuda.max_memory_allocated(),max_reserved=torch.cuda.max_memory_reserved())
        result['phase_seconds']['setup']+=setup
    result.update(mode=r['mode'],depth=r['depth'],source_binding_sha256=r['binding_sha256'],identity=identity,
                  wall_seconds=time.monotonic()-started,reference_release='HOLD')
    w.save(out/'worker-result.json',result);w.save(out/'phase.json',{'name':'complete'})

def main():
    p=argparse.ArgumentParser();p.add_argument('--gate-fd',type=int,required=True);p.add_argument('--parent-pid',type=int,required=True);p.add_argument('--request',required=True);a=p.parse_args()
    c.require(ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGKILL,0,0,0)==0,'Parent-death guard failed')
    c.require(os.getppid()==a.parent_pid and os.read(a.gate_fd,1)==b'1','Parent gate mismatch');os.close(a.gate_fd)
    sys.addaudithook(prohibit_processes);r=w.load(a.request)
    try:execute(r)
    except BaseException as ex:
        w.save(Path(r['out'])/'worker-failure.json',{'error':repr(ex),'later_stages':'NOT_RUN','reference_release':'HOLD'});raise
if __name__=='__main__':main()
