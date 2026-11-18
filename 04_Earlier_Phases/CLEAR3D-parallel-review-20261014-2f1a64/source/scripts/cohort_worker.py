"""Single owned worker; production guards remain in training.train."""
import argparse,ctypes,json,os,signal,sys
from pathlib import Path
from user_watchdog import save,require,digest
from watchdog_generated_worker import prohibit_processes

def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--gate-fd',type=int,required=True);p.add_argument('--parent-pid',type=int,required=True);a=p.parse_args()
    libc=ctypes.CDLL(None,use_errno=True);require(libc.prctl(1,signal.SIGKILL,0,0,0)==0,'Parent death gate');require(os.getppid()==a.parent_pid,'Parent changed');require(os.read(a.gate_fd,1)==b'1','Closed launch gate');os.close(a.gate_fd)
    sys.addaudithook(prohibit_processes);sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    request=json.loads(Path(a.request).read_text());out=Path(request['out'])
    try:
        require(digest(request['bundle'])==request['bundle_sha256'],'Bundle changed')
        import torch
        torch.set_num_threads(1)
        if request['device']=='cuda':
            require(torch.cuda.device_count()==1,'One visible GPU required');torch.cuda.set_per_process_memory_fraction(request['limits']['allocator_bytes']/torch.cuda.get_device_properties(0).total_memory);torch.cuda.reset_peak_memory_stats()
        from lidc3d.epoch_evidence import Observer
        from lidc3d.training import train
        observer=Observer(out);observer('setup')
        train(json.loads(Path(request['bundle']).read_text()),request['config'],out/'run',request['seed'],request['device'],request['resume'],observer=observer)
        epochs=sorted(int(p.stem.split('_')[-1]) for p in (out/'run').glob('epoch_*.json'))
        require(bool(epochs) or request['resume'],'No newly executed epochs without explicit resume')
        from lidc3d.checkpoints import load_set
        sealed=load_set(out/'run/last.pt',{});end=sealed['last']['epoch'];start=epochs[0] if epochs else end+1;del sealed
        save(out/'worker-result.json',{'status':'PASS','device':request['device'],'epochs':epochs,'start_epoch':start,'end_epoch':end,'max_allocated':torch.cuda.max_memory_allocated() if request['device']=='cuda' else 0,'max_reserved':torch.cuda.max_memory_reserved() if request['device']=='cuda' else 0,'reference_release':'HOLD'})
    except BaseException as ex:save(out/'worker-failure.json',{'type':type(ex).__name__,'error':str(ex),'reference_release':'HOLD'});raise
if __name__=='__main__':main()
