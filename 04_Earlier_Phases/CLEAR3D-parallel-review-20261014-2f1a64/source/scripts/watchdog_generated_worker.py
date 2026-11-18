"""Fixed generated fixtures only. No LIDC reader, dataset, split, or training entry."""
import argparse, ctypes, hashlib, json, math, os, signal, sys, time
from pathlib import Path
from user_watchdog import save,append,digest,require

def prohibit_processes(event,args):
 if event in ('os.fork','os.forkpty','os.posix_spawn','subprocess.Popen','os.system','os.exec','pty.spawn'):
  raise RuntimeError('Single-process workload forbids '+event)
def run(device,out):
 started=time.monotonic()
 save(out/'phase.json',{'name':'setup'});fixture={'kind':'generated-only','seed':17,'values':[i/32 for i in range(32)],'num_workers':0,'multiprocessing':False};save(out/'generated-fixture.json',fixture)
 actual=skipped=0;torch=None
 if device=='cuda':
  import torch
  torch.set_num_threads(1);torch.manual_seed(17);torch.cuda.manual_seed_all(17);torch.use_deterministic_algorithms(True)
  require(torch.cuda.device_count()==1,'Exactly one visible GPU required')
  torch.cuda.set_per_process_memory_fraction((1024**3)/torch.cuda.get_device_properties(0).total_memory)
  torch.cuda.reset_peak_memory_stats();model=torch.nn.Linear(32,1).cuda();opt=torch.optim.SGD(model.parameters(),lr=.01)
  x=torch.tensor(fixture['values'],device='cuda').reshape(1,32);target=torch.zeros((1,1),device='cuda')
 training_started=time.monotonic()
 save(out/'phase.json',{'name':'training'})
 weight=1.0
 for i in range(20):
  if torch is None:
   loss=(weight-.25)**2;require(math.isfinite(loss),'Nonfinite generated loss');weight-=.01*2*(weight-.25)
  else:
   opt.zero_grad(set_to_none=True);loss=((model(x)-target)**2).sum();require(bool(torch.isfinite(loss)),'Nonfinite generated loss');loss.backward();opt.step();torch.cuda.synchronize();loss=float(loss.detach().cpu())
  actual+=1;append(out/'boundaries.jsonl',{'boundary':i,'loss':loss,'actual_update':1,'skipped_update':0,'scale_before':None,'scale_after':None,'precision':'FP32 generated plumbing probe; not AMP acceptance'})
  time.sleep(.02)
 validation_started=time.monotonic()
 save(out/'phase.json',{'name':'validation'})
 result={'status':'PASS','device':device,'actual_updates':actual,'skipped_updates':skipped,'fixture_sha256':digest(out/'generated-fixture.json'),'scope':'generated watchdog plumbing only; no LIDC model quality or AMP claim'}
 if torch is not None:
  with torch.no_grad():require(bool(torch.isfinite(model(x)).all()),'Nonfinite generated validation')
  torch.cuda.synchronize();result['max_allocated']=torch.cuda.max_memory_allocated();result['max_reserved']=torch.cuda.max_memory_reserved()
 result['phase_seconds']={'setup':training_started-started,'training':validation_started-training_started,'validation':time.monotonic()-validation_started}
 save(out/'worker-result.json',result);save(out/'phase.json',{'name':'complete'})
def main():
 p=argparse.ArgumentParser();p.add_argument('--gate-fd',type=int,required=True);p.add_argument('--parent-pid',type=int,required=True);p.add_argument('--device',choices=['cpu','cuda'],required=True);p.add_argument('--out',required=True);a=p.parse_args()
 libc=ctypes.CDLL(None,use_errno=True);require(libc.prctl(1,signal.SIGKILL,0,0,0)==0,'Cannot set worker parent-death signal');require(os.getppid()==a.parent_pid,'Parent changed before gate')
 require(os.read(a.gate_fd,1)==b'1','Launch gate closed');os.close(a.gate_fd);sys.addaudithook(prohibit_processes)
 try:run(a.device,Path(a.out))
 except BaseException as e:save(Path(a.out)/'worker-failure.json',{'type':type(e).__name__,'error':str(e)});raise
if __name__=='__main__':main()
