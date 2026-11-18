"""Single owned engineering worker; no direct ungated execution."""
import argparse,ctypes,os,signal,sys,uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import user_watchdog as w
from watchdog_generated_worker import prohibit_processes
from lidc3d import mixed_engineering as e
from lidc3d.provenance import canonical_hash

def main():
 p=argparse.ArgumentParser();p.add_argument('--gate-fd',type=int,required=True);p.add_argument('--parent-pid',type=int,required=True);p.add_argument('--request',required=True);a=p.parse_args()
 require=w.require;require(ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGKILL,0,0,0)==0,'Parent-death guard failed');require(os.getppid()==a.parent_pid,'Parent changed');require(os.read(a.gate_fd,1)==b'1','Gate not released');os.close(a.gate_fd);sys.addaudithook(prohibit_processes)
 r=w.load(a.request);out=Path(r['out']);w.save(out/'phase.json',{'name':'setup'})
 try:
  b=w.bindings(r['binding'],r['binding_sha256']);e.scoped_disposition(r['disposition'],r['disposition_sha256'],r['selection_sha256'],r['binding_sha256'])
  require(w.digest(r['selection'])==r['selection_sha256'],'Selection changed');selection=e.verify_selection(e.bounded_json(r['selection']))
  require(selection['preparation_sha256']==e.PREPARATION,'Wrong derivative')
  import torch;from lidc3d.optimization import POLICY
  require(torch.cuda.device_count()==1,'Exactly reviewed GPU required');torch.cuda.set_per_process_memory_fraction(e.LIMITS['allocator_bytes']/torch.cuda.get_device_properties(0).total_memory);torch.cuda.reset_peak_memory_stats()
  binding={'parent_config_hash':canonical_hash(selection['parent_config']),'experiment_id':e.RULE['experiment_id'],'engineering_policy_sha256':canonical_hash(e.RULE),'run_id':r['run_id'],'config_hash':canonical_hash(selection['config']),'selection_sha256':r['selection_sha256'],'preparation_sha256':e.PREPARATION,'source_binding_sha256':r['binding_sha256'],'source_inventory_sha256':canonical_hash(b['sources']),'disposition_sha256':r['disposition_sha256'],'optimization_policy_sha256':canonical_hash(POLICY),'seed':17,'device':'cuda','amp':bool(selection['config']['training']['amp'])}
  w.save(out/'engineering-identity.json',binding)
  e.run_pass(selection,selection['config'],out/'pass','cuda',binding,phase=lambda name:w.save(out/'phase.json',{'name':name}))
 except BaseException as ex:w.save(out/'worker-failure.json',{'error':repr(ex),'reference_release':'HOLD'});raise
if __name__=='__main__':main()
