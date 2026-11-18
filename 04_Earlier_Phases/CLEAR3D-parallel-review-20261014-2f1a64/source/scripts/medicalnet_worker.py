"""Single owned worker. Explicit parent gate, no standalone training bypass."""
import argparse,ctypes,os,signal,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import user_watchdog as w
from lidc3d import medicalnet_contract as c
from watchdog_generated_worker import prohibit_processes

def execute(r):
 out=Path(r['out']);started=time.monotonic();w.save(out/'phase.json',{'name':'setup'})
 b=w.bindings(r['binding'],r['binding_sha256']);c.require(c.digest(r['checkpoint'])==c.WEIGHTS[r['depth']],'Checkpoint changed')
 # All gates precede Torch/CUDA and selected array access.
 if r['mode']=='pilot':
  from lidc3d import mixed_engineering as e
  c.disposition(c.read(r['disposition'],r['disposition_sha256']),r['binding_sha256']);c.provenance(c.read(r['provenance'],r['provenance_sha256']))
  q=c.read(r['qualification'],r['qualification_sha256']);c.require(q['status']=='PASS' and q['usable_owned_gpu_sample'],'Qualification monitoring absent');c.generated_acceptance(q['worker'],r['depth'],r['binding_sha256'])
  selection=c.read(r['selection'],c.SELECTION);e.verify_selection(selection);cfg=selection['config']
 else:
  from lidc3d.mixed_engineering import CONTRACT
  cfg=CONTRACT['config']
 import torch
 from lidc3d.provenance import seed_all,canonical_hash
 from lidc3d.medicalnet_adapter import build_transfer
 from lidc3d import medicalnet_qualification as g
 c.require(torch.cuda.device_count()==1,'Reviewed single GPU visibility required');seed_all(17)
 torch.cuda.set_per_process_memory_fraction(r['limits']['allocator_bytes']/torch.cuda.get_device_properties(0).total_memory);torch.cuda.reset_peak_memory_stats()
 ident=c.identity(r['depth'],cfg,r['binding_sha256'],c.SELECTION if r['mode']=='pilot' else c.canonical({'generated_fixture_seed':717,'labels':[1,0,1,0,1]}),r['run_id'],r.get('disposition_sha256'))
 ident.update(source_inventory_sha256=canonical_hash(b['sources']))
 w.save(out/'engineering-identity.json',ident)
 training=time.monotonic();w.save(out/'phase.json',{'name':'training'})
 if r['mode']=='generated':
  stage_start=time.monotonic();g.operator_probes('cuda',lambda v:w.append(out/'operators.jsonl',v));c.require(time.monotonic()-stage_start<=180,'Operator stage time limit')
  repeats=[]
  for i in range(2):
   t=time.monotonic();repeats.append(g.generated_repeat(r['depth'],r['checkpoint'],cfg,out/('repeat-'+str(i)),'cuda',ident,lambda v:w.append(out/('boundaries-'+str(i)+'.jsonl'),v)))
   torch.cuda.synchronize();c.require(time.monotonic()-t<=180,'Repeat stage time limit');w.append(out/'stages.jsonl',{'stage':'full_hybrid_repeat','repeat':i,'seconds':time.monotonic()-t,'status':'PASS'})
  c.require(g.exact(repeats[0],repeats[1]),'Exact repeated optimizer/model/prediction comparison failed')
  validation=time.monotonic();w.save(out/'phase.json',{'name':'validation'})
  result={'status':'PASS','stages':['pool_ties_backward','shortcut_backward','full_hybrid_repeat'],'exact_repeat':True,'updates_per_repeat':1,'skips':0,'comparison':'recursive exact tensors/dtypes/shapes, optimizer, losses, predictions, counters','phase_seconds':{'setup':training-started,'training':validation-training,'validation':time.monotonic()-validation},'examples':10}
 else:
  ident.update(parent_config_hash=canonical_hash(selection['parent_config']),engineering_policy_sha256=canonical_hash(e.RULE))
  # Accepted run_pass owns initialization, ordering, weighted optimizer, complete validation and immutable publication.
  # Transfer identity is inside every accepted checkpoint binding; base config remains exactly the accepted contract.
  result=e.run_pass(selection,cfg,out/'pass','cuda',ident,phase=lambda name:w.save(out/'phase.json',{'name':name}),model_factory=lambda model_cfg:build_transfer(model_cfg,r['depth'],r['checkpoint']))
  w.save(out/'pass/expected-identity.json',ident);result['examples']=768
 torch.cuda.synchronize();result.update(depth=r['depth'],device='cuda',source_binding_sha256=r['binding_sha256'],transfer=c.transfer(r['depth']),max_allocated=torch.cuda.max_memory_allocated(),max_reserved=torch.cuda.max_memory_reserved(),wall_seconds=time.monotonic()-started,reference_release='HOLD')
 result['examples_per_training_second']=result['examples']/result['phase_seconds']['training'];result['throughput_scope']='shared GPU engineering; generated stage includes operator/repeat overhead' if r['mode']=='generated' else 'shared GPU selected development pass'
 w.save(out/'worker-result.json',result);w.save(out/'phase.json',{'name':'complete'})

def main():
 p=argparse.ArgumentParser();p.add_argument('--gate-fd',type=int,required=True);p.add_argument('--parent-pid',type=int,required=True);p.add_argument('--request',required=True);a=p.parse_args()
 c.require(ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGKILL,0,0,0)==0,'Parent-death guard failed');c.require(os.getppid()==a.parent_pid and os.read(a.gate_fd,1)==b'1','Parent/gate mismatch');os.close(a.gate_fd);sys.addaudithook(prohibit_processes)
 r=w.load(a.request)
 try:execute(r)
 except BaseException as ex:w.save(Path(r['out'])/'worker-failure.json',{'error':repr(ex),'later_stages':'NOT_RUN','reference_release':'HOLD'});raise
if __name__=='__main__':main()
