"""Plan-only default. Separate scoped disposition required; never freeze or approve."""
import argparse,json,os,signal,sys,time,uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import user_watchdog as w
from lidc3d import engineering as e

def engineering_evidence(out,device,limits):
 r=e.bounded_json(out/'pass/engineering-result.json');w.require(r['status']=='PASS' and r['scope']=='six-scan engineering only; NOT full-cohort validation','Incomplete engineering pass')
 w.require(w.load(out/'phase.json')['name']=='complete','Missing complete phase')
 from lidc3d.optimization import validate_update_state
 state=validate_update_state(r['update_state']);w.require(state['optimizer_updates']>0,'Zero optimizer updates')
 events=[v for _,v in e.rows(out/'pass/boundaries.jsonl',32)];boundaries=[v for v in events if v['kind']=='accumulation_boundary']
 w.require(len(boundaries)==state['boundaries'] and sum(v['optimizer_updates'] for v in boundaries)==state['optimizer_updates'] and sum(v['status']=='SKIPPED_AMP_NONFINITE' for v in boundaries)==state['skipped_updates'],'Boundary accounting mismatch')
 w.require(all(v['status'] in ('UPDATED','SKIPPED_AMP_NONFINITE') for v in boundaries),'Failed boundary')
 v=r['validation_scan_status'];w.require(len(v)==2 and len({s['scan_id'] for s in v})==2 and all(s['status']=='ok' and s['candidates']==s['predictions'] for s in v),'Incomplete two-scan validation')
 durations=r['phase_seconds'];w.require(durations['setup']<=limits['setup_seconds'] and durations['training']+durations['validation']<=limits['run_seconds'] and durations['validation']<=limits['validation_seconds'],'Worker phase limit exceeded')
 w.require(r['max_allocated']<=limits['allocator_bytes'],'Allocator limit exceeded')
 # Hashes/pointer checked here without allocating checkpoint tensors in supervisor.
 p=e.bounded_json(out/'pass/engineering-commit.json');w.require(p['status']=='COMMITTED','Checkpoint not committed');g=out/'pass'/p['generation'];w.require(g.resolve().parent==(out/'pass/engineering-checkpoints').resolve(),'Checkpoint escapes run');w.require(w.digest(g/'commit.json')==p['commit_sha256']==r['checkpoint_commit_sha256'],'Commit hash changed');c=e.bounded_json(g/'commit.json');w.require(w.digest(g/'state.pt')==c['state_sha256'],'Checkpoint bytes changed');return r

def main(argv=None):
 p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group();m.add_argument('--select',action='store_true');m.add_argument('--execute',action='store_true');p.add_argument('--binding');p.add_argument('--binding-sha256');p.add_argument('--prepared');p.add_argument('--selection');p.add_argument('--selection-sha256');p.add_argument('--disposition');p.add_argument('--disposition-sha256');p.add_argument('--out');p.add_argument('--acknowledge-development-metadata',action='store_true');p.add_argument('--acknowledge-sampled-risk',action='store_true');p.add_argument('--acknowledge-shared-gpu-permission',action='store_true');a=p.parse_args(argv)
 if not a.select and not a.execute:
  print(json.dumps({'status':'PLAN_ONLY','rule':e.RULE,'limits':e.LIMITS,'reference_release':'HOLD','requires':'external scoped exploratory disposition; immutable selection review before pixel access','no_real_execution_authorized':True}));return 0
 w.require(a.binding and a.binding_sha256 and a.out,'Binding and fresh output required');b=w.bindings(a.binding,a.binding_sha256);out=Path(a.out).absolute();root=Path(__file__).resolve().parents[2]
 w.require(out.parent.resolve()==root/'runs' and not out.exists() and not any(p.is_symlink() for p in [out,*out.parents]),'Fresh isolated runs output required')
 if a.select:
  w.require(a.acknowledge_development_metadata and a.prepared,'Explicit development metadata scope required');start=time.monotonic()
  def deadline():w.require(time.monotonic()-start<e.LIMITS['setup_seconds'],'Selection setup deadline')
  selection=e.make_selection(a.prepared,e.PREPARATION,deadline);out.mkdir();w.save(out/'selection.json',selection);w.save(out/'selection-receipt.json',{'sha256':w.digest(out/'selection.json'),'source_binding_sha256':a.binding_sha256,'status':'REVIEW_REQUIRED_BEFORE_PIXEL_ACCESS','reference_release':'HOLD'});return 0
 w.require(a.acknowledge_sampled_risk and a.acknowledge_shared_gpu_permission,'Explicit shared-use and sampled-risk acknowledgements required')
 w.require(a.selection and a.selection_sha256 and a.disposition and a.disposition_sha256,'Scoped disposition and reviewed selection required')
 e.scoped_disposition(a.disposition,a.disposition_sha256,a.selection_sha256,a.binding_sha256)
 w.require(w.digest(a.selection)==a.selection_sha256,'Selection changed');selection=e.bounded_json(a.selection);w.require(selection['preparation_sha256']==e.PREPARATION and selection['rule']==e.RULE,'Wrong selection scope')
 w.require(out!=Path(selection['root']) and Path(selection['root']) not in out.parents,'Output overlaps preparation')
 w.require(w.available_ram()>=e.LIMITS['min_ram_bytes'] and w.disk_free(root)>=e.LIMITS['min_disk_bytes'],'Insufficient RAM/disk headroom')
 out.mkdir();child=None;handlers={};stop=[False]
 try:
  before=w.reviewed.gpu_snapshot();w.append(out/'gpu-before.jsonl',{'utc':w.utc(),'gpu':before});w.require(before['free_bytes']>=e.LIMITS['min_gpu_bytes'],'Insufficient selected GPU memory')
  request={k:getattr(a,k) for k in ('binding','binding_sha256','selection','selection_sha256','disposition','disposition_sha256')};request.update(out=str(out),run_id=uuid.uuid4().hex);w.save(out/'request.json',request);w.save(out/'invocation.json',{'argv':sys.argv if argv is None else argv,'limits':e.LIMITS,'reference_release':'HOLD'})
  for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,lambda s,f:stop.__setitem__(0,True))
  with (out/'worker.log').open('xb',buffering=0) as log:
   child=w.OwnedWorker([sys.executable,'-B',str(root/'work/scripts/engineering_worker.py'),'--request',str(out/'request.json')],w.child_env('cuda'),log)
   w.save(out/'owned-worker.json',child.identity)
   result=w.supervise(child,out,'cuda',e.LIMITS,observe=w.reviewed.gpu_snapshot,stop=lambda:stop[0],validate_evidence=engineering_evidence,completion_reason='ENGINEERING_COMPLETE_NOT_FULL_COHORT')
  samples=list(e.rows(out/'samples.jsonl',4000)) if (out/'samples.jsonl').exists() else []
  usable=any(v['owned_gpu_bytes'] is not None and v['owned_gpu_bytes']>0 for _,v in samples)
  w.save(out/'monitoring-acceptance.json',{'computation':result['status'],'usable_owned_gpu_sample':usable,'status':'PASS' if result['status']=='PASS' and usable else 'INCOMPLETE','throughput_scope':'shared GPU measurements','reference_release':'HOLD'})
  return 0 if result['status']=='PASS' and usable else 1
 except BaseException as ex:
  cleanup=child.cleanup() if child is not None else None;w.save(out/'launcher-failure.json',{'error':repr(ex),'cleanup':cleanup,'reference_release':'HOLD'});return 1
 finally:
  for sig,old in handlers.items():signal.signal(sig,old)
if __name__=='__main__':raise SystemExit(main())
