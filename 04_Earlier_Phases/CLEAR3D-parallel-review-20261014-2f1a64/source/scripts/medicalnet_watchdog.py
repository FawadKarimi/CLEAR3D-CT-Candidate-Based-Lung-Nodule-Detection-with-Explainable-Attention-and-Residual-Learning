"""Plan-only default; isolated transfer qualification and scoped engineering launcher."""
import argparse,json,os,signal,sys,uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import user_watchdog as w
from lidc3d import medicalnet_contract as c

def prerequisites(a):
 c.require(a.binding and a.binding_sha256 and a.checkpoint and a.depth and a.out,'Missing bound inputs')
 b=w.bindings(a.binding,a.binding_sha256)
 c.require(c.digest(a.checkpoint)==c.WEIGHTS[a.depth],'Checkpoint hash mismatch')
 c.require(a.acknowledge_sampled_risk and a.acknowledge_shared_gpu_permission,'Explicit shared/sampled acknowledgement required')
 if a.mode=='pilot':
  c.require(all(getattr(a,k) for k in ('selection','disposition','disposition_sha256','provenance','provenance_sha256','qualification','qualification_sha256')),'Pilot prerequisites absent')
  c.require(c.digest(a.selection)==c.SELECTION,'Selection hash mismatch')
  c.disposition(c.read(a.disposition,a.disposition_sha256),a.binding_sha256)
  c.provenance(c.read(a.provenance,a.provenance_sha256))
  q=c.read(a.qualification,a.qualification_sha256)
  c.require(q.get('status')=='PASS' and q.get('usable_owned_gpu_sample') is True,'Generated monitoring acceptance absent')
  c.generated_acceptance(q['worker'],a.depth,a.binding_sha256)
 return b

def evidence(out,device,limits):
 r=w.load(out/'request.json');result=c.read(out/'worker-result.json')
 c.require(result['status']=='PASS' and result['device']==device and result['depth']==r['depth'],'Worker identity/result mismatch')
 c.require(result['source_binding_sha256']==r['binding_sha256'] and result['transfer']==c.transfer(r['depth']),'Transfer identity mismatch')
 c.require(w.load(out/'phase.json')['name']=='complete','Incomplete phase')
 c.require(result['max_allocated']<=limits['allocator_bytes'],'Allocator budget')
 durations=result['phase_seconds'];c.require(durations['setup']<=limits['setup_seconds'] and durations['training']+durations['validation']<=limits['run_seconds'] and durations['validation']<=limits['validation_seconds'],'Phase budget')
 if r['mode']=='generated':c.generated_acceptance(result,r['depth'],r['binding_sha256'])
 else:
  from mixed_engineering_watchdog import engineering_evidence
  engineering_evidence(out,device,limits)
 # Verify immutable publication and the exact identity without GPU tensor loading.
 folders=['pass'] if r['mode']=='pilot' else ['repeat-0','repeat-1']
 for name in folders:
  p=c.read(out/name/'engineering-commit.json');c.require(p['status']=='COMMITTED','Incomplete checkpoint commit')
  g=(out/name/p['generation']);c.require(g.resolve().parent==(out/name/'engineering-checkpoints').resolve(),'Checkpoint escape')
  commit=c.read(g/'commit.json',p['commit_sha256']);expected=c.read(out/name/'expected-identity.json')
  c.require(commit['binding']==expected and commit['binding']['transfer']==c.transfer(r['depth']),'Checkpoint transfer binding mismatch')
  c.require(c.digest(g/'state.pt')==commit['state_sha256'],'Checkpoint bytes changed')
 return result

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--mode',choices=['generated','pilot'],default='generated');p.add_argument('--depth',type=int,choices=[18,10])
 for name in ('binding','binding-sha256','checkpoint','out','selection','disposition','disposition-sha256','provenance','provenance-sha256','qualification','qualification-sha256'):p.add_argument('--'+name)
 p.add_argument('--acknowledge-shared-gpu-permission',action='store_true');p.add_argument('--acknowledge-sampled-risk',action='store_true');a=p.parse_args(argv)
 if not a.execute:
  print(json.dumps({'status':'PLAN_ONLY','policy':c.POLICY,'generated_limits':c.GENERATED_LIMITS,'pilot_limits':c.LIMITS,'reference_release':'HOLD','torch_imported':'torch' in sys.modules}));return 0
 prerequisites(a);out=Path(a.out).absolute();work=Path(__file__).resolve().parents[1]
 c.require(out.parent==work.parent/'runs' and out.parent.is_dir() and not out.exists() and not any(p.is_symlink() for p in [out,*out.parents]),'Unsafe/existing output')
 limits=c.GENERATED_LIMITS if a.mode=='generated' else c.LIMITS
 c.require(w.available_ram()>=limits['min_ram_bytes'] and w.disk_free(out.parent)>=limits['min_disk_bytes'],'Resource headroom')
 out.mkdir();child=None;handlers={};stopped=[False]
 try:
  r={**vars(a),'out':str(out),'run_id':uuid.uuid4().hex,'limits':limits};w.save(out/'request.json',r);w.save(out/'invocation.json',{'argv':sys.argv if argv is None else argv,'reference_release':'HOLD'})
  snap=w.reviewed.gpu_snapshot();w.append(out/'gpu-before.jsonl',snap);c.require(snap['free_bytes']>=limits['min_gpu_bytes'],'GPU headroom')
  for s in (signal.SIGINT,signal.SIGTERM):handlers[s]=signal.signal(s,lambda *args:stopped.__setitem__(0,True))
  with (out/'worker.log').open('xb',buffering=0) as log:
   child=w.OwnedWorker([sys.executable,'-B',str(work/'scripts/medicalnet_worker.py'),'--request',str(out/'request.json')],w.child_env('cuda'),log);w.save(out/'owned-worker.json',child.identity)
   result=w.supervise(child,out,'cuda',limits,observe=w.reviewed.gpu_snapshot,stop=lambda:stopped[0],validate_evidence=evidence,completion_reason='MEDICALNET_ENGINEERING_ONLY')
  usable=False
  if (out/'samples.jsonl').exists():
   with (out/'samples.jsonl').open() as f:
    for i,line in enumerate(f):
     c.require(i<2000 and len(line)<1048576,'Sample bound');usable|=bool(json.loads(line).get('owned_gpu_bytes'))
  accepted={**result,'usable_owned_gpu_sample':usable,'status':'PASS' if result['status']=='PASS' and usable else 'INCOMPLETE','throughput_scope':'shared GPU; generated fixtures' if a.mode=='generated' else 'shared GPU; selected development engineering only'}
  w.save(out/'qualification-acceptance.json',accepted);return 0 if accepted['status']=='PASS' else 1
 except BaseException as ex:
  w.save(out/'launcher-failure.json',{'error':repr(ex),'cleanup':child.cleanup() if child else None,'reference_release':'HOLD'});return 1
 finally:
  for s,old in handlers.items():signal.signal(s,old)
if __name__=='__main__':raise SystemExit(main())
