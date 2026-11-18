from pathlib import Path
import sys,os,json,hashlib,zipfile,subprocess,time,datetime,traceback
E=Path('/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-diagnostic-training-review-20261013-6e91ab');X=Path(__file__).parent;W=E/'work'
sys.path[:0]=[str(W/'src'),str(W/'scripts')]
import diagnostic_training_watchdog as d
w=d.w;c=d.c
H='adb44927810502300f8931421ab867c27054b4c081d0f2061b55127349047310'
def save(n,v):w.save(X/n,v)
def fresh(p):
 assert not any(q.is_symlink() for q in [p,*p.parents])
 assert p.parent.is_dir() and not p.exists()
 for suffix in ('.shell.log','.shell.exit'):assert not Path(str(p)+suffix).exists()
report={'status':'PREFLIGHT','runs':{'18':'NOT_RUN','10':'NOT_RUN'},'reference_release':'HOLD','cap4000_policy':'NOT_ADOPTED'}
try:
 z=E.parent/(E.name+'-review.zip');assert w.digest(z)=='69ebbab6a43763726fba461d0d69f53d4c0d7b82877f0f91fa792c9e79f0339e'
 with zipfile.ZipFile(z) as f:
  assert f.testzip() is None
  for name in ('proposed-resnet18.sh','proposed-resnet10.sh','DRAFT-training-disposition.json'):
   matches=[n for n in f.namelist() if n.endswith('/'+name) or n==name];assert len(matches)==1
   assert f.read(matches[0])==(E/name).read_bytes()
 assert Path(d.__file__).resolve()==W/'scripts/diagnostic_training_watchdog.py'
 assert Path(c.__file__).resolve()==W/'src/lidc3d/diagnostic_training.py'
 b=w.bindings(E/'training-binding.json',H)
 for depth in (18,10):fresh(Path(c.DATA['outputs'][str(depth)]))
 draft=E/'DRAFT-training-disposition.json';draft_hash=w.digest(draft)
 adoption=json.loads(draft.read_text());adoption.update(status='ADOPTED',supplied_by='Fawad Karimi, user instruction in this conversation',date='2026-10-14 Asia/Bangkok (conversation date)',instruction="I adopt the packaged diagnostic training disposition in a separate ADOPTED record attributed to this instruction. Preserve the DRAFT. Bind the reviewed training identity: adb44927810502300f8931421ab867c27054b4c081d0f2061b55127349047310. I acknowledge the six reference holds, prior validation exposure, uncertain pretraining overlap, label-guided repeated sampling, only 15 unique training positives, no positive-coverage improvement, sampled monitoring and unmeasured CUDA instrumentation overhead. This is scoped engineering authorization, not scientific reference approval or cap4000 policy adoption. Authorize exactly one ResNet18 diagnostic run, followed by exactly one ResNet10 run only if ResNet18 passes all technical checks.")
 adoption['recorded_server_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();adoption['attribution_limitation']='User instruction; authorship, signature and factual completeness are not independently authenticated.'
 ap=E/'ADOPTED-training-disposition.json'
 with ap.open('x') as f:json.dump(adoption,f,indent=2,sort_keys=True);f.write(chr(10))
 ah=w.digest(ap);save('adoption-record.json',{'path':str(ap),'sha256':ah,'draft_sha256':draft_hash,'bounded_technical_acceptance_zip_sha256':w.digest(z)})
 prev=None
 for depth in (18,10):
  out=Path(c.DATA['outputs'][str(depth)]);fresh(out)
  req={'mode':'research','depth':depth,'binding':str(E/'training-binding.json'),'binding_sha256':H,'out':str(out),'disposition':str(ap),'disposition_sha256':ah,'acknowledge_shared_gpu_permission':True,'acknowledge_sampled_risk':True}
  if prev:req.update(previous=str(prev),previous_sha256=w.digest(prev))
  d.prerequisites(req)
  snap=w.reviewed.gpu_snapshot();ram=w.available_ram();disk=w.disk_free(out.parent)
  assert snap['free_bytes']>=c.RESEARCH_LIMITS['min_gpu_bytes'] and ram>=c.RESEARCH_LIMITS['min_ram_bytes'] and disk>=c.RESEARCH_LIMITS['min_disk_bytes']
  save('preflight-'+str(depth)+'.json',{'status':'PASS','source_files':len(b['sources']),'pins':len(b['pins']),'imports':{'controller':d.__file__,'contract':c.__file__},'gpu':snap,'ram':ram,'disk':disk,'request':req,'server_utc':w.utc()})
  argv=['bash',str(E/('proposed-resnet'+str(depth)+'.sh')),ah]
  if prev:argv.append(w.digest(prev))
  start=time.monotonic();save('invocation-'+str(depth)+'.json',{'argv':argv,'server_utc':w.utc()})
  with (X/('shell-'+str(depth)+'.log')).open('xb') as log:
   proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=w.child_env('cpu'))
   save('shell-'+str(depth)+'-identity.json',w.process_identity(proc.pid));code=proc.wait()
  save('shell-'+str(depth)+'-exit.json',{'exit_status':code,'elapsed_seconds':time.monotonic()-start})
  report['runs'][str(depth)]={'shell_exit':code};save('sequence.json',report)
  assert code==0,'Shell failure; no retry'
  assert int(Path(str(out)+'.shell.exit').read_text())==0
  acceptance=c.read(out/'acceptance.json');c.accepted(acceptance,'research',depth,H)
  worker=d.validate_worker(out/'research','cuda',c.RESEARCH_LIMITS)
  assert worker['update_state']['optimizer_updates']==180 and worker['update_state']['skipped_updates']==0
  own=w.load(out/'research/owned-worker.json');pp=Path('/proc')/str(own['pid'])
  assert not pp.exists() or w.process_identity(own['pid'])['start_ticks']!=own['start_ticks'],'Owned worker survives'
  w.bindings(E/'training-binding.json',H);c.verify_inputs();assert w.digest(draft)==draft_hash
  report['runs'][str(depth)]={'status':'PASS','shell_exit':code,'acceptance_sha256':w.digest(out/'acceptance.json'),'wall_seconds':acceptance['wall_seconds'],'diagnostics':worker['diagnostics']};save('sequence.json',report)
  prev=out/'acceptance.json'
 report['status']='PASS'
except BaseException as error:
 report['status']='FAIL';report['error']=repr(error);report['traceback']=traceback.format_exc()
finally:
 save('sequence.json',report)
 print(json.dumps(report,indent=2),flush=True)
