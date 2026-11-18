from pathlib import Path
import sys,json,hashlib,zipfile,shutil,math,traceback,collections,datetime
X=Path(__file__).parent;E=X.parent/'CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-diagnostic-training-review-20261013-6e91ab';W=E/'work';H='adb44927810502300f8931421ab867c27054b4c081d0f2061b55127349047310'
sys.path[:0]=[str(W/'src'),str(W/'scripts')]
import diagnostic_training_watchdog as d
w=d.w;c=d.c
def read(p):return json.loads(p.read_text())
def finite(v):
 if isinstance(v,float):assert math.isfinite(v)
 elif isinstance(v,dict):
  for z in v.values():finite(z)
 elif isinstance(v,list):
  for z in v:finite(z)
seq=read(X/'sequence.json');assert seq['status'] in ('PASS','FAIL')
P=X/'compact';P.mkdir();recon={'sequence':seq,'backbones':{},'reference_release':'HOLD','cap4000_policy':'NOT_ADOPTED'};omitted=[]
b=w.bindings(E/'training-binding.json',H);c.verify_inputs();recon['post_bindings']={'source_files':len(b['sources']),'pins':len(b['pins']),'inputs':'UNCHANGED'}
for depth in (18,10):
 out=Path(c.DATA['outputs'][str(depth)]);v={'status':'NOT_RUN'};recon['backbones'][str(depth)]=v
 if not out.exists():continue
 v['status']='OBSERVED';stage=out/'research';state=stage/'state'
 try:
  acc=read(out/'acceptance.json');v['acceptance']=acc;v['controller_exit']=int(Path(str(out)+'.shell.exit').read_text());v['shell']=read(X/('shell-'+str(depth)+'-exit.json'))
  own=read(stage/'owned-worker.json');proc=Path('/proc')/str(own['pid']);v['owned_worker_absent']=not proc.exists() or w.process_identity(own['pid'])['start_ticks']!=own['start_ticks']
  assert v['owned_worker_absent']
  counts=collections.Counter();classes={};lrs={};grad=[];changes=collections.Counter();clipped=0;val=[];micro={0:[],1:[],2:[]};boundaries=[]
  with (state/'training-diagnostics.jsonl').open() as f:
   for line in f:
    row=json.loads(line);finite(row);kind=row['kind'];counts[kind]+=1
    if kind=='class_losses':
     ep=row['epoch'];micro[ep].append(row['microbatch'])
     for label,z in row['classes'].items():
      key=str(ep)+':'+label;q=classes.setdefault(key,{'count':0,'weighted_loss_sum':0.,'unweighted_loss_sum':0.,'probability_sum':0.})
      for k in ('count','weighted_loss_sum','unweighted_loss_sum'):q[k]+=z[k]
      q['probability_sum']+=(z['positive_probability']['mean'] or 0)*z['count']
    elif kind=='optimizer_boundary':
     boundaries.append(row);grad.append(row['gradient_norm_before_clip']);clipped+=int(row['clipping_applied']);lrs.setdefault(str(row['epoch']),{})
     for g in row['groups']:lrs[str(row['epoch'])].setdefault(str(g['index']),set()).add(g['lr'])
     for k,z in row['families_sampled'].items():changes[k]+=z['changed_values']
    elif kind=='complete_validation_diagnostics':val.append(row)
  for q in classes.values():
   if q['count']:
    q['mean_unweighted_loss']=q['unweighted_loss_sum']/q['count'];q['mean_positive_probability']=q['probability_sum']/q['count']
  v.update(diagnostic_counts=dict(counts),training_class_statistics=classes,group_learning_rates={ep:{k:sorted(z) for k,z in groups.items()} for ep,groups in lrs.items()},gradient_norm_range=[min(grad),max(grad)] if grad else None,clipped_boundaries=clipped,sampled_changed_values=dict(changes),validation_diagnostics=val)
  samples=[]
  with (stage/'samples.jsonl').open() as f:
   for line in f:samples.append(json.loads(line))
  v['resources']={'samples':len(samples),'usable_gpu_samples':sum(bool(z.get('owned_gpu_bytes')) for z in samples),'peak_rss':max((z['owned_worker']['rss_bytes'] for z in samples),default=0),'peak_owned_gpu':max((z.get('owned_gpu_bytes') or 0 for z in samples),default=0),'no_observed_descendants':all(not z['owned_worker']['children'] for z in samples)}
  commits={};chain=[]
  for p in state.glob('epoch-checkpoints/*/commit.json'):
   z=read(p);assert w.digest(p.parent/'state.pt')==z['state_sha256'];commits[w.digest(p)]=z
  pointer=read(state/'epoch-commit.json');cursor=pointer['commit_sha256'];seen=set()
  while cursor:
   assert cursor not in seen;seen.add(cursor);z=commits[cursor];chain.append({'sha256':cursor,'epoch':z['epoch'],'parent':z['parent']});parent=z['parent']
   cursor=parent if isinstance(parent,str) else parent.get('commit_sha256') if parent else None
  v['checkpoint_chain']=chain
  if acc['status']=='PASS':
   c.accepted(acc,'research',depth,H);worker=d.validate_worker(stage,'cuda',c.RESEARCH_LIMITS);v['worker']=worker
   for ep in range(3):
    assert micro[ep]==list(range(480));assert classes[str(ep)+':0']['count']==720 and classes[str(ep)+':1']['count']==240
    plan=read(Path(c.DATA['inputs']['draw'+str(ep)]['path']));got=read(state/('sampling-'+str(ep)+'.json'));assert got['source_indices']==plan['indices'];assert {k:z for k,z in got.items() if k!='source_indices'}==plan['receipt']
   assert len(boundaries)==180 and all(z['actual_updates']==1 and z['counters']['skipped_updates']==0 and z['counters']['optimizer_updates']==i+1 for i,z in enumerate(boundaries))
   assert [z['epoch'] for z in chain]==[2,1,0,-1]
   v['independent_reconciliation']='PASS'
  else:v['independent_reconciliation']='FAILED_EXECUTION_PRESERVED'
 except BaseException as error:v['reconciliation_error']=repr(error);v['traceback']=traceback.format_exc();v['independent_reconciliation']='FAIL'
 for p in sorted(out.rglob('*')):
  if not p.is_file():continue
  assert not p.is_symlink()
  if p.suffix in ('.pt','.npy','.npz'):
   omitted.append({'path':str(p),'bytes':p.stat().st_size,'sha256':w.digest(p),'reason':'Large tensor bytes retained on server; independent tensor comparison/resume requires original bytes.'});continue
  target=P/'runs'/out.name/p.relative_to(out);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
 for suffix in ('.shell.log','.shell.exit'):
  p=Path(str(out)+suffix)
  if p.exists():shutil.copyfile(p,P/(out.name+suffix))
for n in ['training-binding.json','DRAFT-training-disposition.json','ADOPTED-training-disposition.json','configuration-resnet18.json','configuration-resnet10.json','proposed-resnet18.sh','proposed-resnet10.sh','PROPOSED-INVOCATIONS.md']:
 shutil.copyfile(E/n,P/n)
for rel,h in b['sources'].items():
 p=W/rel;assert w.digest(p)==h;target=P/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
for p in X.iterdir():
 if p.is_file():shutil.copyfile(p,P/p.name)
for name,v in [('reconciliation.json',recon),('omitted-checkpoints.json',omitted)]:
 (P/name).write_text(json.dumps(v,indent=2,sort_keys=True)+chr(10))
(P/'README.md').write_text('Diagnostic engineering execution evidence. Reference HOLD and cap4000 NOT_ADOPTED remain. Exactly one attempt per launched backbone; no retry or resume. Full logs and scalar instrumentation are included. Sampled monitoring is not containment; parameter and activation statistics are sampled. Only 15 unique training positives, repeated by accepted draw plans. No positive-coverage improvement, model-quality winner, untouched-test or scientific approval claim. Checkpoint tensors remain at inventoried server paths; this compact ZIP does not permit independent tensor comparisons without those bytes. Server timestamps are UTC; conversation adoption date uses Asia/Bangkok. CUDA instrumentation overhead has no matched instrumentation-off measurement. See reconciliation.json for observed results and failures.\n')
files=sorted(p for p in P.rglob('*') if p.is_file());(P/'SHA256SUMS').write_text(''.join(w.digest(p)+'  '+str(p.relative_to(P))+chr(10) for p in files))
z=X.parent/(X.name+'-evidence.zip');assert not z.exists()
with zipfile.ZipFile(z,'x',zipfile.ZIP_DEFLATED) as f:
 for p in sorted(P.rglob('*')):
  if p.is_file():f.write(p,str(p.relative_to(P)))
with zipfile.ZipFile(z) as f:
 assert f.testzip() is None
 for line in f.read('SHA256SUMS').decode().splitlines():
  h,n=line.split('  ',1);assert hashlib.sha256(f.read(n)).hexdigest()==h
receipt={'path':str(z),'bytes':z.stat().st_size,'sha256':w.digest(z),'zip_crc_and_member_checksums':'PASS','backbones':{k:{'status':v.get('independent_reconciliation',v['status']),'resources':v.get('resources'),'error':v.get('reconciliation_error')} for k,v in recon['backbones'].items()}}
w.save(X/'package-receipt.json',receipt);(X/'package-receipt.txt').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
