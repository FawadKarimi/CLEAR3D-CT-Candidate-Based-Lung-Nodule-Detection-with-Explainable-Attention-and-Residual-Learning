"""Bounded six-scan engineering path. No release, threshold or quality selection."""
from __future__ import annotations
import copy, hashlib, heapq, json, math, os, random, time, uuid
from pathlib import Path
from .provenance import sha256, canonical_hash, verify_scan_identities, write_json
from .derivative import HOLD_IDS
PREPARATION='47f5a6a740b405d758431e56f241993e48c93b330fa7859d93a7133378fa120a'
RULE={'version':'engineering-six-v1','seed':17,'roles':{'train':4,'val':2},'rows_per_scan':64,'total_rows':384,'batch_size':2,'accumulation':8,'num_workers':0,'passes':1,'proposal_policy':'existing minimum-threshold pool, no threshold selection','class_rebalancing':False}
LIMITS={'setup_seconds':900,'run_seconds':1800,'validation_seconds':600,'total_seconds':2700,'rss_bytes':8*1024**3,'allocator_bytes':4*1024**3,'gpu_bytes':6*1024**3,'sample_seconds':1.,'min_ram_bytes':32*1024**3,'reserve_ram_bytes':8*1024**3,'min_gpu_bytes':10*1024**3,'min_disk_bytes':20*1024**3}
def require(ok,why):
 if not ok:raise ValueError(why)
def regular(path):
 p=Path(path).absolute();require(not any(x.is_symlink() for x in [p,*p.parents]) and p.is_file(),'Unsafe/missing file');return p
def bounded_json(path,limit=16*1024**2):
 p=regular(path);require(p.stat().st_size<=limit,'Oversized metadata');return json.loads(p.read_text())
def rows(path,max_rows=None,line_limit=1024**2):
 with regular(path).open('rb') as f:
  ordinal=0
  while True:
   line=f.readline(line_limit+1)
   if not line:break
   require(len(line)<=line_limit and line.strip(),'Oversized/empty JSONL row')
   require(max_rows is None or ordinal<max_rows,'Metadata row bound exceeded')
   yield ordinal,json.loads(line);ordinal+=1
def checked(root,relative,expected):
 rel=Path(relative);require(not rel.is_absolute() and '..' not in rel.parts,'Path escapes root');p=regular(Path(root)/rel);require(Path(root).resolve() in p.resolve().parents,'Path escapes root')
 h=expected if isinstance(expected,str) else expected['sha256'];require(sha256(p)==h,'Hash changed: '+relative)
 if isinstance(expected,dict):require(p.stat().st_size==expected['bytes'],'Size changed: '+relative)
 return p
def rank_scan(sid):return hashlib.sha256(('LIDC3D-engineering-v1|17|'+sid).encode()).hexdigest()
def rank_row(sid,ordinal):return hashlib.sha256(('LIDC3D-engineering-candidate-v1|17|'+sid+'|'+str(ordinal)).encode()).hexdigest()
def select_scans(metadata):
 identities=[]
 for s in metadata:
  require(isinstance(s['scan_id'],str) and Path(s['scan_id']).name==s['scan_id'] and s['scan_id'] not in ('.','..'),'Unsafe scan ID');require(s['split'] in ('train','val') and s.get('fold')==0,'Not fold-0 development')
  identities.append({k:s[k] for k in ('scan_id','patient_id','split','pixel_fingerprint','series_uid','patient_grouped_subset')})
 require(len(identities)<=797,'Population bound');verify_scan_identities(identities)
 selected=[];skipped=[]
 for role,number in RULE['roles'].items():
  seen=set();chosen=[]
  for s in sorted((s for s in identities if s['split']==role),key=lambda s:(rank_scan(s['scan_id']),s['scan_id'])):
   if s['patient_id'] in seen:skipped.append({'scan_id':s['scan_id'],'reason':'same_patient'});continue
   seen.add(s['patient_id']);chosen.append(s)
   if len(chosen)==number:break
  require(len(chosen)==number,'Insufficient prespecified patients');selected.extend(chosen)
 return selected,skipped

def select_rows(path,sid,expected_hash,expected_count,deadline=lambda:None):
 """Hash the entire stream, retain <=64 rows including distinct duplicate ordinals."""
 heap=[];h=hashlib.sha256();count=0
 with regular(path).open('rb') as f:
  while True:
   deadline();raw=f.readline(16385)
   if not raw:break
   require(len(raw)<=16384 and raw.strip(),'Invalid proposal line bound');h.update(raw);row=json.loads(raw)
   require(row.get('scan_id')==sid and row.get('id')==f'{sid}:candidate:{count:06d}','Proposal order/identity mismatch')
   require(all(len(row[k])==3 and all(math.isfinite(float(x)) for x in row[k]) for k in ('zyx','center_lps')),'Invalid coordinates')
   rank=int(rank_row(sid,count),16);item=(-rank,-count,count,row)
   if len(heap)<64:heapq.heappush(heap,item)
   elif item[:2]>heap[0][:2]:heapq.heapreplace(heap,item)
   count+=1
 require(h.hexdigest()==expected_hash and count==expected_count,'Incomplete/changed proposal stream')
 return [{'ordinal':i,'rank':rank_row(sid,i),'row':r} for _,__,i,r in sorted(heap,key=lambda x:(-x[0],x[2]))]

def make_selection(root,expected_preparation,deadline=lambda:None):
 """Metadata/proposal-only; never open arrays, raw sources, audits or mask files."""
 root=Path(root).absolute();prep=bounded_json(checked(root,'preparation.json',expected_preparation))
 require(prep['status']=='PROPOSALS_NOT_YET_FROZEN','Wrong preparation stage')
 scans_path=checked(root,'scans.jsonl',prep['files']['scans.jsonl'])
 selected,skipped=select_scans(s for _,s in rows(scans_path,797));ids={s['scan_id'] for s in selected}
 protocol=bounded_json(checked(root,'protocol.json',prep['derivation']['files']['protocol.json']))
 fold=next(f for f in protocol['folds'] if f['fold']==0)
 for s in selected:
  role='validation' if s['split']=='val' else 'train';detail=fold['roles'][role]
  require(s['series_uid'] in detail['series_uids'] and s['patient_id'] in detail['patient_ids'] and s['patient_grouped_subset'] in detail['subsets'],'Protocol role mismatch')
 require(set(fold['roles']['train']['patient_ids']).isdisjoint(fold['roles']['validation']['patient_ids']) and set(fold['roles']['test']['patient_ids']).isdisjoint(set(fold['roles']['train']['patient_ids'])|set(fold['roles']['validation']['patient_ids'])),'Protocol patient conflict')
 holds=bounded_json(checked(root,'release-holds.json',prep['derivation']['files']['release-holds.json']))
 require(set(HOLD_IDS).issubset(holds['unresolved']),'Holds missing')
 byid={s['scan_id']:s for _,s in rows(scans_path,797) if s['scan_id'] in ids};entries=[]
 for identity in selected:
  deadline();sid=identity['scan_id'];s=byid[sid];require(s['status']=='ok','Selected scan failed; no replacement')
  record=bounded_json(checked(root,sid+'/record.json',prep['artifacts'][sid+'/record.json']));require(record==s,'Record differs')
  cp=bounded_json(checked(root,sid+'/checkpoint.json',prep['artifacts'][sid+'/checkpoint.json']))
  require(cp['scan_id']==sid and cp['preparation_key']==prep['preparation_key'],'Checkpoint ancestry mismatch')
  expected=prep['artifacts'][sid+'/proposals.jsonl'];require(cp['files']['proposals.jsonl']==expected,'Proposal binding mismatch')
  picked=select_rows(root/sid/'proposals.jsonl',sid,expected['sha256'],cp['proposal_count'],deadline)
  require(all(not r.get('mask_path') for r in s['positive']+s.get('ignored',[])),'External reference masks outside reviewed LUNA scope')
  entries.append({'scan':s,'selected':picked,'proposal_sha256':expected['sha256'],'proposal_count':cp['proposal_count'],'arrays':{n:prep['artifacts'][sid+'/'+n+'.npy'] for n in ('image','valid')}})
 require(sum(len(e['selected']) for e in entries)<=384,'Row cap exceeded')
 require(sha256(scans_path)==prep['files']['scans.jsonl'],'Scan metadata changed during selection')
 return {'schema':'engineering-selection-v1','rule':RULE,'preparation_sha256':expected_preparation,'root':str(root),'entries':entries,'skipped_same_patient':skipped,'reference_release':'HOLD','validation_scope':'two selected development scans only','config':prep['config']}

def scoped_disposition(path,expected_hash,selection_hash,source_hash):
 """An external factual decision, not created or authenticated by this adapter."""
 p=regular(path);require(sha256(p)==expected_hash,'Disposition hash mismatch');d=bounded_json(p)
 require(d.get('scope')=='six_scan_exploratory_engineering' and d.get('reference_release')=='HOLD','Wrong disposition scope')
 require(d.get('preparation_sha256')==PREPARATION and d.get('selection_sha256')==selection_hash and d.get('source_binding_sha256')==source_hash,'Disposition binding mismatch')
 require(d.get('rule')==RULE and d.get('limits')==LIMITS,'Disposition limits/rule mismatch')
 require(d.get('permitted_label_use')=='engineering_loss_only' and d.get('candidate_policy')==RULE['proposal_policy'],'Label/candidate policy not covered')
 require(all(isinstance(d.get(k),str) and d[k].strip() for k in ('decision_by','date','conditions','decision_text')),'Missing scoped factual decision')
 require(set(HOLD_IDS).issubset(d.get('acknowledged_unresolved_holds',[])),'Unresolved findings omitted');return d

def verify_selection(selection):
 require(selection.get('rule')==RULE and selection.get('reference_release')=='HOLD','Selection rule changed')
 rebuilt=make_selection(selection['root'],selection['preparation_sha256'])
 require(rebuilt==selection,'Selection does not match prespecified rule');return selection

def labelled_rows(entries):
 from .references import candidate_label
 usable=[];statuses=[]
 for e in entries:
  s=e['scan'];ignored=[];kept=[]
  for r in e['selected']:
   label=candidate_label(r['row']['center_lps'],s['positive'],s.get('ignored',[]))
   if label==-1:ignored.append(r['ordinal'])
   else:kept.append({**r['row'],'label':label});usable.append({**r['row'],'label':label})
  statuses.append({'scan_id':s['scan_id'],'selected':len(e['selected']),'usable':len(kept),'ignored_ordinals':ignored,'status':'LABELLED'})
 return usable,statuses

def prepare_arrays(selection):
 """Only selected image/valid bytes; mmap patch reader used later. No mask/source CT."""
 import numpy as np
 for e in selection['entries']:
  s=e['scan'];sid=s['scan_id']
  for n,dtype in [('image',np.dtype('float32')),('valid',np.dtype('bool'))]:
   p=checked(selection['root'],sid+'/'+n+'.npy',e['arrays'][n]);require(Path(s[n+'_path']).absolute()==p,'Array path escapes selected preparation')
   a=np.load(p,mmap_mode='r',allow_pickle=False);require(a.ndim==3 and a.dtype==dtype,'Array type/shape changed')
   if n=='image':shape=a.shape
   else:require(a.shape==shape,'Validity shape differs')

# Reuse the reviewed durable publication primitives with an explicit engineering
# schema. No fabricated CPM/best/stopping fields or best.pt are created.
def publish_checkpoint(out,state,binding):
 import torch
 from .checkpoints import _json_atomic,_sync_dir,_boundary
 out=Path(out);previous=None
 if (out/'engineering-commit.json').exists():previous=load_checkpoint(out,binding)
 generation=out/'engineering-checkpoints'/uuid.uuid4().hex
 parent=previous['commit_sha256'] if previous else None
 require(state['stage'] in ('INITIALIZED','TRAINED'),'Invalid engineering checkpoint stage')
 if previous:require(previous['state']['stage']=='INITIALIZED' and state['stage']=='TRAINED','Checkpoint stage replay refused')
 try:
  _json_atomic(out/'engineering-commit.json',{'status':'PUBLISHING','previous_commit':parent});_boundary('engineering_intent')
  generation.mkdir(parents=True);_sync_dir(generation.parent)
  with (generation/'state.pt').open('xb') as f:torch.save({**state,'binding':binding},f);f.flush();os.fsync(f.fileno())
  _boundary('engineering_state')
  record={'schema':'engineering-checkpoint-v1','status':'COMMITTED','binding':binding,'stage':state['stage'],'state_sha256':sha256(generation/'state.pt'),'parent_commit':parent}
  _json_atomic(generation/'commit.json',record);_boundary('engineering_sealed')
  _json_atomic(out/'engineering-commit.json',{'status':'COMMITTED','generation':generation.relative_to(out).as_posix(),'commit_sha256':sha256(generation/'commit.json')})
 except BaseException as ex:
  write_json(out/('checkpoint-failure-'+uuid.uuid4().hex+'.json'),{'error':repr(ex),'generation':str(generation),'previous_commit':parent});raise

def load_checkpoint(out,binding):
 import torch
 out=Path(out);pointer=bounded_json(out/'engineering-commit.json');require(pointer['status']=='COMMITTED','Incomplete engineering publication')
 rel=Path(pointer['generation']);require(len(rel.parts)==2 and rel.parts[0]=='engineering-checkpoints' and not rel.is_absolute(),'Invalid generation')
 gen=out/rel;record=bounded_json(checked(out,rel.as_posix()+'/commit.json',pointer['commit_sha256']))
 require(record['schema']=='engineering-checkpoint-v1' and record['binding']==binding and record['status']=='COMMITTED','Checkpoint identity mismatch')
 p=checked(out,rel.as_posix()+'/state.pt',record['state_sha256']);state=torch.load(p,map_location='cpu',weights_only=False)
 require(state['binding']==binding and state['stage']==record['stage'],'State binding mismatch')
 from .optimization import validate_update_state
 validate_update_state(state['update_state']);require(canonical_hash(state['config'])==binding['config_hash'] and state['selection_sha256']==binding['selection_sha256'],'Checkpoint configuration/selection mismatch');require(state['stage']!='TRAINED' or state['update_state']['optimizer_updates']>0,'Trained checkpoint has no updates');return {'state':state,'commit_sha256':pointer['commit_sha256'],'record':record}

def run_pass(selection,config,out,device_name,binding,phase=lambda name:None,model_factory=None):
 """One fixed pass, no experiment selection. Callable synthetic seam, no CLI bypass."""
 import numpy as np,torch
 from torch.utils.data import DataLoader
 from .data import CandidateDataset
 from .models import DetectorClassifier
 from .provenance import seed_all
 from .training import parameter_groups,epoch_lr,predict,_record_validation
 from .optimization import train_epoch,new_update_state,POLICY,append_diagnostic
 out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();status=[]
 try:
  require(selection['rule']==RULE and len(selection['entries'])==6,'Invalid selected scope');verify_scan_identities([x['scan'] for x in selection['entries']]);require(all(len(x['selected'])<=64 for x in selection['entries']),'Row cap changed');require(config==selection['config'] and canonical_hash(config)==binding['config_hash'],'Configuration changed')
  require(config['training']['microbatch']==2 and config['training']['accumulation']==8 and config['training']['alignment_weight']==0,'Training policy changed')
  prepare_arrays(selection);seed_all(17);device=torch.device(device_name)
  model=(model_factory or DetectorClassifier)(config['model']).to(device)
  groups=parameter_groups(model,config['training'])
  for g in groups:g['lr']=epoch_lr(g['lr'],0,config['training'])
  opt=torch.optim.AdamW(groups);amp=bool(config['training']['amp'] and device.type=='cuda');scaler=torch.amp.GradScaler('cuda',enabled=amp);state=new_update_state()
  train=[e for e in selection['entries'] if e['scan']['split']=='train'];val=[e for e in selection['entries'] if e['scan']['split']=='val']
  require(len(train)==4 and len(val)==2,'Role count mismatch');train_rows,status=labelled_rows(train);write_json(out/'training-label-status.json',status)
  dataset=CandidateDataset([e['scan'] for e in train],train_rows,config['model']['patch_size'],True,17)
  loader=DataLoader(dataset,batch_size=2,shuffle=False,num_workers=0)
  def snapshot(stage):return {'stage':stage,'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'update_state':copy.deepcopy(state),'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,'selection_sha256':binding['selection_sha256'],'config':config,'rows_consumed':len(train_rows) if stage=='TRAINED' else 0}
  publish_checkpoint(out,snapshot('INITIALIZED'),binding);training_start=time.monotonic();phase('training');model.train()
  loss=train_epoch(model,loader,opt,scaler,torch.tensor(config['training']['class_weights'],device=device,dtype=torch.float32),device,amp,8,config['training']['clip_grad_norm'],state,lambda event:append_diagnostic(out/'boundaries.jsonl',event),0)
  require(math.isfinite(loss) and state['optimizer_updates']>0,'No finite positive-update pass')
  publish_checkpoint(out,snapshot('TRAINED'),binding)
  for s in status:s['status']='COMPLETE_ONE_PASS'
  write_json(out/'training-scan-status.json',status);validation_start=time.monotonic();phase('validation')
  val_scans=[e['scan'] for e in val];val_rows=[r['row'] for e in val for r in e['selected']]
  predictions,statuses=predict(model,val_scans,val_rows,config['model']['patch_size'],device,2)
  _record_validation(out,'engineering_validation',val_scans,predictions,statuses)
  from .references import candidate_label
  byid={s['scan_id']:s for s in val_scans};validation_losses=[];ignored=0
  for r in predictions:
   s=byid[r['scan_id']];label=candidate_label(r['center_lps'],s['positive'],s.get('ignored',[]))
   if label==-1:ignored+=1;continue
   value=torch.nn.functional.cross_entropy(torch.tensor([r['logits']],dtype=torch.float32),torch.tensor([label]),weight=torch.tensor(config['training']['class_weights'],dtype=torch.float32),reduction='sum').item();require(math.isfinite(value),'Nonfinite validation loss');validation_losses.append(value)
  restored=load_checkpoint(out,binding);require(restored['state']['update_state']==state,'Checkpoint counters differ')
  result={'status':'PASS','scope':'six-scan engineering only; NOT full-cohort validation','train_loss':loss,'validation_weighted_loss_sums':validation_losses,'validation_ignored':ignored,'validation_scan_status':statuses,'update_state':state,'checkpoint_commit_sha256':restored['commit_sha256'],'phase_seconds':{'setup':training_start-started,'training':validation_start-training_start,'validation':time.monotonic()-validation_start},'reference_release':'HOLD'}
  if device.type=='cuda':result.update(max_allocated=torch.cuda.max_memory_allocated(),max_reserved=torch.cuda.max_memory_reserved())
  write_json(out/'engineering-result.json',result);phase('complete');return result
 except BaseException as ex:
  write_json(out/'engineering-failure.json',{'status':'INCOMPLETE','error':repr(ex),'training_scan_status':status,'no_automatic_resume':True});raise
