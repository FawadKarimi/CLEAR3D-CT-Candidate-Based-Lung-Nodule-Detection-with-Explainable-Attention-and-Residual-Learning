"""Separate label-guided engineering policy; original engineering guard unchanged."""
from __future__ import annotations
import copy,hashlib,heapq,json,math,os,random,time,uuid
from pathlib import Path
from . import engineering as original
from . import positive_engineering as parent
from .engineering import require,regular,bounded_json,rows,checked,prepare_arrays,publish_checkpoint,load_checkpoint
from .provenance import sha256,canonical_hash,verify_scan_identities,write_json
PREPARATION=original.PREPARATION
LIMITS=original.LIMITS.copy()
HOLD_IDS=original.HOLD_IDS
CONTRACT=json.loads(Path(__file__).with_name('fp32_contract.json').read_text())
RULE={**parent.RULE,'version':'fp32-five-positive-v1','experiment_id':'lidc3d-engineering-fp32-five-positive-v1-seed17','minimum_selected_train_positives_total':5,'exact_training_positive_rows':5,'exact_validation_rows':68,'precision':'float32','amp':False,'expected_boundaries':1,'expected_denominator':15,'contract_sha256':canonical_hash(CONTRACT)}
SCHEMA='fp32-five-positive-selection-v1'
def effective_config(parent_config):
 c=copy.deepcopy(parent_config);require(c['training']['class_weights']==[1,3],'Positive class weight must remain 3');c['training']['amp']=False;return c

def verify_config(selection,config):
 require(config==effective_config(selection['parent_config']) and config==selection['config'],'FP32 effective config changed')
 require(config['training']['amp'] is False,'FP32 only')

def rank_scan(sid):return hashlib.sha256(('LIDC3D-positive-engineering-scan-v1|17|'+sid).encode()).hexdigest()
def rank_row(sid,label,ordinal):return hashlib.sha256(('LIDC3D-positive-engineering-row-v1|17|'+sid+'|'+str(label)+'|'+str(ordinal)).encode()).hexdigest()
def inspection_pool(metadata):
 pool,excluded=parent.inspection_pool(metadata)
 require([(s['scan_id'],s['patient_id'],s['split']) for s in pool]==[(s['scan_id'],s['patient_id'],s['role']) for s in CONTRACT['pool']],'Fixed inspection identities changed')
 return pool,excluded

def select_rows(path,scan,expected_hash,expected_count,deadline=lambda:None):
 from .references import candidate_label
 require(type(expected_count) is int and 0<=expected_count<=RULE['proposal_stream_cap_per_scan'],'Per-stream cap/count')
 require(all(not r.get('mask_path') for r in scan['positive']+scan.get('ignored',[])),'External reference masks forbidden')
 sid=scan['scan_id'];heaps={0:[],1:[]};counts={'positive':0,'negative':0,'ignored':0};h=hashlib.sha256();count=0
 with regular(path).open('rb') as f:
  while True:
   deadline();raw=f.readline(16385)
   if not raw:break
   require(count<expected_count and count<RULE['proposal_stream_cap_per_scan'],'Stream count/cap exceeded')
   require(len(raw)<=16384 and raw.strip(),'Invalid proposal line bound');h.update(raw);row=json.loads(raw)
   require(row.get('scan_id')==sid and row.get('id')==f'{sid}:candidate:{count:06d}','Proposal order/identity mismatch')
   require(all(len(row[k])==3 and all(math.isfinite(float(x)) for x in row[k]) for k in ('zyx','center_lps')),'Invalid coordinates')
   label=int(candidate_label(row['center_lps'],scan['positive'],scan.get('ignored',[])));counts[{1:'positive',0:'negative',-1:'ignored'}[label]]+=1
   if label!=-1:
    rank=int(rank_row(sid,label,count),16);item=(-rank,-count,count,row,hashlib.sha256(raw).hexdigest());heap=heaps[label]
    if len(heap)<32:heapq.heappush(heap,item)
    elif item[:2]>heap[0][:2]:heapq.heapreplace(heap,item)
   count+=1
 require(h.hexdigest()==expected_hash and count==expected_count,'Incomplete/changed proposal stream')
 selected=[{'ordinal':i,'rank':rank_row(sid,label,i),'label':label,'row':r,'payload_sha256':canonical_hash(r),'raw_sha256':rawhash} for label in [1,0] for _,__,i,r,rawhash in sorted(heaps[label],key=lambda x:(-x[0],x[2]))]
 return selected,counts

def make_selection(root,expected_preparation,deadline=lambda:None,observe=lambda event:None):
 root=Path(root).absolute();deadline();prep=bounded_json(checked(root,'preparation.json',expected_preparation))
 require(prep['status']=='PROPOSALS_NOT_YET_FROZEN','Wrong preparation stage')
 scans_path=checked(root,'scans.jsonl',prep['files']['scans.jsonl']);metadata=[s for _,s in rows(scans_path,797)];pool,excluded=inspection_pool(metadata)
 protocol=bounded_json(checked(root,'protocol.json',prep['derivation']['files']['protocol.json']));fold=next(f for f in protocol['folds'] if f['fold']==0)
 tr=set(fold['roles']['train']['patient_ids']);va=set(fold['roles']['validation']['patient_ids']);te=set(fold['roles']['test']['patient_ids']);require(tr.isdisjoint(va) and te.isdisjoint(tr|va),'Protocol patient conflict')
 holds=bounded_json(checked(root,'release-holds.json',prep['derivation']['files']['release-holds.json']));require(set(HOLD_IDS).issubset(holds['unresolved']),'Holds missing')
 prepared=[];total=0
 for s in pool:
  deadline();sid=s['scan_id'];require(s['status']=='ok','Pool scan failed; no replacement')
  role='validation' if s['split']=='val' else 'train';detail=fold['roles'][role];require(s['series_uid'] in detail['series_uids'] and s['patient_id'] in detail['patient_ids'] and s['patient_grouped_subset'] in detail['subsets'],'Protocol role mismatch')
  require(all(not r.get('mask_path') for r in s['positive']+s.get('ignored',[])),'External reference masks forbidden')
  require(bounded_json(checked(root,sid+'/record.json',prep['artifacts'][sid+'/record.json']))==s,'Record differs')
  cp=bounded_json(checked(root,sid+'/checkpoint.json',prep['artifacts'][sid+'/checkpoint.json']));require(cp['scan_id']==sid and cp['preparation_key']==prep['preparation_key'],'Checkpoint ancestry mismatch')
  expected=prep['artifacts'][sid+'/proposals.jsonl'];require(cp['files']['proposals.jsonl']==expected,'Proposal binding mismatch');count=cp['proposal_count']
  require(type(count) is int and 0<=count<=RULE['proposal_stream_cap_per_scan'],'Per-stream cap/count');total+=count;prepared.append((s,expected,count))
 require(total<=RULE['proposal_stream_cap_total'],'Aggregate inspection stream cap')
 all_entries=[];inspection=[]
 for s,expected,count in prepared:
  sid=s['scan_id'];picked,counts=select_rows(root/sid/'proposals.jsonl',s,expected['sha256'],count,deadline)
  eligible=counts['positive']>=1 and counts['negative']>=1
  inspection.append({'scan_id':sid,'patient_id':s['patient_id'],'role':s['split'],'scan_rank':rank_scan(sid),'proposal_count':count,'proposal_sha256':expected['sha256'],'counts':counts,'eligible':eligible,'selected':False,'reason':'not_selected_rank' if eligible else 'insufficient_classes'})
  observe({**inspection[-1],'verified_rows':copy.deepcopy(picked),'preparation_sha256':expected_preparation});all_entries.append({'scan':s,'selected':picked,'counts':counts,'proposal_sha256':expected['sha256'],'proposal_count':count,'arrays':{n:prep['artifacts'][sid+'/'+n+'.npy'] for n in ('image','valid')}})
 entries=[]
 require(len(inspection)==18,'Incomplete fixed pool')
 for actual,recorded in zip(inspection,CONTRACT['pool']):
  require(all(actual[k]==recorded[k] for k in ('scan_id','patient_id','role','scan_rank','proposal_count','proposal_sha256','counts','eligible')),'Fixed inspection receipt/count drift')
 for expected in CONTRACT['fixed']:
  entry=next(x for x in all_entries if x['scan']['scan_id']==expected['scan_id'])
  require(entry['counts']==expected['counts'],'Fixed selected counts changed')
  if entry['scan']['split']=='train':entry['selected']=[r for r in entry['selected'] if r['label']==1]
  require(sum(r['label']==1 for r in entry['selected'])==expected['counts']['positive'],'Exact positive count drift')
  require(sum(r['label']==0 for r in entry['selected'])==(0 if entry['scan']['split']=='train' else 32),'Exact negative count drift')
  entries.append(entry)
 require(sum(len(x['selected']) for x in entries if x['scan']['split']=='train')==5,'Exactly five training rows required')
 require(sum(len(x['selected']) for x in entries if x['scan']['split']=='val')==68,'Exactly 68 validation rows required')
 for x in inspection:
  if x['scan_id'] in {e['scan']['scan_id'] for e in entries}:x.update(selected=True,reason='fixed_fp32_identity')
 require(sha256(scans_path)==prep['files']['scans.jsonl'],'Scan metadata changed during selection')
 return {'schema':SCHEMA,'rule':RULE,'preparation_sha256':expected_preparation,'root':str(root),'entries':entries,'inspection':inspection,'skipped_same_patient':excluded,'inspection_total_rows':total,'reference_release':'HOLD','validation_scope':'two label-guided development scans only; NOT full-cohort validation','label_guided_selection':True,'parent_config':prep['config'],'config':effective_config(prep['config'])}

def verify_selection(selection):
 require(selection.get('schema')==SCHEMA and selection.get('rule')==RULE and selection.get('reference_release')=='HOLD','Positive selection rule changed')
 require(make_selection(selection['root'],selection['preparation_sha256'])==selection,'Positive selection differs from prespecified rule');return selection

def labelled_rows(entries):
 from .references import candidate_label
 ordered=[];statuses=[]
 for entry in entries:
  s=entry['scan'];counts={0:0,1:0};seen=set()
  for r in entry['selected']:
   require(r['ordinal'] not in seen,'Duplicate selected ordinal');seen.add(r['ordinal']);label=int(candidate_label(r['row']['center_lps'],s['positive'],s.get('ignored',[])))
   require(label in [0,1] and r['label']==label and r['rank']==rank_row(s['scan_id'],label,r['ordinal']),'Changed/ignored selected label or rank')
   require(r['payload_sha256']==canonical_hash(r['row']) and len(r['raw_sha256'])==64,'Row payload hash changed');require(r['row']['id']==s['scan_id']+':candidate:'+format(r['ordinal'],'06d') and r['row']['scan_id']==s['scan_id'],'Row identity changed');counts[label]+=1;ordered.append((label,rank_scan(s['scan_id']),s['scan_id'],r['rank'],r['ordinal'],{**r['row'],'label':label,'source_ordinal':r['ordinal'],'row_rank':r['rank'],'payload_sha256':r['payload_sha256'],'raw_sha256':r['raw_sha256'],'proposal_sha256':entry['proposal_sha256']}))
  expected=next((x for x in CONTRACT['fixed'] if x['scan_id']==s['scan_id']),None);require(expected is not None and (s['patient_id'],s['split'])==(expected['patient_id'],expected['role']),'Fixed scan identity drift');require(counts[1]==expected['counts']['positive'] and counts[0]==(0 if s['split']=='train' else 32),'FP32 selected class counts')
  statuses.append({'scan_id':s['scan_id'],'selected':len(entry['selected']),'usable':len(entry['selected']),'positive':counts[1],'negative':counts[0],'ignored_ordinals':[],'status':'LABELLED'})
 ordered.sort(key=lambda x:(-x[0],x[1],x[2],x[3],x[4]));return [x[-1] for x in ordered],statuses

def scoped_disposition(path,expected_hash,selection_hash,source_hash):
 p=regular(path);require(sha256(p)==expected_hash,'Disposition hash mismatch');d=bounded_json(p)
 require(d.get('scope')=='fp32_five_positive_exploratory_engineering' and d.get('reference_release')=='HOLD','Wrong positive disposition scope')
 require(d.get('preparation_sha256')==PREPARATION and d.get('selection_sha256')==selection_hash and d.get('source_binding_sha256')==source_hash,'Disposition binding mismatch')
 require(d.get('rule')==RULE and d.get('limits')==LIMITS,'Disposition policy/limits mismatch')
 require(d.get('permitted_label_use')=='engineering_loss_only' and d.get('label_guided_selection_acknowledged') is True,'Label-guided scope not acknowledged')
 require(all(isinstance(d.get(k),str) and d[k].strip() for k in ('decision_by','date','conditions','decision_text')),'Missing factual scoped decision')
 require(d.get('fp32_not_amp_coverage_acknowledged') is True,'FP32 diagnostic acknowledgment missing');require(set(HOLD_IDS).issubset(d.get('acknowledged_unresolved_holds',[])),'Holds missing');return d

def run_pass(selection,config,out,device_name,binding,phase=lambda name:None,model_factory=None):
 """One fixed pass, no experiment selection. Callable synthetic seam, no CLI bypass."""
 import numpy as np,torch
 from torch.utils.data import DataLoader
 from .data import CandidateDataset
 from .models import DetectorClassifier
 from .provenance import seed_all
 from .training import parameter_groups,epoch_lr,predict,_record_validation
 from .optimization import new_update_state,POLICY,append_diagnostic
 from .fp32_optimization import train_fp32_epoch as train_epoch,coverage_summary
 out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();status=[]
 try:
  require(selection.get('schema')==SCHEMA and selection['rule']==RULE and len(selection['entries'])==6,'Invalid selected scope');verify_scan_identities([x['scan'] for x in selection['entries']]);require(all(len(x['selected'])<=64 for x in selection['entries']),'Row cap changed');verify_config(selection,config);require(canonical_hash(config)==binding['config_hash'],'Configuration changed');require(binding.get('amp') is False and binding.get('parent_config_hash')==canonical_hash(selection['parent_config']) and binding.get('engineering_policy_sha256')==canonical_hash(RULE),'FP32 checkpoint identity changed')
  require(config['training']['microbatch']==2 and config['training']['accumulation']==8 and config['training']['alignment_weight']==0,'Training policy changed')
  prepare_arrays(selection);seed_all(17);device=torch.device(device_name)
  model=(model_factory or DetectorClassifier)(config['model']).to(device)
  groups=parameter_groups(model,config['training'])
  for g in groups:g['lr']=epoch_lr(g['lr'],0,config['training'])
  opt=torch.optim.AdamW(groups);amp=False;scaler=torch.amp.GradScaler('cuda',enabled=amp);state=new_update_state()
  train=[e for e in selection['entries'] if e['scan']['split']=='train'];val=[e for e in selection['entries'] if e['scan']['split']=='val']
  require(len(train)==4 and len(val)==2,'Role count mismatch');train_rows,status=labelled_rows(train);require(len(train_rows)==5 and all(r['label']==1 for r in train_rows),'Exactly five positive training rows required');write_json(out/'training-label-status.json',status)
  dataset=CandidateDataset([e['scan'] for e in train],train_rows,config['model']['patch_size'],True,17)
  loader=DataLoader(dataset,batch_size=2,shuffle=False,num_workers=0)
  def snapshot(stage):return {'stage':stage,'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'update_state':copy.deepcopy(state),'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,'selection_sha256':binding['selection_sha256'],'config':config,'rows_consumed':len(train_rows) if stage=='TRAINED' else 0}
  publish_checkpoint(out,snapshot('INITIALIZED'),binding);training_start=time.monotonic();phase('training');model.train()
  loss=train_epoch(model,loader,opt,scaler,torch.tensor(config['training']['class_weights'],device=device,dtype=torch.float32),device,amp,8,config['training']['clip_grad_norm'],state,lambda event:append_diagnostic(out/'boundaries.jsonl',event),0,ordered_rows=train_rows)
  require(math.isfinite(loss) and state['optimizer_updates']==1 and state['boundaries']==1 and state['skipped_updates']==0,'Wrong FP32 update count')
  publish_checkpoint(out,snapshot('TRAINED'),binding)
  for s in status:s['status']='COMPLETE_ONE_PASS'
  write_json(out/'training-scan-status.json',status);validation_start=time.monotonic();phase('validation')
  val_scans=[e['scan'] for e in val];val_rows=[row for entry in val for row in labelled_rows([entry])[0]]
  predictions,statuses=predict(model,val_scans,val_rows,config['model']['patch_size'],device,2)
  _record_validation(out,'engineering_validation',val_scans,predictions,statuses)
  from .references import candidate_label
  byid={s['scan_id']:s for s in val_scans};validation_losses=[];ignored=0;class_losses={'positive':[],'negative':[]}
  for r in predictions:
   s=byid[r['scan_id']];label=candidate_label(r['center_lps'],s['positive'],s.get('ignored',[]))
   if label==-1:ignored+=1;continue
   value=torch.nn.functional.cross_entropy(torch.tensor([r['logits']],dtype=torch.float32),torch.tensor([label]),weight=torch.tensor(config['training']['class_weights'],dtype=torch.float32),reduction='sum').item();require(math.isfinite(value),'Nonfinite validation loss');validation_losses.append(value);class_losses['positive' if label==1 else 'negative'].append(value)
  restored=load_checkpoint(out,binding);require(restored['state']['update_state']==state,'Checkpoint counters differ')
  coverage=coverage_summary([x for _,x in rows(out/'boundaries.jsonl',64)]);require(class_losses['positive'],'No positive validation losses')
  result={'positive_coverage':coverage,'validation_class_losses':class_losses,'status':'PASS','scope':'FP32 five-positive diagnostic only; NOT AMP coverage or full-cohort validation','train_loss':loss,'validation_weighted_loss_sums':validation_losses,'validation_ignored':ignored,'validation_scan_status':statuses,'update_state':state,'checkpoint_commit_sha256':restored['commit_sha256'],'phase_seconds':{'setup':training_start-started,'training':validation_start-training_start,'validation':time.monotonic()-validation_start},'reference_release':'HOLD'}
  if device.type=='cuda':result.update(max_allocated=torch.cuda.max_memory_allocated(),max_reserved=torch.cuda.max_memory_reserved())
  require(len(predictions)==68 and len(validation_losses)==68 and ignored==0,'Incomplete FP32 validation rows');write_json(out/'validation-completion.json',{'status':'COMPLETE','expected_rows':68,'completed_rows':68,'scan_status':statuses,'checkpoint_stage':'TRAINED','checkpoint_commit_sha256':restored['commit_sha256'],'reference_release':'HOLD'});write_json(out/'engineering-result.json',result);phase('complete');return result
 except BaseException as ex:
  write_json(out/'engineering-failure.json',{'status':'INCOMPLETE','error':repr(ex),'training_scan_status':status,'no_automatic_resume':True});raise
