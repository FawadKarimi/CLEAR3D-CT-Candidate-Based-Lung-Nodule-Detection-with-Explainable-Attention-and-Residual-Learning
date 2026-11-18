"""Observation-only hooks around the unchanged weighted/AMP update loop."""
import copy,hashlib,math
import torch
from .optimization import train_epoch

def parameter_hash(model):
 h=hashlib.sha256()
 for name,p in model.named_parameters():
  if p.requires_grad:
   h.update((name+'|'+str(p.dtype)+'|'+str(tuple(p.shape))).encode());h.update(p.detach().reshape(-1).contiguous().cpu().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def optimizer_steps(model,optimizer):
 return {name:int(optimizer.state.get(p,{}).get('step',0)) for name,p in model.named_parameters() if p.requires_grad}
def qualifies(event):
 try:
  proof=event['optimizer_evidence'];labels=event['class_counts'];notes=event['class_microbatches'];micro=event['microbatches']
  if not proof or not notes or len(notes)!=len(micro):return False
  names=proof['gradient_parameter_names']
  return (event['status']=='UPDATED' and event['optimizer_updates']==1 and labels['positive']>0 and labels['negative']==0 and math.isfinite(event['norm_before_clip']) and event['norm_before_clip']>0 and event['normalized_gradients']['gradient_tensors']>0 and not event['normalized_gradients']['nonfinite'] and not event['unscaled_gradients']['nonfinite'] and bool(names) and all(proof['steps_after'][n]==proof['steps_before'][n]+1 for n in names) and proof['parameter_sha256_before']!=proof['parameter_sha256_after'] and proof['parameters_changed'] is True and proof['optimizer_step_advanced'] is True and sum(n['positive'] for n in notes)==labels['positive'] and all(n['negative']==0 and n['positive']==m['samples'] and len(n['rows'])==m['samples'] and all(r['label']==1 for r in n['rows']) and math.isfinite(n['positive_weighted_loss_sum']) and math.isfinite(m['weighted_loss_sum']) and m['scaled_loss_finite'] for n,m in zip(notes,micro)))
 except (KeyError,TypeError,ValueError):return False
def coverage_summary(events):
 boundaries=[x for x in events if x.get('kind')=='accumulation_boundary'];successful=[x for x in boundaries if qualifies(x)]
 if not successful:raise FloatingPointError('No verified positive-only optimizer update')
 return {'status':'PASS_POSITIVE_ONLY_UPDATE','successful_positive_only_boundaries':[x['last_microbatch'] for x in successful],'positive_only_skips':sum(x.get('status')=='SKIPPED_AMP_NONFINITE' and x.get('class_counts',{}).get('negative')==0 for x in boundaries),'not_model_quality':True}

def train_positive_epoch(model,loader,optimizer,scaler,weights,device,amp,accumulation,max_norm,state,emit,epoch,*,ordered_rows):
 if not isinstance(optimizer,torch.optim.AdamW):raise ValueError('Reviewed AdamW required')
 notes=[];pending={};events=[];handles=[];offset=[0]
 class ObservedLoader:
  def __len__(self):return len(loader)
  def __iter__(self):
   for step,batch in enumerate(loader):
    indices=batch['index'].detach().cpu().tolist();labels=batch['label'].detach().cpu().tolist();expected=ordered_rows[offset[0]:offset[0]+len(labels)]
    if indices!=list(range(offset[0],offset[0]+len(labels))) or labels!=[r['label'] for r in expected]:raise ValueError('Row/label consumption changed')
    if any(x not in (0,1) for x in labels):raise ValueError('Ignored label entered loss')
    notes.append({'step':step,'rows':[{'id':r['id'],'scan_id':r['scan_id'],'label':r['label']} for r in expected],'positive':labels.count(1),'negative':labels.count(0)})
    offset[0]+=len(labels);yield batch
 def forward_observer(module,args,logits):
  n=notes[-1];y=torch.tensor([r['label'] for r in n['rows']],device=logits.device,dtype=torch.long)
  losses=torch.nn.functional.cross_entropy(logits.detach(),y,weight=weights,reduction='none')
  n['positive_weighted_loss_sum']=float(losses[y==1].sum());n['negative_weighted_loss_sum']=float(losses[y==0].sum())
  if not bool(torch.isfinite(losses).all()):raise FloatingPointError('Nonfinite class diagnostic loss')
 def before_step(opt,args,kwargs):
  pending.clear();pending.update(parameter_sha256_before=parameter_hash(model),steps_before=optimizer_steps(model,opt),gradient_parameter_names=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is not None])
 def after_step(opt,args,kwargs):
  pending.update(parameter_sha256_after=parameter_hash(model),steps_after=optimizer_steps(model,opt))
  pending['parameters_changed']=pending['parameter_sha256_before']!=pending['parameter_sha256_after']
  pending['optimizer_step_advanced']=bool(pending['gradient_parameter_names']) and all(pending['steps_after'][n]==pending['steps_before'][n]+1 for n in pending['gradient_parameter_names'])
 def observed_emit(event):
  event=copy.deepcopy(event)
  if event['kind']=='accumulation_boundary':
   selected=[notes[m['step']] for m in event['microbatches']];event['class_microbatches']=copy.deepcopy(selected)
   event['class_counts']={k:sum(m[k] for m in selected) for k in ['positive','negative']}
   event['optimizer_evidence']=copy.deepcopy(pending) if event['optimizer_updates'] else None
   event['positive_only_verified']=qualifies(event);pending.clear();events.append(event)
  emit(event)
 try:
  handles=[model.register_forward_hook(forward_observer),optimizer.register_step_pre_hook(before_step),optimizer.register_step_post_hook(after_step)]
  loss=train_epoch(model,ObservedLoader(),optimizer,scaler,weights,device,amp,accumulation,max_norm,state,observed_emit,epoch)
  if offset[0]!=len(ordered_rows):raise ValueError('Incomplete selected-row consumption')
  coverage_summary(events);return loss
 except BaseException as ex:
  emit({'kind':'positive_coverage_failure','error':repr(ex),'rows_consumed':offset[0],'expected_rows':len(ordered_rows),'class_microbatches':notes});raise
 finally:
  for h in handles:h.remove()
