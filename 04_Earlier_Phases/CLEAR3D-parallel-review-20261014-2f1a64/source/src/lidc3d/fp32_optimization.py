"""Strict five-row diagnostic wrapper; unchanged weighted optimizer core."""
import copy,math
import torch
from . import positive_optimization as parent
from .optimization import new_update_state

def coverage_summary(events):
 result=parent.coverage_summary(events)
 boundaries=[e for e in events if e.get('kind')=='accumulation_boundary']
 if len(boundaries)!=1:raise ValueError('Exactly one FP32 boundary required')
 e=boundaries[0]
 if not (parent.qualifies(e) and e['amp'] is False and e['denominator']==15 and e['scale_before']==e['scale_after']==1 and e['incomplete_group'] is True and e['last_microbatch']==2 and [x['samples'] for x in e['microbatches']]==[2,2,1] and e['class_counts']=={'positive':5,'negative':0}):raise ValueError('FP32 boundary contract changed')
 ids=[(r['scan_id'],r['id']) for m in e['class_microbatches'] for r in m['rows']]
 if len(ids)!=5 or len(set(ids))!=5:raise ValueError('Duplicate FP32 consumption')
 proof=e['fp32_evidence']
 if not (proof['model_dtype']=='float32' and proof['input_dtypes']==['float32']*3 and proof['output_dtypes']==['float32']*3 and proof['scaler_enabled'] is False and proof['post_clip_finite'] is True and math.isfinite(proof['post_clip_norm']) and proof['post_clip_norm']>0):raise ValueError('FP32 precision/gradient evidence incomplete')
 return {**result,'status':'PASS_FP32_FIVE_POSITIVE_UPDATE','actual_updates':1,'skipped_updates':0,'rows':5,'denominator':15,'not_amp_coverage':True}

def train_fp32_epoch(model,loader,optimizer,scaler,weights,device,amp,accumulation,max_norm,state,emit,epoch,*,ordered_rows):
 events=[];handles=[];proof={'model_dtype':'float32','input_dtypes':[],'output_dtypes':[],'scaler_enabled':bool(scaler.is_enabled()),'row_identities':copy.deepcopy(ordered_rows)}
 def check_precision():
  if amp is not False or scaler.is_enabled() or scaler.get_scale()!=1:raise ValueError('FP32 only: AMP/scaler enabled')
  if torch.is_autocast_enabled(device.type):raise ValueError('FP32 only: external autocast enabled')
  if any(t.is_floating_point() and t.dtype!=torch.float32 for t in list(model.parameters())+list(model.buffers())):raise ValueError('FP32 model dtype required')
 try:
  check_precision()
  if epoch!=0 or accumulation!=8 or len(loader)!=3 or len(ordered_rows)!=5 or any(r['label']!=1 for r in ordered_rows) or len({(r['scan_id'],r['id']) for r in ordered_rows})!=5:raise ValueError('Five unique FP32 rows required')
  if weights.dtype!=torch.float32 or weights.tolist()!=[1.,3.]:raise ValueError('Reviewed class weights [1,3] required')
  if state!=new_update_state():raise ValueError('Fresh FP32 update state required')
  def pre(module,args):
   check_precision();x=args[0]
   if x.dtype!=torch.float32:raise ValueError('FP32 input dtype required')
   expected=[2,2,1];i=len(proof['input_dtypes'])
   if i>=3 or x.shape[0]!=expected[i]:raise ValueError('FP32 microbatch consumption changed')
   proof['input_dtypes'].append('float32')
  def post(module,args,out):
   if out.dtype!=torch.float32:raise ValueError('FP32 output dtype required')
   proof['output_dtypes'].append('float32')
  def clipped(opt,args,kwargs):
   grads=[p.grad.detach() for p in model.parameters() if p.grad is not None]
   finite=bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
   norm=math.sqrt(sum(float(g.double().square().sum()) for g in grads)) if finite else float('nan')
   proof.update(post_clip_finite=finite,post_clip_norm=norm)
   if not finite or not math.isfinite(norm) or norm<=0:raise FloatingPointError('No finite nonzero clipped gradient')
  def record(event):
   event=copy.deepcopy(event)
   if event.get('kind')=='accumulation_boundary':event['fp32_evidence']=copy.deepcopy(proof)
   events.append(event);emit(event)
  handles=[model.register_forward_pre_hook(pre),model.register_forward_hook(post),optimizer.register_step_pre_hook(clipped)]
  loss=parent.train_positive_epoch(model,loader,optimizer,scaler,weights,device,False,8,max_norm,state,record,0,ordered_rows=ordered_rows)
  coverage_summary(events)
  if state['boundaries']!=1 or state['optimizer_updates']!=1 or state['skipped_updates']!=0:raise ValueError('FP32 counters changed')
  return loss
 except BaseException as ex:
  emit({'kind':'fp32_diagnostic_failure','error':repr(ex),'precision_evidence':copy.deepcopy(proof),'expected_rows':5,'no_retry':True});raise
 finally:
  for h in handles:h.remove()
