"""Stdlib-only reanalysis of immutable accepted scalar evidence. No Torch/data reads."""
import argparse, collections, hashlib, json, math, pathlib, statistics

def main():
 p=argparse.ArgumentParser();p.add_argument('--accepted',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
 root=a.accepted; stage=root/'runs/diagnostic-epochs18-001/research'; hashes={}
 def read(path):
  raw=path.read_bytes();hashes[str(path.relative_to(root))]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)};return raw
 def obj(path):return json.loads(read(path))
 def lines(path):return [json.loads(x) for x in read(path).splitlines()]
 rows=lines(stage/'state/training-diagnostics.jsonl');samples=lines(stage/'samples.jsonl');phases=lines(stage/'epoch-phases.jsonl')
 acc=obj(stage.parent/'acceptance.json');wd=obj(stage/'watchdog-result.json');rec=obj(root/'reconciliation.json');config=obj(root/'configuration-resnet18.json')
 counts=collections.Counter(r['kind'] for r in rows);classes={};acts={};bounds=[]
 def stats(v):return {'count':len(v),'min':min(v) if v else None,'max':max(v) if v else None,'mean':statistics.fmean(v) if v else None,'median':statistics.median(v) if v else None}
 for r in rows:
  if r['kind']=='class_losses':
   for label,z in r['classes'].items():
    if not z['count']:continue
    q=classes.setdefault(f"{r['epoch']}:{label}",{'count':0,'unweighted_loss_sum':0.,'weighted_loss_sum':0.,'p_sum':0.,'p2_sum':0.,'p_min':1.,'p_max':0.})
    n=z['count'];s=z['positive_probability'];q['count']+=n
    for k in ('unweighted_loss_sum','weighted_loss_sum'):q[k]+=z[k]
    q['p_sum']+=s['mean']*n;q['p2_sum']+=s['rms']**2*n;q['p_min']=min(q['p_min'],s['min']);q['p_max']=max(q['p_max'],s['max'])
  elif r['kind']=='activation_sample':
   for i,s in enumerate(r['statistics']):
    key=f"{r['epoch']}:{r['module']}:{i}";q=acts.setdefault(key,{'summaries':0,'nonfinite':0,'all_zero_summaries':0,'rms':[],'means':[],'sample_ranges':[]})
    q['summaries']+=1;q['nonfinite']+=s['nonfinite_count'];q['all_zero_summaries']+=s['zeros']==s['sample_count'];q['rms'].append(s['rms']);q['means'].append(s['mean']);q['sample_ranges'].append(s['max']-s['min'])
  elif r['kind']=='optimizer_boundary':bounds.append(r)
 for q in classes.values():
  n=q['count'];q['p_mean']=q['p_sum']/n;q['p_sd']=math.sqrt(max(0,q['p2_sum']/n-q['p_mean']**2));q['unweighted_loss_mean']=q['unweighted_loss_sum']/n
 for q in acts.values():
  for k in ('rms','means','sample_ranges'):q[k]=stats(q[k])
 epochs={}
 for ep in range(3):
  b=[x for x in bounds if x['epoch']==ep];families={}
  for x in b:
   for fam,z in x['families_sampled'].items():
    q=families.setdefault(fam,{'changed_values_across_boundaries':0,'sampled_values_across_boundaries':0,'boundaries_with_change':0,'relative_sample_delta':[]})
    q['changed_values_across_boundaries']+=z['changed_values'];q['sampled_values_across_boundaries']+=z['sample_count'];q['boundaries_with_change']+=z['changed_values']>0
    if z['before_squared_sum']>0:q['relative_sample_delta'].append(math.sqrt(z['delta_squared_sum']/z['before_squared_sum']))
  for q in families.values():q['relative_sample_delta']=stats(q['relative_sample_delta'])
  epochs[str(ep)]={'boundaries':len(b),'actual_updates':sum(x['actual_updates'] for x in b),'clipped':sum(x['clipping_applied'] for x in b),'grad_before':stats([x['gradient_norm_before_clip'] for x in b]),'grad_after':stats([x['gradient_norm_after_clip'] for x in b]),'denominator':stats([x['denominator'] for x in b]),'groups':b[0]['groups'] if b else [],'families':families}
 intervals=[]
 for i,r in enumerate(phases[:-1]):intervals.append({**r,'to_next_phase_seconds':phases[i+1]['monotonic']-r['monotonic']})
 gaps=[b['monotonic']-a['monotonic'] for a,b in zip(samples,samples[1:])]
 external={}
 for s in samples:
  for proc in s['gpu']['processes']:
   if proc['pid']!=s['owned_worker']['pid']:external.setdefault(str(proc['pid']),[]).append(proc['gpu_bytes'])
 result={'scope':'Existing scalar evidence only; no new arrays, Torch, CUDA or training','input_hashes':hashes,'diagnostic_counts':dict(counts),'classes':classes,'activations':acts,'epochs':epochs,'phase_intervals':intervals,'initial_observed_setup_seconds':phases[0]['monotonic']-samples[0]['monotonic'],'observed_partial_epoch2_seconds_to_last_sample':samples[-1]['monotonic']-phases[-1]['monotonic'],'resource_samples':len(samples),'sample_gap_seconds':stats(gaps),'shared_process_gpu_bytes':{k:stats(v) for k,v in external.items()},'watchdog':wd,'controller_seconds':acc['wall_seconds'],'shell_seconds':rec['backbones']['18']['shell']['elapsed_seconds'],'weighted_constant_predictor':{'training_positive_draws':240,'training_negative_draws':720,'positive_weight':3,'negative_weight':1,'constant_optimum_positive_probability':0.5,'normalized_loss_at_optimum':math.log(2),'limitation':'Whole-epoch constant-predictor reference; group normalization and shuffled class proportions mean not an exact stationary point of every minibatch.'},'limitations':['Activation channel/spatial summaries are not per-input distances.','Training probability variation includes dropout, augmentation and temporal model changes.','Parameter samples are repeated observations, not unique parameters or complete norms; weight decay can change parameters without data gradient.','Phase intervals include checks or publication between phase markers; no fine-grained I/O or GPU-compute timings.','No matched diagnostics-off measurement, GPU utilization trace or full tensor comparison.','Epoch2 includes two forward rows beyond completed boundaries.','No validation-based policy selection.']}
 assert counts['optimizer_boundary']==153 and counts['class_losses']==1225
 assert sum(q['count'] for q in classes.values())==2450
 assert all(r['counters']['skipped_updates']==0 for r in bounds)
 assert all(r['counters']['optimizer_updates']==i+1 for i,r in enumerate(bounds))
 assert all(q['nonfinite']==0 for q in acts.values())
 a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps({'status':'PASS','rows':len(rows),'updates':len(bounds),'output':str(a.output)}))
if __name__=='__main__':main()
