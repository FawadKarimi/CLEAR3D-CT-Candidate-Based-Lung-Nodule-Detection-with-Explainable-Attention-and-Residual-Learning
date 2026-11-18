"""Output-bound engineering entry; no production approval and no new selection."""
import copy,json,math
from pathlib import Path
from . import scoped_epochs_contract as old
from .scoped_epochs_contract import base,require,read,digest,canonical,SAMPLING,BUDGET,RESEARCH_LIMITS,QUAL_LIMITS
from .diagnostics_policy import POLICY
from .cap4000_labels import bounded_json,hash_file,LabelRows,DiskTrainingSample,checkpoint_identity
SCOPE='cap4000-fixed12-diagnostic-three-epoch-v1'
DATA=json.loads(Path(__file__).with_name('diagnostic_training_contract.json').read_text())
def check_config(cfg):
    v=copy.deepcopy(cfg);require(v.pop('diagnostic_data',None)==DATA,'Diagnostic input/output binding changed')
    old.check_config(v);require(v.get('diagnostics')==POLICY,'Reviewed diagnostics required');return cfg

def disposition(d,source,configs):
    from .engineering import HOLD_IDS
    expected={'scope':SCOPE,'source_binding_sha256':source,'configuration_sha256':configs,'diagnostic_data':DATA,
      'sampling':SAMPLING,'budget':BUDGET,'limits':RESEARCH_LIMITS,'reference_release':'HOLD','cap4000_policy':'NOT_ADOPTED',
      'endpoint':'final_epoch_2_loss_diagnostics_only','unique_training_positives':15}
    require(d.get('status')=='ADOPTED' and all(d.get(k)==v for k,v in expected.items()),'Separate output-bound training adoption required')
    require(set(HOLD_IDS).issubset(d.get('acknowledged_unresolved_holds',[])),'Six holds missing')
    for k in ('label_guided_sampling','prior_validation_exposure','pretraining_overlap_uncertain','sampled_monitoring','not_scientific_approval','fp32_not_amp_coverage','no_positive_coverage_improvement','cuda_instrumentation_unmeasured'):
        require(d.get(k) is True,'Missing acknowledgement '+k)
    require(all(isinstance(d.get(k),str) and d[k].strip() for k in ('supplied_by','instruction','date')),'Actual adoption attribution absent')
    return d

def identity(cfg,source,run_id,mode,disposition_hash=None):
    require(mode=='research','Diagnostic entry supports research only');check_config(cfg)
    v=old.identity(cfg,source,run_id,mode,disposition_hash);i=DATA['inputs']
    v=checkpoint_identity(v,i['label_selection']['sha256'],i['sidecar']['sha256'],i['validation']['sha256'],i['manifest']['sha256'],i['index']['sha256'],source)
    v.update(scope=SCOPE,selection_sha256=i['label_selection']['sha256'],parent_selection_sha256=i['parent_selection']['sha256'],
      draw_plan_sha256=[i['draw'+str(e)]['sha256'] for e in range(3)],diagnostics_sha256=canonical(POLICY),diagnostic_data_sha256=canonical(DATA),cap4000_policy='NOT_ADOPTED')
    return v

def verify_inputs():
    # Streaming bytes only; no proposal stream, label recomputation, or array read.
    for key,s in DATA['inputs'].items():
        p=Path(s['path']);require(p.stat().st_size==s['bytes'] and hash_file(p)==s['sha256'],'Input changed '+key)

def accepted(v,mode,depth,source):
    old.accepted(v,mode,depth,source)
    require(v.get('diagnostic_scope')==SCOPE,'Previous diagnostic scope mismatch')

def real_data():
    from .provenance import verify_scan_identities
    from . import mixed_engineering as e
    from .data import CandidateDataset
    i=DATA['inputs'];parent=bounded_json(i['parent_selection']['path']);val=bounded_json(i['validation']['path'])
    train=[x for x in parent['entries'] if x['scan']['split']=='train']
    require(len(train)==12 and len(val)==6 and val==[x for x in parent['entries'] if x['scan']['split']=='val'],'Fixed identity/validation changed')
    verify_scan_identities([x['scan'] for x in train+val])
    require(not ({x['scan']['patient_id'] for x in train}&{x['scan']['patient_id'] for x in val}),'Patient leakage')
    rows=LabelRows(i['sidecar']['path'],i['sidecar']['sha256'])
    try:
        require(len(rows)==47899,'Usable count drift')
        counts=dict(rows.db.execute('SELECT label,count(*) FROM labels GROUP BY label'))
        require(counts=={-1:101,0:47884,1:15},'Accepted class counts drift')
        require(set(x[0] for x in rows.db.execute('SELECT DISTINCT scan FROM labels'))=={x['scan']['scan_id'] for x in train},'Training identities drift')
        rows.draw_plans=[bounded_json(i['draw'+str(epoch)]['path']) for epoch in range(3)]
        for epoch in range(3):
            sample=DiskTrainingSample(rows,SAMPLING,17);sample.set_epoch(epoch)
        vr,_=e.labelled_rows(val)
        require(len(vr)==384 and sum(x['label'] for x in vr)==15,'Unchanged validation labels/count')
        e.prepare_arrays(parent) # Existing hash/geometry gate, only fixed 18 image/valid pairs.
        return rows,[x['scan'] for x in val],vr,lambda sample:CandidateDataset([x['scan'] for x in train],sample,48,True,17)
    except BaseException:rows.close();raise

def diagnostic_evidence(state):
    state=Path(state);p=state/'training-diagnostics.jsonl';require(p.stat().st_size<=POLICY['max_log_bytes'],'Diagnostic byte limit')
    counts={'class_losses':0,'optimizer_boundary':0,'complete_validation_diagnostics':0};updates=[];classes={0:0,1:0};validation=[]
    with p.open() as f:
        for n,line in enumerate(f):
            require(n<20000 and len(line)<=1048576,'Diagnostic admission');v=json.loads(line);kind=v['kind']
            if kind in counts:counts[kind]+=1
            if kind=='class_losses':
                for k in (0,1):
                    q=v['classes'][str(k)];classes[k]+=q['count']
                    require(math.isfinite(q['weighted_loss_sum']) and math.isfinite(q['unweighted_loss_sum']),'Nonfinite diagnostic loss')
            if kind=='optimizer_boundary':
                require(v['actual_updates']==1 and v['status']=='UPDATED' and v['counters']['skipped_updates']==0,'Per-boundary update mismatch')
                updates.append(v['counters']['optimizer_updates']);require(math.isfinite(v['gradient_norm_before_clip']) and v['gradient_norm_before_clip']>0,'Invalid gradient summary')
            if kind=='complete_validation_diagnostics':validation.append((v['epoch'],v['rows']))
    require(counts=={'class_losses':1440,'optimizer_boundary':180,'complete_validation_diagnostics':3} and updates==list(range(1,181)),'Incomplete diagnostics')
    require(classes=={0:2160,1:720} and validation==[(0,384),(1,384),(2,384)],'Diagnostic class/validation drift')
    for epoch in range(3):
        actual=read(state/f'sampling-{epoch}.json');plan=bounded_json(DATA['inputs']['draw'+str(epoch)]['path'])
        require(actual=={**plan['receipt'],'source_indices':plan['indices']},'Accepted draw plan changed')
    return {'counts':counts,'training_draws_by_class':classes,'validation':validation,'sha256':digest(p),'bytes':p.stat().st_size}
