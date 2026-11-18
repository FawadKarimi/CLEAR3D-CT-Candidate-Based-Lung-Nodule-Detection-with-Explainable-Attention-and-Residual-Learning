"""Metadata-only admission for the fixed HOLD engineering study; no approval flags."""
import copy
from pathlib import Path
from . import medicalnet_contract as base

require, read, digest, canonical = base.require, base.read, base.digest, base.canonical
SCOPE = 'medicalnet-three-epoch-engineering-v1'
GENERATED = 'medicalnet-generated-epoch-resume-v1'
SAMPLING = {'policy':'positive-cycle-negative-hash-v1','positive_draws':240,
            'negative_draws':720,'max_source_rows':283200,'max_epoch_rows':65536}
GENERATED_SAMPLING = {**SAMPLING,'positive_draws':3,'negative_draws':2}
QUAL_LIMITS = {**base.LIMITS,'total_seconds':120,'setup_seconds':120,
               'run_seconds':120,'validation_seconds':120}
RESEARCH_LIMITS = {**base.LIMITS,'total_seconds':780,'run_seconds':780}
BUDGET = {'total_seconds':1800,'generated_per_depth':120,'research_per_depth':780,
          'depth_order':[18,10],'no_budget_transfer':True}

def config(depth, checkpoint, provenance, provenance_hash, diagnostics=False):
    from .mixed_engineering import CONTRACT
    c = copy.deepcopy(CONTRACT['config'])
    c['training'].update(max_epochs=3,warmup_epochs=1,patience=3,seeds=[17])
    c['training_sampling'] = copy.deepcopy(SAMPLING)
    c['medicalnet'] = dict(depth=depth,checkpoint=str(checkpoint),checkpoint_sha256=base.WEIGHTS[depth],
                          provenance=str(provenance),provenance_sha256=provenance_hash)
    if diagnostics:
        from .diagnostics_policy import POLICY
        c['diagnostics']=copy.deepcopy(POLICY)
    return c

def check_config(c):
    m=c['medicalnet']
    require(c==config(m['depth'],m['checkpoint'],m['provenance'],m['provenance_sha256'],diagnostics='diagnostics' in c),
            'Scoped configuration changed')
    return c

def disposition(d,binding_hash,configs):
    from .engineering import HOLD_IDS
    require(d.get('status')=='ADOPTED' and d.get('scope')==SCOPE,'Separate adopted epoch disposition required')
    expected={'source_binding_sha256':binding_hash,'configuration_sha256':configs,
              'selection_sha256':base.SELECTION,'preparation_sha256':base.PREPARATION,
              'sampling':SAMPLING,'budget':BUDGET,'limits':RESEARCH_LIMITS,
              'reference_release':'HOLD','endpoint':'final_epoch_2_loss_diagnostics_only'}
    require(all(d.get(k)==v for k,v in expected.items()),'Disposition identity/policy mismatch')
    for key in ('label_guided_sampling','prior_validation_exposure','pretraining_overlap_uncertain',
                'sampled_monitoring','not_scientific_approval','fp32_not_amp_coverage'):
        require(d.get(key) is True,'Missing acknowledgement: '+key)
    require(set(HOLD_IDS).issubset(d.get('acknowledged_unresolved_holds',[])),'Six holds missing')
    require(all(isinstance(d.get(k),str) and d[k].strip() for k in ('supplied_by','instruction','date')),
            'Actual adoption attribution absent')
    return d

def identity(c,source,run_id,mode,disposition_hash=None):
    m=c['medicalnet']; generated=mode=='generated'
    return {'schema':'scoped-epoch-identity-v1','scope':GENERATED if generated else SCOPE,
            'run_id':run_id,'source_binding_sha256':source,'config_hash':canonical(c),
            'sampling_sha256':canonical(GENERATED_SAMPLING if generated else SAMPLING),
            'selection_sha256':canonical({'fixture_seed':717,'train_labels':[1,0,1,0,1],
                                          'shape':[1,48,48,48]}) if generated else base.SELECTION,
            'preparation_sha256':None if generated else base.PREPARATION,
            'depth':m['depth'],'checkpoint_sha256':m['checkpoint_sha256'],
            'provenance_sha256':m['provenance_sha256'],'disposition_sha256':disposition_hash,
            'transfer':{**base.transfer(m['depth']),'no_resume':False,'execution_protocol':'scoped-epoch-commit-v1'},
            'seed':17,'precision':'FP32','endpoint':'final_epoch','reference_release':'HOLD'}

def accepted(result,mode,depth,source):
    require(result.get('status')=='PASS' and result.get('mode')==mode and result.get('depth')==depth
            and result.get('source_binding_sha256')==source,'Prior execution acceptance mismatch')
    require(result.get('all_owned_workers_reaped') is True and result.get('usable_owned_gpu_samples') is True,
            'Monitoring/cleanup acceptance absent')
    require(result.get('budget')==BUDGET and result.get('reference_release')=='HOLD','Budget/HOLD changed')
    if mode=='generated':require(result.get('exact_resume') is True,'Resume qualification absent')
    return result
