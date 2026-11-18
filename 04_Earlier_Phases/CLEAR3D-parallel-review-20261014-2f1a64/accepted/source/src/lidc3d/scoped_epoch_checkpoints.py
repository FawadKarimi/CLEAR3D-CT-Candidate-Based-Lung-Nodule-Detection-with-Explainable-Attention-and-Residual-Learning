"""Immutable final-endpoint epoch state. No best model, CPM or production approval."""
import os,uuid
from pathlib import Path
import torch
from .checkpoints import _json_atomic,_sync_dir,_boundary
from .provenance import sha256,canonical_hash,write_json
from .scoped_epochs_contract import require,read

def load(out,binding):
    out=Path(out);p=read(out/'epoch-commit.json')
    require(p.get('status')=='COMMITTED','Incomplete epoch publication')
    rel=Path(p['generation'])
    require(len(rel.parts)==2 and rel.parts[0]=='epoch-checkpoints' and not rel.is_absolute(),'Invalid epoch generation')
    gen=out/rel
    require(not any(x.is_symlink() for x in [gen,*gen.parents]),'Symlink checkpoint')
    record=read(gen/'commit.json',p['commit_sha256'])
    require(record['schema']=='scoped-epoch-commit-v1' and record['status']=='COMMITTED'
            and record['binding']==binding,'Epoch binding mismatch')
    require(sha256(gen/'state.pt')==record['state_sha256'],'Checkpoint hash mismatch')
    state=torch.load(gen/'state.pt',map_location='cpu',weights_only=False)
    require(state['binding']==binding and canonical_hash(state['config'])==binding['config_hash'],
            'Checkpoint state identity mismatch')
    require(state['epoch']==record['epoch'] and state['stage']==('INITIALIZED' if state['epoch']==-1 else 'TRAINED'),
            'Checkpoint epoch/stage mismatch')
    from .optimization import validate_update_state
    validate_update_state(state['update_state'])
    require(state['scheduler']['epoch']==state['epoch'] and state['scheduler']['rule']=='epoch_lr_v1','Scheduler mismatch')
    require(state['epoch']==-1 or state.get('validation_complete') is True,'Unvalidated epoch cannot resume')
    return {'state':state,'record':record,'commit_sha256':p['commit_sha256'],'generation':str(gen)}

def publish(out,state,binding,imported=None):
    out=Path(out);previous=load(out,binding) if (out/'epoch-commit.json').exists() else None
    require(not(previous and imported),'Import requires a fresh checkpoint destination')
    parent=previous or imported
    if parent:
        require(state['epoch']==parent['state']['epoch']+(0 if imported else 1),'Epoch sequence changed')
    else:require(state['epoch']==-1,'First state must be INITIALIZED')
    require(state['stage']==('INITIALIZED' if state['epoch']==-1 else 'TRAINED'),'Invalid stage')
    require(state['epoch']==-1 or state.get('validation_complete') is True,'Incomplete validation')
    gen=out/'epoch-checkpoints'/uuid.uuid4().hex
    ancestry={'commit_sha256':parent['commit_sha256'],'generation':parent['generation']} if parent else None
    try:
        _json_atomic(out/'epoch-commit.json',{'status':'PUBLISHING','previous_committed':ancestry})
        _boundary('scoped_intent');gen.mkdir(parents=True,exist_ok=False);_sync_dir(gen.parent)
        with (gen/'state.pt').open('xb') as f:
            torch.save({**state,'binding':binding},f);f.flush();os.fsync(f.fileno())
        _boundary('scoped_state')
        record={'schema':'scoped-epoch-commit-v1','status':'COMMITTED','binding':binding,
                'epoch':state['epoch'],'state_sha256':sha256(gen/'state.pt'),'parent':ancestry}
        _json_atomic(gen/'commit.json',record);_boundary('scoped_sealed')
        _json_atomic(out/'epoch-commit.json',{'status':'COMMITTED','generation':gen.relative_to(out).as_posix(),
                                            'commit_sha256':sha256(gen/'commit.json')})
    except BaseException as error:
        write_json(out/('epoch-publication-failure-'+uuid.uuid4().hex+'.json'),
                   {'error':repr(error),'generation':str(gen),'previous_committed':ancestry})
        raise
