"""Loss-only engineering epochs, shared by generated restart and fixed selection runs."""
import copy,math,random,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from . import scoped_epoch_checkpoints as cp
from . import scoped_epochs_contract as c
from .provenance import seed_all,write_json
from .medicalnet_finetuning import TrainingSample
from .training import parameter_groups,epoch_lr,_record_validation
from .optimization import train_epoch,new_update_state,append_diagnostic
from .indexed import TrackedLoader

class Rows(list):
    role='train'

class GeneratedDataset(Dataset):
    def __init__(self,rows):self.rows=rows;self.epoch=0
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];g=torch.Generator().manual_seed(717+r['fixture_index']+100*self.epoch)
        return {'image':torch.rand(1,48,48,48,generator=g),'label':r['label'],'index':i}

def generated_data():
    train=Rows({'id':f'generated-train-{i}','label':v,'fixture_index':i} for i,v in enumerate([1,0,1,0,1]))
    scans=[{'scan_id':'generated-val-a'},{'scan_id':'generated-val-b'}]
    val=[{'id':f'generated-val-{i}','scan_id':scans[i//2]['scan_id'],'label':i%2,'fixture_index':50+i} for i in range(4)]
    return train,scans,val

def generated_validate(model,scans,rows,device,cfg):
    predictions=[];statuses=[];model.eval()
    with torch.no_grad():
        for scan in scans:
            selected=[r for r in rows if r['scan_id']==scan['scan_id']];local=[];error=None
            try:
                x=torch.stack([GeneratedDataset([r])[0]['image'] for r in selected]).to(device)
                logits=model(x).float().cpu();c.require(torch.isfinite(logits).all().item(),'Nonfinite validation logits')
                local=[{'id':r['id'],'scan_id':scan['scan_id'],'logits':v.tolist()} for r,v in zip(selected,logits)]
            except Exception as ex:error=repr(ex);local=[]
            predictions.extend(local);statuses.append({'scan_id':scan['scan_id'],'status':'failed' if error else 'ok',
                'reason':error,'candidates':len(selected),'predictions':len(local)})
    return predictions,statuses

def real_data(selection):
    from . import mixed_engineering as e
    from .provenance import verify_scan_identities
    # Caller has reverified the exact selection and streams before this pixel gate.
    c.require(selection['preparation_sha256']==c.base.PREPARATION and len(selection['entries'])==18,'Selection scope')
    verify_scan_identities([x['scan'] for x in selection['entries']])
    train=[x for x in selection['entries'] if x['scan']['split']=='train'];val=[x for x in selection['entries'] if x['scan']['split']=='val']
    tr,_=e.labelled_rows(train);vr,_=e.labelled_rows(val)
    c.require(len(train)==12 and len(val)==6 and len(tr)==768 and sum(r['label'] for r in tr)==15
              and len(vr)==384 and sum(r['label'] for r in vr)==15,'Fixed population/count drift')
    e.prepare_arrays(selection)
    from .data import CandidateDataset
    train_scans=[x['scan'] for x in train]
    return Rows(tr),[x['scan'] for x in val],vr,lambda sample:CandidateDataset(train_scans,sample,48,True,17)

def real_validate(model,scans,rows,device,cfg):
    from .training import predict
    return predict(model,scans,rows,48,device,2)

def fit(cfg,binding,out,device,train_rows,val_scans,val_rows,dataset_factory,validator,
        model_factory,*,epochs,sampling,resume=None,stop_after=None,phase=lambda event:None,sampling_factory=None):
    """No production bundle or approval flag accepted. No scientific selection metrics."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    c.require(c.canonical(cfg)==binding['config_hash'],'Configuration binding mismatch')
    c.require(c.canonical(sampling)==binding['sampling_sha256'],'Sampling binding mismatch')
    from . import diagnostic_training as diagnostic
    if binding['scope']==diagnostic.SCOPE:
        diagnostic.check_config(cfg)
        c.require(sampling_factory is not None and binding.get('diagnostic_data_sha256')==diagnostic.canonical(diagnostic.DATA),'Diagnostic disk/input guard')
    c.require(binding['scope'] in (c.SCOPE,c.GENERATED,diagnostic.SCOPE) and binding['reference_release']=='HOLD','Scoped identity required')
    c.require(cfg['training']['amp'] is False,'FP32 only')
    if sampling_factory is not None:
        from .cap4000_labels import DiskTrainingSample, SCOPE as LABEL_SCOPE
        c.require(sampling_factory is DiskTrainingSample and binding.get('candidate_label_adapter')==LABEL_SCOPE,'Scoped disk sampler identity')
        for key in ('label_selection_sha256','label_sidecar_sha256','unchanged_validation_sha256','candidate_manifest_sha256','metadata_index_sha256','source_binding_sha256'):
            value=binding.get(key)
            c.require(isinstance(value,str) and len(value)==64 and all(x in '0123456789abcdef' for x in value),'Missing disk sampler binding '+key)
    write_json(out/'identity.json',binding)
    try:
        seed_all(17);device=torch.device(device);model=model_factory(cfg['model']).to(device)
        groups=parameter_groups(model,cfg['training']);bases=[g['lr'] for g in groups]
        opt=torch.optim.AdamW(groups);scaler=torch.amp.GradScaler('cuda',enabled=False);state=new_update_state()
        generator=torch.Generator().manual_seed(17);sample=(sampling_factory or TrainingSample)(train_rows,sampling,17)
        dataset=dataset_factory(sample);loader=DataLoader(dataset,batch_size=2,shuffle=True,generator=generator,num_workers=0)
        start=0;restored=cp.load(resume,binding) if resume else None
        def snapshot(epoch,validation=None):
            return {'stage':'INITIALIZED' if epoch<0 else 'TRAINED','epoch':epoch,'config':cfg,
                'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),
                'scheduler':{'epoch':epoch,'base_lrs':bases,'rule':'epoch_lr_v1'},'update_state':copy.deepcopy(state),
                'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
                'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,'loader_rng':generator.get_state(),
                'sampling_receipt':copy.deepcopy(sample.receipt),'validation_complete':validation is not None,
                'validation':validation}
        if restored:
            s=restored['state'];c.require(s['scheduler']=={'epoch':s['epoch'],'base_lrs':bases,'rule':'epoch_lr_v1'},'Resume scheduler changed')
            c.require(-1<=s['epoch']<epochs-1,'Resume must have remaining epochs')
            model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);scaler.load_state_dict(s['scaler'])
            state=copy.deepcopy(s['update_state']);start=s['epoch']+1;cp.publish(out,s,binding,imported=restored)
            # Restore after model/setup/import, all of which can consume randomness.
            torch.set_rng_state(s['torch_rng']);np.random.set_state(s['numpy_rng']);random.setstate(s['python_rng'])
            generator.set_state(s['loader_rng'])
            if device.type=='cuda':torch.cuda.set_rng_state_all(s['cuda_rng'])
            del s,restored
        else:cp.publish(out,snapshot(-1),binding)
        setup=time.monotonic()-started;training_seconds=validation_seconds=0.;reports=[]
        expected_updates=math.ceil(math.ceil(len(sample)/2)/cfg['training']['accumulation'])
        for epoch in range(start,epochs):
            sample.set_epoch(epoch);dataset.epoch=epoch
            write_json(out/f'sampling-{epoch}.json',{**sample.receipt,'source_indices':sample.indices})
            for group,base in zip(opt.param_groups,bases):group['lr']=epoch_lr(base,epoch,cfg['training'])
            model.train();before=state['optimizer_updates'];t=time.monotonic();phase({'phase':'training','epoch':epoch})
            events=[]
            def emit(v):events.append(v);append_diagnostic(out/'boundaries.jsonl',v)
            from .diagnostics import Diagnostics, POLICY as DIAGNOSTIC_POLICY
            diagnostic_config=cfg.get('diagnostics')
            if diagnostic_config is not None:c.require(diagnostic_config==DIAGNOSTIC_POLICY,'Diagnostic binding mismatch')
            diagnostics=Diagnostics(out/'training-diagnostics.jsonl') if diagnostic_config else None
            loss=train_epoch(model,TrackedLoader(loader,out/f'consumption-{epoch}.json',len(sample)),opt,scaler,
                torch.tensor(cfg['training']['class_weights'],device=device,dtype=torch.float32),device,False,
                cfg['training']['accumulation'],cfg['training']['clip_grad_norm'],state,emit,epoch,diagnostics=diagnostics)
            training_seconds+=time.monotonic()-t
            c.require(math.isfinite(loss) and state['optimizer_updates']-before==expected_updates
                      and state['skipped_updates']==0,'Finite exact update count required')
            c.require(all(v.get('norm_before_clip',0)>0 and not v.get('normalized_gradients',{}).get('nonfinite') for v in events),
                      'Finite nonzero normalized gradients required')
            c.require(all(torch.isfinite(p).all().item() for p in model.parameters()),'Nonfinite model parameters')
            c.require(all(float(v['step'])==state['optimizer_updates'] for v in opt.state.values()),'AdamW steps disagree')
            t=time.monotonic();phase({'phase':'validation','epoch':epoch})
            predictions,statuses=validator(model,val_scans,val_rows,device,cfg)
            _record_validation(out,f'validation-{epoch}',val_scans,predictions,statuses)
            ids={r['id']:r['label'] for r in val_rows}
            c.require(len(predictions)==len(ids)==len(val_rows) and {r['id'] for r in predictions}==set(ids),'Incomplete validation rows')
            losses={0:[],1:[]}
            for r in predictions:
                label=ids[r['id']];value=torch.nn.functional.cross_entropy(torch.tensor([r['logits']]),torch.tensor([label]),
                    weight=torch.tensor(cfg['training']['class_weights'],dtype=torch.float32),reduction='sum').item()
                c.require(math.isfinite(value),'Nonfinite validation loss');losses[label].append(value)
            validation={'complete':True,'rows':len(predictions),'scan_status':[{k:v for k,v in s.items() if k!='seconds'} for s in statuses],
                        'class_loss_sums':losses,'endpoint':'loss_diagnostics_only'}
            if diagnostics is not None:diagnostics.validation(predictions,ids,cfg['training']['class_weights'],epoch)
            validation_seconds+=time.monotonic()-t
            cp.publish(out,snapshot(epoch,validation),binding)
            report={'epoch':epoch,'loss':loss,'actual_updates':state['optimizer_updates']-before,'skips':0,
                    'validation_rows':len(predictions),'validation_classes':{str(k):len(v) for k,v in losses.items()}}
            write_json(out/f'epoch-{epoch}.json',report);reports.append(report)
            if stop_after==epoch:
                result={'status':'BOUNDARY_INTERRUPTED','epoch':epoch,'update_state':state,'epochs':reports}
                break
        else:result={'status':'PASS','endpoint_epoch':epochs-1,'update_state':state,'epochs':reports}
        loaded=cp.load(out,binding)
        result.update(checkpoint_commit_sha256=loaded['commit_sha256'],phase_seconds={'setup':setup,'training':training_seconds,'validation':validation_seconds},
                      reference_release='HOLD',no_best_epoch_selection=True)
        write_json(out/'epoch-result.json',result)
        return result
    except BaseException as error:
        write_json(out/'epoch-failure.json',{'error':repr(error),'reference_release':'HOLD','automatic_resume':False});raise

def exact(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b

def compare(full,resumed,binding):
    a=cp.load(full,binding)['state'];b=cp.load(resumed,binding)['state']
    checks={k:bool(exact(a[k],b[k])) for k in a}
    c.require(all(checks.values()),'Exact resumed checkpoint state mismatch')
    return {'status':'PASS','exact':True,'checks':checks,'tolerance':0,'checkpoint_load':'CPU',
            'excluded':'commit UUIDs, paths, timings and publication ancestry are distinct evidence, not tensor state'}
