"""Read-only, bounded scalar diagnostics. Sampled changes are not full norms."""
import math
import torch
from .optimization import append_diagnostic
from .diagnostics_policy import POLICY

def family(name):
    return 'encoder' if name.startswith('encoder.') else 'transformer' if name.startswith('transformers.') else 'head_bridge_other'

def sample(t,n):
    t=t.detach();count=min(t.numel(),n)
    if not count:return []
    ix=torch.arange(count,device=t.device,dtype=torch.long)*(max(t.numel()-1,0))//max(count-1,1)
    coords=torch.unravel_index(ix,t.shape) if t.ndim else ()
    v=t[coords] if t.ndim else t.unsqueeze(0)
    return v.double().cpu().tolist()

def summary(v):
    good=[x for x in v if math.isfinite(x)]
    return {'sample_count':len(v),'nonfinite_count':len(v)-len(good),'min':min(good) if good else None,'max':max(good) if good else None,'mean':math.fsum(good)/len(good) if good else None,'rms':math.sqrt(math.fsum(x*x for x in good)/len(good)) if good else None,'zeros':sum(x==0 for x in good)}

class Diagnostics:
    def __init__(self,path):
        from pathlib import Path
        self.path=Path(path);self.handles=[];self.before={};self.classes={};self.context={}
    def emit(self,event):
        import json
        if (self.path.stat().st_size if self.path.exists() else 0)+len(json.dumps(event))+1>POLICY['max_log_bytes']:raise ValueError('Diagnostic byte limit')
        append_diagnostic(self.path,event)
    def attach(self,model):
        names=[]
        for name,module in model.named_modules():
            if name in ('encoder','head') or (name.startswith('transformers.') and name.count('.')==1):names.append((name,module))
        for name,module in names[:POLICY['activation_hooks']]:
            def hook(mod,args,out,name=name):
                values=out if isinstance(out,(tuple,list)) else [out]
                stats=[summary(sample(v,POLICY['activation_values'])) for v in values[:4] if isinstance(v,torch.Tensor)]
                self.emit({'kind':'activation_sample','module':name,**self.context,'statistics':stats,'full_tensor_retained':False})
            self.handles.append(module.register_forward_hook(hook))
    def close(self):
        for h in self.handles:h.remove()
        self.handles=[];self.before={}
    def microbatch(self,logits,y,weights,epoch,step):
        with torch.no_grad():
            ce=torch.nn.functional.cross_entropy(logits.detach(),y,reduction='none').double()
            prob=torch.softmax(logits.detach().double(),dim=1)[:,1]
            classes={}
            for label in (0,1):
                mask=y==label;n=int(mask.sum());u=float(ce[mask].sum());w=u*float(weights[label]);p=prob[mask].cpu().tolist()
                classes[str(label)]={'count':n,'unweighted_loss_sum':u,'weighted_loss_sum':w,'positive_probability':summary(p)}
            self.emit({'kind':'class_losses','epoch':epoch,'microbatch':step,'classes':classes})
    def capture(self,model):
        self.before={};left=POLICY['parameter_values_total']
        with torch.no_grad():
            for name,p in model.named_parameters():
                if p.requires_grad and left:
                    values=sample(p,min(left,POLICY['parameter_values_per_tensor']));self.before[name]=values;left-=len(values)
    def boundary(self,model,optimizer,event):
        families={}
        with torch.no_grad():
            named=dict(model.named_parameters())
            for name,old in self.before.items():
                new=sample(named[name],len(old));d=families.setdefault(family(name),{'sample_count':0,'changed_values':0,'delta_squared_sum':0.,'before_squared_sum':0.})
                d['sample_count']+=len(old);d['changed_values']+=sum(x!=y for x,y in zip(old,new));d['delta_squared_sum']+=math.fsum((y-x)**2 for x,y in zip(old,new));d['before_squared_sum']+=math.fsum(x*x for x in old)
        self.before={}
        self.emit({'kind':'optimizer_boundary',**self.context,'status':event['status'],'actual_updates':event['optimizer_updates'],'groups':[{'index':i,'lr':float(g['lr']),'weight_decay':float(g.get('weight_decay',0)),'parameter_tensors':len(g['params'])} for i,g in enumerate(optimizer.param_groups)],'gradient_norm_before_clip':event.get('norm_before_clip'),'gradient_norm_after_clip':event.get('norm_after_clip'),'denominator':event['denominator'],'clip_limit':event.get('clip_limit'),'clipping_applied':event.get('norm_before_clip',0)>event.get('clip_limit',float('inf')),'families_sampled':families,'parameter_changes_are_sampled':True,'counters':event['counters']})

    def validation(self,predictions,labels,weights,epoch):
        classes={str(k):{'count':0,'unweighted_loss_sum':0.,'weighted_loss_sum':0.,'probability_sum':0.,'probability_min':None,'probability_max':None} for k in (0,1)}
        for r in predictions:
            k=labels[r['id']];v=r['logits'];maximum=max(v);exp=[math.exp(x-maximum) for x in v];den=sum(exp);prob=exp[1]/den;loss=maximum+math.log(den)-v[k]
            d=classes[str(k)];d['count']+=1;d['unweighted_loss_sum']+=loss;d['weighted_loss_sum']+=loss*weights[k];d['probability_sum']+=prob
            d['probability_min']=prob if d['probability_min'] is None else min(d['probability_min'],prob);d['probability_max']=prob if d['probability_max'] is None else max(d['probability_max'],prob)
        for d in classes.values():d['probability_mean']=d['probability_sum']/d['count'] if d['count'] else None
        self.emit({'kind':'complete_validation_diagnostics','epoch':epoch,'classes':classes,'rows':sum(d['count'] for d in classes.values()),'not_model_quality_acceptance':True})
