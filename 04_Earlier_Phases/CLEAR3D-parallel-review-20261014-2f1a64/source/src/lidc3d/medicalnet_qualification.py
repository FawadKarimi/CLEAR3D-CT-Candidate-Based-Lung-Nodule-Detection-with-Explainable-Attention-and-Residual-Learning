"""Generated-only tests of actual transfer models; no data-file loading."""
import copy,math,time
from pathlib import Path
from . import medicalnet_contract as c

def exact(a,b):
 import torch
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
 return a==b

def operator_probes(device,emit):
 import torch
 from torch.nn import functional as F
 from .medicalnet_adapter import DeterministicMaxPool,shortcut_a
 g=torch.Generator().manual_seed(117)
 for shape in ((1,2,4,5,6),(1,1,1,2,3),(2,2,24,24,24)):
  for ties in (False,True):
   x=torch.rand(shape,generator=g) if not ties else torch.zeros(shape)
   cpu=x.requires_grad_();gpu=x.detach().to(device).requires_grad_()
   a=F.max_pool3d(cpu,3,2,1);b=DeterministicMaxPool()(gpu);grad=torch.rand(a.shape,generator=g)
   a.backward(grad);b.backward(grad.to(device))
   torch.testing.assert_close(b.cpu(),a,rtol=0,atol=0)
   torch.testing.assert_close(gpu.grad.cpu(),cpu.grad,rtol=1e-5,atol=1e-6)
   emit({'stage':'pool_ties_backward','shape':shape,'ties':ties,'rtol':1e-5,'atol':1e-6,'status':'PASS'})
 for stride in (1,2):
  x=torch.randn(2,3,5,6,7,generator=g).to(device).requires_grad_();y=shortcut_a(x,6,stride);y.sum().backward()
  expected=torch.zeros_like(x);expected[:,:,::stride,::stride,::stride]=1
  c.require(torch.equal(x.grad,expected),'Shortcut gradient mismatch')
  c.require(torch.equal(y[:,:3],x[:,:,::stride,::stride,::stride]) and torch.count_nonzero(y[:,3:])==0,'Shortcut forward mismatch')
  emit({'stage':'shortcut_backward','stride':stride,'status':'PASS'})

def generated_repeat(depth,weights,cfg,out,device,binding,emit,model_factory=None):
 import torch
 from .medicalnet_adapter import build_transfer
 from .provenance import seed_all,write_json
 from .training import parameter_groups,epoch_lr
 from .optimization import train_epoch,new_update_state
 from .engineering import publish_checkpoint,load_checkpoint
 seed_all(17);out=Path(out);out.mkdir();write_json(out/'expected-identity.json',binding)
 model=(model_factory or (lambda cfg:build_transfer(cfg,depth,weights)))(cfg['model']).to(device)
 groups=parameter_groups(model,cfg['training'])
 for group in groups:group['lr']=epoch_lr(group['lr'],0,cfg['training'])
 optimizer=torch.optim.AdamW(groups);scaler=torch.amp.GradScaler('cuda',enabled=False);state=new_update_state()
 generator=torch.Generator().manual_seed(717);x=torch.rand(5,1,48,48,48,generator=generator);y=torch.tensor([1,0,1,0,1])
 batches=[{'image':x[i:i+2],'label':y[i:i+2]} for i in range(0,5,2)]
 def snapshot(stage):return {'stage':stage,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'update_state':copy.deepcopy(state),'config':cfg,'selection_sha256':binding['selection_sha256']}
 publish_checkpoint(out,snapshot('INITIALIZED'),binding)
 before={k:p.detach().cpu().clone() for k,p in model.named_parameters()};events=[]
 def event(v):events.append(v);emit(v)
 model.train();loss=train_epoch(model,batches,optimizer,scaler,torch.tensor([1.,3.],device=device),torch.device(device),False,8,1.,state,event,0)
 c.require(math.isfinite(loss) and state['optimizer_updates']==1 and state['skipped_updates']==0,'Finite usable update required')
 c.require(events[-1]['denominator']==11 and events[-1]['norm_before_clip']>0,'Weighted incomplete-group gradient failure')
 # Accepted optimizer clears gradients after the step; inspect its pre-clear evidence.
 gradients=events[-1]['normalized_gradients']
 c.require(not gradients['missing'] and not gradients['nonfinite'] and gradients['gradient_tensors']==len(list(model.parameters())),'Nonfinite/missing individual gradients')
 c.require(all(float(v['step'])==1 for v in optimizer.state.values()),'AdamW advancement missing')
 changed=[k for k,p in model.named_parameters() if not torch.equal(before[k],p.detach().cpu())];c.require(bool(changed),'No changed parameters')
 publish_checkpoint(out,snapshot('TRAINED'),binding);loaded=load_checkpoint(out,binding) # map_location CPU in accepted implementation
 model.eval()
 with torch.no_grad():prediction=model(x[:2].to(device)).detach().cpu()
 write_json(out/'update-evidence.json',{'loss':loss,'changed_parameters':changed,'state':state,'denominator':11,'gradient_norm':events[-1]['norm_before_clip'],'checkpoint_commit':loaded['commit_sha256'],'cpu_checkpoint_load':True})
 return {'loss':loss,'model':loaded['state']['model'],'optimizer':loaded['state']['optimizer'],'prediction':prediction,'update_state':state}
