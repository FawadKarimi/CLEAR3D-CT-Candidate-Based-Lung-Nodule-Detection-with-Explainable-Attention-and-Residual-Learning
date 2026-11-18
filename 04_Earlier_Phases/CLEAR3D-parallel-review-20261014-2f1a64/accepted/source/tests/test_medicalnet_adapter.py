import copy,hashlib,json
from pathlib import Path
import pytest,torch
from torch.nn import functional as F
from lidc3d.medicalnet_adapter import *
ROOT=Path(__file__).resolve().parents[2]
CFG={'kind':'hybrid','width':16,'patch_size':48,'embed_dim':128,'heads':4,'layers':2,'ff_dim':256,'dropout':.1,'head_dropout':.3,'scales':[0,1,2],'positional_encoding':True,'fusion':'hierarchical'}
@pytest.mark.parametrize('shape',[(1,2,4,5,6),(2,1,5,5,5),(1,1,1,2,3),(1,2,24,24,24)])
@pytest.mark.parametrize('ties',[False,True])
def test_maxpool_value_gradient(shape,ties):
 torch.manual_seed(4);x=torch.randn(shape,dtype=torch.float64,requires_grad=True)
 if ties:x.data.fill_(1)
 y=x.detach().clone().requires_grad_();a=DeterministicMaxPool()(x);b=F.max_pool3d(y,3,2,1)
 assert torch.equal(a,b);g=torch.randn_like(a);a.backward(g);b.backward(g)
 torch.testing.assert_close(x.grad,y.grad,rtol=1e-12,atol=1e-12)

def test_shortcut_values_and_gradient():
 x=torch.randn(2,3,5,6,7,requires_grad=True);a=shortcut_a(x,6,2);b=official.downsample_basic_block(x,6,2,True)
 assert torch.equal(a,b) and not b.requires_grad
 a.sum().backward();expected=torch.zeros_like(x);expected[:,:,::2,::2,::2]=1;assert torch.equal(x.grad,expected)

@pytest.mark.parametrize('depth',[18,10])
def test_exact_checkpoint_mapping_and_encoder_forward(depth):
 m=backbone(depth);mapping=load_encoder(m,ROOT/f'weights/resnet_{depth}.pth',WEIGHTS[depth]);assert len(mapping)==len(m.state_dict())
 oracle=backbone(depth,False);oracle.load_state_dict(m.state_dict());m.eval();oracle.eval();x=torch.rand(1,1,16,16,16)
 def features(b):
  y=b.maxpool(b.relu(b.bn1(b.conv1(x))))
  for layer in (b.layer1,b.layer2,b.layer3,b.layer4):y=layer(y)
  return y
 with torch.no_grad():torch.testing.assert_close(features(m),features(oracle),rtol=0,atol=0)

@pytest.mark.parametrize('kind',['hash','missing','unexpected','shape','dtype','nonfinite'])
def test_loader_refuses(tmp_path,kind):
 m=backbone(10);s={'module.'+k:v.clone() for k,v in m.state_dict().items()};key='module.conv1.weight'
 if kind=='missing':del s[key]
 if kind=='unexpected':s['module.bad']=torch.zeros(1)
 if kind=='shape':s[key]=torch.zeros(1)
 if kind=='dtype':s[key]=s[key].double()
 if kind=='nonfinite':s[key].flatten()[0]=float('nan')
 p=tmp_path/'x.pth';torch.save({'state_dict':s},p)
 with pytest.raises(ValueError):load_encoder(m,p,'0'*64 if kind=='hash' else digest(p))

@pytest.mark.parametrize('depth',[18,10])
def test_production_shape_full_hybrid_backward_and_bn(depth):
 torch.manual_seed(17);model=build_transfer(CFG,depth,ROOT/f'weights/resnet_{depth}.pth');model.train()
 bn={k:v.clone() for k,v in model.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k}
 x=torch.rand(1,1,48,48,48);r=model(x,explain=True)
 assert [tuple(f.shape) for f in r['features']]==[(1,32,12,12,12),(1,64,6,6,6),(1,128,3,3,3)]
 loss=F.cross_entropy(r['logits'],torch.tensor([1]));loss.backward()
 assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
 assert model.encoder.backbone.conv1.weight.grad.norm()>0
 assert all(torch.equal(v,model.state_dict()[k]) for k,v in bn.items())

def test_matched_initialization_and_rng():
 torch.manual_seed(777);before=torch.get_rng_state().clone();a=build_transfer(CFG,18,ROOT/'weights/resnet_18.pth');assert torch.equal(before,torch.get_rng_state())
 b=build_transfer(CFG,10,ROOT/'weights/resnet_10.pth')
 common={k:v for k,v in a.state_dict().items() if not k.startswith('encoder.backbone.')}
 assert all(torch.equal(v,b.state_dict()[k]) for k,v in common.items())
 assert a.config['transfer']['checkpoint_sha256']!=b.config['transfer']['checkpoint_sha256']

def test_no_architecture_expansion_or_scratch():
 with pytest.raises(ValueError):backbone(34)
 with pytest.raises(ValueError):build_transfer(CFG,10,ROOT/'upstream/LICENSE')
 with pytest.raises(ValueError):build_transfer({**CFG,'width':32},10,ROOT/'weights/resnet_10.pth')

@pytest.mark.parametrize('depth',[18,10])
def test_weighted_incomplete_actual_update_and_state_reload(depth):
 from lidc3d.optimization import train_epoch,new_update_state
 from lidc3d.training import parameter_groups
 from lidc3d.provenance import canonical_hash
 torch.manual_seed(17);m=build_transfer(CFG,depth,ROOT/f'weights/resnet_{depth}.pth');m.train()
 cfg={'lr_cnn':1e-4,'lr_transformer':5e-5,'lr_head':1e-4,'weight_decay':1e-4}
 opt=torch.optim.AdamW(parameter_groups(m,cfg),foreach=False,fused=False)
 before=m.head[-1].weight.detach().clone();events=[];state=new_update_state()
 g=torch.Generator().manual_seed(17);batches=[{'image':torch.rand(2,1,48,48,48,generator=g),'label':torch.tensor([1,0])},{'image':torch.rand(1,1,48,48,48,generator=g),'label':torch.tensor([1])}]
 loss=train_epoch(m,batches,opt,torch.amp.GradScaler('cuda',enabled=False),torch.tensor([1.,3.]),torch.device('cpu'),False,8,1,state,events.append,0)
 assert math.isfinite(loss) and state['optimizer_updates']==1 and state['skipped_updates']==0
 assert events[-1]['denominator']==7 and events[-1]['incomplete_group'] and events[-1]['norm_before_clip']>0
 assert not torch.equal(before,m.head[-1].weight)
 assert all(float(v['step'])==1 for v in opt.state.values())
 clone=build_transfer(CFG,depth,ROOT/f'weights/resnet_{depth}.pth');clone.load_state_dict(m.state_dict(),strict=True)
 m.eval();clone.eval()
 with torch.no_grad():assert torch.equal(m(batches[-1]['image']),clone(batches[-1]['image']))
 assert canonical_hash(m.config)!=canonical_hash(build_transfer(CFG,10 if depth==18 else 18,ROOT/f'weights/resnet_{10 if depth==18 else 18}.pth').config)
