"""Isolated MedicalNet transfer adapter. No research runner or approval bypass."""
from __future__ import annotations
import hashlib,math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import once_differentiable
from . import medicalnet_vendor as official
from .models import DetectorClassifier

WEIGHTS={18:'38b3a17444c5c18a795355a868e6a2d9ce9318054067f5373d337f55c0f61da3',10:'2824951eaef30e115929b89bec72342380b08d14792bda1449974ac849e497c0'}
POLICY='med3d-original-mirror-fp32-frozen-bn-pyramid124-v1'

def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

class _MaxPool(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x):
  y,idx=F.max_pool3d(x,3,2,1,return_indices=True)
  ctx.save_for_backward(idx);ctx.shape=x.shape
  return y
 @staticmethod
 @once_differentiable
 def backward(ctx,g):
  (idx,)=ctx.saved_tensors;B,C,D,H,W=ctx.shape;od,oh,ow=g.shape[-3:]
  result=g.new_zeros(B,C,D+2,H+2,W+2)
  z=2*torch.arange(od,device=g.device)-1;y=2*torch.arange(oh,device=g.device)-1;x=2*torch.arange(ow,device=g.device)-1
  # Each offset writes unique locations. Offset accumulation order is fixed;
  # no scatter_add/atomic accumulation. First-order backward only.
  for dz in range(3):
   for dy in range(3):
    for dx in range(3):
     zz=z+dz;yy=y+dy;xx=x+dx
     linear=(zz[:,None,None]*H+yy[None,:,None])*W+xx[None,None,:]
     valid=(zz[:,None,None]>=0)&(zz[:,None,None]<D)&(yy[None,:,None]>=0)&(yy[None,:,None]<H)&(xx[None,None,:]>=0)&(xx[None,None,:]<W)
     sl=(slice(None),slice(None),slice(dz,dz+2*od,2),slice(dy,dy+2*oh,2),slice(dx,dx+2*ow,2))
     result[sl]=result[sl]+g*((idx==linear)&valid)
  return result[:,:,1:D+1,1:H+1,1:W+1]

class DeterministicMaxPool(nn.Module):
 def forward(self,x):return _MaxPool.apply(x)

def shortcut_a(x,planes,stride,no_cuda=False):
 # Exactly kernel-1 average pooling in the forward, without upstream .data detach.
 out=x[:,:,::stride,::stride,::stride]
 return torch.cat((out,out.new_zeros(out.shape[0],planes-out.shape[1],*out.shape[2:])),1)

def backbone(depth,corrected=True):
 if type(depth) is not int or depth not in (18,10):raise ValueError('Only ResNet18 and ResNet10')
 m=getattr(official,'resnet'+str(depth))(sample_input_D=48,sample_input_H=48,sample_input_W=48,num_seg_classes=2,shortcut_type='A' if depth==18 else 'B',no_cuda=True)
 del m.conv_seg
 if corrected:
  m.maxpool=DeterministicMaxPool()
  if depth==18:
   from functools import partial
   for layer in (m.layer2,m.layer3,m.layer4):
    block=layer[0];block.downsample=partial(shortcut_a,planes=block.conv1.out_channels,stride=block.stride)
 return m

def load_encoder(m,path,expected):
 if digest(path)!=expected:raise ValueError('Checkpoint byte identity mismatch')
 raw=torch.load(path,map_location='cpu',weights_only=True)
 if not isinstance(raw,dict) or set(raw)!={'state_dict'}:raise ValueError('Unexpected checkpoint envelope')
 source=raw['state_dict'];target=m.state_dict();mapped={};report=[]
 for key,value in source.items():
  if not key.startswith('module.'):raise ValueError('Unexpected source prefix')
  dest=key[len('module.'):]
  if dest in mapped or dest not in target:raise ValueError('Unexpected or duplicate tensor: '+key)
  if not isinstance(value,torch.Tensor) or value.shape!=target[dest].shape or value.dtype!=target[dest].dtype:raise ValueError('Tensor shape/dtype mismatch: '+key)
  if value.is_floating_point() and not bool(torch.isfinite(value).all()):raise ValueError('Nonfinite source tensor: '+key)
  mapped[dest]=value
  report.append({'source':key,'target':'encoder.backbone.'+dest,'shape':list(value.shape),'dtype':str(value.dtype),'kind':'parameter' if dest in dict(m.named_parameters()) else 'buffer','tensor_sha256':hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()})
 if set(mapped)!=set(target):raise ValueError('Missing encoder tensor')
 m.load_state_dict(mapped,strict=True)
 return report

class MedicalEncoder(nn.Module):
 def __init__(self,depth,path,seed=2718):
  super().__init__();self.depth=depth
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(seed);self.backbone=backbone(depth);self.mapping=load_encoder(self.backbone,path,WEIGHTS[depth])
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(seed);self.bridge=nn.ModuleList([nn.Conv3d(a,b,1,bias=False) for a,b in ((64,32),(128,64),(512,128))])
  self.train(True)
 def train(self,mode=True):
  super().train(mode)
  for m in self.backbone.modules():
   if isinstance(m,nn.BatchNorm3d):m.eval()
  return self
 def forward(self,x):
  if x.ndim!=5 or x.shape[1:]!=(1,48,48,48):raise ValueError('Fixed B,1,48,48,48 contract')
  m=self.backbone;x=m.maxpool(m.relu(m.bn1(m.conv1(x))))
  f1=m.layer1(x);f2=m.layer2(f1);f3=m.layer3(f2);f4=m.layer4(f3)
  a,b,c=[layer(f) for layer,f in zip(self.bridge,(f1,f2,f4))]
  # Nonoverlapping 2x2x2 mean, avoiding CUDA adaptive pooling backward.
  c=c.reshape(c.shape[0],128,3,2,3,2,3,2).mean((3,5,7))
  return [a,b,c]

def build_transfer(cfg,depth,path,seed=17):
 if cfg['width']!=16 or cfg['patch_size']!=48 or cfg['kind']!='hybrid':raise ValueError('Unreviewed model contract')
 if type(depth) is not int or depth not in WEIGHTS:raise ValueError('Unreviewed depth')
 if digest(path)!=WEIGHTS[depth]:raise ValueError('Pretrained checkpoint mandatory; no scratch fallback')
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(seed);model=DetectorClassifier(cfg)
 model.encoder=MedicalEncoder(depth,path)
 model.config={**cfg,'transfer':{'policy':POLICY,'depth':depth,'checkpoint_sha256':WEIGHTS[depth],'pretraining_population':'original-name mirror, official equivalence unresolved','frozen_bn_statistics':True,'bridge_seed':2718,'initialization_seed':seed}}
 return model
