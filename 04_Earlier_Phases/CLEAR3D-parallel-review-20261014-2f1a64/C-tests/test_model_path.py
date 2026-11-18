"""Generated CPU probes; no checkpoint, research input, CUDA, or source edits.

Run: CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -B
  C-tests/test_model_path.py --source /exact/reviewed/work/src
Uses production operators and dimensions, randomly initialized encoder tensors.
The patched loader is a fixture, NOT checkpoint/provenance qualification.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True)
args, rest = parser.parse_known_args()
if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise SystemExit('Require CUDA_VISIBLE_DEVICES empty')
sys.path.insert(0, str(Path(args.source).resolve()))
import torch
from torch import nn
from torch.nn import functional as F
from lidc3d.models import DetectorClassifier, ScaleTransformer, global_average_pool3d
from lidc3d.medicalnet_adapter import MedicalEncoder, shortcut_a
from lidc3d.diagnostics import Diagnostics

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
CFG = dict(kind='hybrid', width=16, patch_size=48, embed_dim=128, heads=4,
           layers=2, ff_dim=256, dropout=.1, head_dropout=.3, scales=[0,1,2],
           positional_encoding=True, fusion='hierarchical')


class ModelPathTests(unittest.TestCase):
    def test_balanced_weighted_constant_predictor_stationary_point(self):
        # Algebraic control; not evidence that the real model is constant.
        labels = torch.tensor([1]*240 + [0]*720)
        parameter = torch.zeros(1,2,dtype=torch.float64,requires_grad=True)
        weights = torch.tensor([1.,3.],dtype=torch.float64)
        loss = F.cross_entropy(parameter.expand(960,2),labels,weight=weights)
        loss.backward()
        self.assertAlmostEqual(float(loss.detach()), __import__('math').log(2),places=12)
        self.assertLess(float(parameter.grad.abs().max()), 1e-14)
        shifted = torch.tensor([[0.,.5]],dtype=torch.float64).expand(960,2)
        self.assertGreater(float(F.cross_entropy(shifted,labels,weight=weights)),float(loss.detach()))

    def test_production_shape_input_variation_and_all_families_backward(self):
        torch.manual_seed(17)
        model = DetectorClassifier(CFG)
        with patch('lidc3d.medicalnet_adapter.load_encoder', return_value=[]):
            model.encoder = MedicalEncoder(18, 'GENERATED_FIXTURE_NO_FILE')
        model.train()
        bn = [m for m in model.encoder.backbone.modules() if isinstance(m, nn.BatchNorm3d)]
        self.assertTrue(bn and all(not m.training for m in bn))
        bn_before = {k: v.clone() for k,v in model.state_dict().items()
                     if 'running_' in k or 'num_batches_tracked' in k}
        # Disable dropout only through eval: same fixed model, deterministic input comparison.
        model.eval()
        generator = torch.Generator().manual_seed(219)
        inputs = torch.rand(2,1,48,48,48,generator=generator)
        inputs[1] = (inputs[1] * .35 + .2).flip(-1)
        inputs.requires_grad_()
        result = model(inputs, explain=True)
        self.assertEqual([tuple(t.shape) for t in result['features']],
                         [(2,32,12,12,12),(2,64,6,6,6),(2,128,3,3,3)])
        for t in result['features']:
            self.assertTrue(t.requires_grad)
            self.assertGreater(float((t[0]-t[1]).abs().max().detach()), 0)
        logits = result['logits']
        self.assertTrue(torch.isfinite(logits).all())
        self.assertGreater(float((logits[0]-logits[1]).abs().max().detach()), 1e-7)
        F.cross_entropy(logits, torch.tensor([0,1]), weight=torch.tensor([1.,3.])).backward()
        families = {'encoder': 'encoder.backbone.', 'bridges': 'encoder.bridge.',
                    'transformer': 'transformers.', 'fusion': 'merge.', 'classifier': 'head.'}
        evidence = {}
        for family,prefix in families.items():
            tensors = [p for n,p in model.named_parameters() if n.startswith(prefix)]
            self.assertTrue(tensors)
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in tensors))
            norm = sum(float(p.grad.double().square().sum()) for p in tensors) ** .5
            self.assertGreater(norm, 0)
            evidence[family] = norm
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertGreater(float(inputs.grad.double().norm()), 0)
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k,v in bn_before.items()))
        self.assertTrue(all(m.weight.requires_grad and m.weight.grad is not None for m in bn))
        print('GENERATED_GRADIENT_EVIDENCE', json.dumps(evidence, sort_keys=True))

    def test_pooling_does_not_cancel_channel_specific_signal(self):
        torch.manual_seed(103)
        transformer = ScaleTransformer(32, CFG).double().eval()
        x = torch.randn(1,32,3,3,3,dtype=torch.float64,requires_grad=True)
        y = x.detach().clone(); y[:,0] += .7
        a,_ = transformer(x); b,_ = transformer(y)
        pa = global_average_pool3d(a).flatten(1)
        pb = global_average_pool3d(b).flatten(1)
        # LayerNorm's channel centering does not imply the learned projected pooled vector is constant.
        self.assertGreater(float((pa-pb).abs().max().detach()), 1e-5)
        (pa * torch.linspace(-1,1,32,dtype=torch.float64)).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.norm()), 0)

    def test_corrected_shortcut_keeps_input_gradient(self):
        x = torch.randn(2,3,5,6,7,requires_grad=True)
        result = shortcut_a(x,6,2)
        self.assertTrue(torch.equal(result[:,:3], x[:,:,::2,::2,::2]))
        self.assertEqual(int(torch.count_nonzero(result[:,3:])), 0)
        result.sum().backward()
        expected = torch.zeros_like(x); expected[:,:,::2,::2,::2] = 1
        self.assertTrue(torch.equal(x.grad,expected))

    def test_observers_preserve_exact_logits_gradients_update_and_rng(self):
        class Fixture(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(nn.Linear(8,16),nn.ReLU())
                self.head = nn.Sequential(nn.Dropout(.3),nn.Linear(16,2))
            def forward(self,x): return self.head(self.encoder(x))
        torch.manual_seed(63)
        model = Fixture(); other = copy.deepcopy(model)
        x = torch.randn(4,8); y = torch.tensor([1,0,1,0]); weights=torch.tensor([1.,3.])
        start = torch.get_rng_state().clone()
        def step(m, diagnostics=None):
            torch.set_rng_state(start)
            optimizer=torch.optim.AdamW(m.parameters(),lr=1e-4,foreach=False,fused=False)
            if diagnostics: diagnostics.attach(m); diagnostics.capture(m)
            out=m(x)
            if diagnostics: diagnostics.microbatch(out,y,weights,0,0)
            loss=F.cross_entropy(out,y,weight=weights);loss.backward()
            gradients={n:p.grad.clone() for n,p in m.named_parameters()}
            optimizer.step()
            if diagnostics: diagnostics.close()
            return out.detach(), gradients, optimizer.state_dict(),torch.get_rng_state().clone()
        a=step(model)
        with tempfile.TemporaryDirectory() as directory:
            d=Diagnostics(Path(directory)/'diagnostics.jsonl');b=step(other,d)
            events=[json.loads(line) for line in d.path.read_text().splitlines()]
            self.assertTrue(any(e['kind']=='activation_sample' for e in events))
            self.assertTrue(any(e['kind']=='class_losses' for e in events))
            self.assertFalse(d.handles or d.before)
        self.assertTrue(torch.equal(a[0],b[0]) and torch.equal(a[3],b[3]))
        for n in a[1]: self.assertTrue(torch.equal(a[1][n],b[1][n]))
        for n,p in model.state_dict().items(): self.assertTrue(torch.equal(p,other.state_dict()[n]))
        for key,state in a[2]['state'].items():
            for name,value in state.items():
                self.assertTrue(torch.equal(value,b[2]['state'][key][name]))

if __name__ == '__main__':
    print(json.dumps({'python':sys.version,'torch':torch.__version__,'source':str(Path(args.source).resolve()),
                      'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'torch_threads':torch.get_num_threads(),
                      'generated_random_weights_only':True,'not_pretrained_or_cuda_evidence':True}))
    unittest.main(argv=[sys.argv[0]]+rest,verbosity=2)
