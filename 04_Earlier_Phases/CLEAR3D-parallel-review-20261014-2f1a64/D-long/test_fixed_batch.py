"""Generated-only tests of the proposed fixed-batch runner; no GPU/data files."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

p=argparse.ArgumentParser()
p.add_argument('--source',required=True)
p.add_argument('--runner',required=True)
p.add_argument('--checkpoints',choices=('enabled','disabled'),default='enabled')
a,rest=p.parse_known_args()
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise SystemExit('Require CUDA_VISIBLE_DEVICES empty')
sys.path.insert(0,str(Path(a.source).resolve()))
spec=importlib.util.spec_from_file_location('fixed_batch_under_test',Path(a.runner).resolve())
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

class FixedBatchTests(unittest.TestCase):
    def test_choose_first_unique_order_and_reject_drift(self):
        values=([3,3,7,1,9]*48)+([101+i%24 for i in range(720)])
        expected=[3,7,1,9]+list(range(101,113))
        self.assertEqual(runner.choose(values),expected)
        self.assertEqual(runner.choose(list(values)),expected)
        bad=[values[:-1],values+[999],[-1]+values[1:],[True]+values[1:],
             [1]*240+values[240:],values[:240]+[101]*720,
             values[:240]+[3,7,1,9]+values[244:]]
        for item in bad:
            with self.subTest(length=len(item)),self.assertRaises(ValueError):runner.choose(item)

    def test_adoption_draft_missing_holds_and_binding_refusal(self):
        value=dict(status='ADOPTED',scope=runner.SCOPE,binding_sha256='a'*64,
                   policy=copy.deepcopy(runner.POLICY),limits=copy.deepcopy(runner.LIMITS),
                   reference_release='HOLD',cap4000_policy='NOT_ADOPTED',
                   supplied_by='SYNTHETIC TEST ONLY',instruction='Generated test fixture, no authorization',date='synthetic',
                   holds=['reference_origin','excluded_reference_outliers','geometry_and_source_scope',
                          'protocol_evaluation_disclosure','prior_test_access','independent_reference_review'])
        for k in ('label_guided','prior_exposure_uncertain','pretraining_overlap_uncertain','sampled_monitoring','no_scientific_approval','train_only'):value[k]=True
        runner.adoption(value,'a'*64)
        for key,new in [('status','DRAFT'),('holds',value['holds'][:-1]),('supplied_by',''),
                        ('reference_release','APPROVED'),('cap4000_policy','ADOPTED'),('binding_sha256','b'*64),
                        ('sampled_monitoring',False)]:
            changed=copy.deepcopy(value);changed[key]=new
            with self.subTest(key=key),self.assertRaises(ValueError):runner.adoption(changed,'a'*64)

    def test_default_nonexecuting_no_torch(self):
        run=subprocess.run([sys.executable,'-B',str(Path(a.runner).resolve())],
                           capture_output=True,text=True,timeout=15,check=False)
        self.assertEqual(run.returncode,0,run.stderr)
        result=json.loads(run.stdout)
        self.assertEqual(result['status'],'PLAN_ONLY')
        self.assertFalse(result['torch_imported'])
        self.assertFalse(result['policy']['validation'])
        self.assertFalse(result['policy']['resume'])

    def test_generated_24_updates_denominator_checkpoint_and_diagnostics(self):
        import torch
        from torch import nn
        from lidc3d.provenance import canonical_hash
        from lidc3d.training import epoch_lr
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(17)
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__();self.encoder=nn.Sequential(nn.Linear(4,8),nn.Tanh());self.head=nn.Linear(8,2)
            def forward(self,x):return self.head(self.encoder(x))
        model=Tiny();initial={n:z.clone() for n,z in model.state_dict().items()}
        images=torch.randn(16,4);labels=torch.tensor([1]*4+[0]*12)
        batches=[dict(image=images[i:i+2],label=labels[i:i+2]) for i in range(0,16,2)]
        cfg={'training':dict(lr_cnn=1e-4,lr_transformer=5e-5,lr_head=1e-4,weight_decay=1e-4,
                            warmup_epochs=5,max_epochs=50,min_lr=1e-6,
                            class_weights=[1.,3.],clip_grad_norm=1)}
        identity=dict(scope='GENERATED_TEST_ONLY',config_hash=canonical_hash(cfg),selection_sha256='1'*64)
        with tempfile.TemporaryDirectory() as name:
            out=Path(name)
            result=runner.train_fixed(model,batches,cfg,'cpu',out,identity,checkpoint=a.checkpoints=='enabled')
            self.assertEqual((result['status'],result['updates'],result['skips'],result['draws']),('PASS',120,0,1920))
            records=[json.loads(z) for z in (out/'boundaries.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),120)
            self.assertTrue(all(z['denominator']==24 and z['norm_before_clip']>0 and z['status']=='UPDATED' for z in records))
            self.assertEqual([z['counters']['optimizer_updates'] for z in records],list(range(1,121)))
            self.assertTrue(any(not torch.equal(z,model.state_dict()[n]) for n,z in initial.items()))
            diagnostics=[json.loads(z) for z in (out/'training-diagnostics.jsonl').read_text().splitlines()]
            self.assertEqual(sum(z['kind']=='class_losses' for z in diagnostics),120*8)
            boundaries=[z for z in diagnostics if z['kind']=='optimizer_boundary']
            self.assertEqual(len(boundaries),120)
            self.assertTrue(all(g['lr']==epoch_lr(cfg['training']['lr_head'],0,cfg['training'])
                                for z in boundaries for g in z['groups']))
            self.assertFalse(any(z['kind']=='complete_validation_diagnostics' for z in diagnostics))
            if a.checkpoints=='enabled':
                from lidc3d.engineering import load_checkpoint
                final=load_checkpoint(out,identity)
                self.assertEqual(final['state']['stage'],'TRAINED')
                self.assertFalse(final['state']['validation_complete'])
                self.assertEqual(final['state']['update_state']['optimizer_updates'],120)
                self.assertTrue(all(float(v['step'])==120 for v in final['state']['optimizer']['state'].values()))
                self.assertIsNotNone(final['record']['parent_commit'])
                commits=[json.loads(z.read_text()) for z in (out/'engineering-checkpoints').glob('*/commit.json')]
                self.assertEqual(sorted(z['stage'] for z in commits),['INITIALIZED','TRAINED'])
                with self.assertRaises(ValueError):load_checkpoint(out,{**identity,'scope':'changed'})
            else:self.assertFalse((out/'engineering-commit.json').exists())
        print(json.dumps({'generated_fixture':True,'checkpoints_tested':a.checkpoints=='enabled','torch':torch.__version__}))

if __name__=='__main__':unittest.main(argv=[sys.argv[0]]+rest,verbosity=2)
