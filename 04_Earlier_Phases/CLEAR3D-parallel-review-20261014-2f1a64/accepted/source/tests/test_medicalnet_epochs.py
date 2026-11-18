"""Generated CPU fixtures. Tiny model doubles are not real-model CUDA parity."""
import copy
import json
import random
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from lidc3d import training, checkpoints as cp, medicalnet_finetuning as ft
from lidc3d.provenance import read_json, sha256
from test_ingest_and_guards import make_bundle
from test_checkpoint_sets import assert_same, tree_hashes


class Rows(list):
    role = 'train'


def sampling(p=4, n=2):
    return dict(policy='positive-cycle-negative-hash-v1', positive_draws=p,
                negative_draws=n, max_source_rows=283200, max_epoch_rows=65536)


def test_bounded_sampling_replacement_and_epoch_determinism():
    rows = Rows([{'id': str(i), 'label': int(i < 2)} for i in range(10)])
    a = ft.TrainingSample(rows, sampling(6, 4), 17)
    b = ft.TrainingSample(rows, sampling(6, 4), 17)
    assert a.indices == b.indices and len(a) == 10
    assert sum(a[i]['label'] for i in range(len(a))) == 6
    assert len(set(a.indices[:6])) == 2 and len(set(a.indices[6:])) == 4
    a.set_epoch(2); b.set_epoch(2)
    assert a.indices == b.indices and a.receipt == b.receipt
    assert rows == [{'id': str(i), 'label': int(i < 2)} for i in range(10)]


@pytest.mark.parametrize('fault', ['role', 'ignored', 'duplicate', 'no_positive', 'few_negative', 'oversize'])
def test_sampling_refusals(fault):
    rows = Rows([{'id': str(i), 'label': int(i == 0)} for i in range(3)])
    spec = sampling()
    if fault == 'role': rows.role = 'val'
    if fault == 'ignored': rows[0]['label'] = -1
    if fault == 'duplicate': rows[1]['id'] = '0'
    if fault == 'no_positive': rows[0]['label'] = 0
    if fault == 'few_negative': spec['negative_draws'] = 3
    if fault == 'oversize': spec['max_epoch_rows'] = 2
    with pytest.raises(ValueError): ft.TrainingSample(rows, spec, 17)


def test_admission_precedes_iteration():
    class Huge:
        role = 'train'
        def __len__(self): return 283201
        def __iter__(self): pytest.fail('must reject before scanning source')
    with pytest.raises(ValueError, match='before allocation'):
        ft.TrainingSample(Huge(), sampling(), 17)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.head = torch.nn.Linear(1, 2); self.drop = torch.nn.Dropout(.2)
    def forward(self, x):
        # Exercise torch, Python and NumPy RNG restoration, not just weights.
        noise = random.random() + float(np.random.random())
        return self.head(self.drop(x.mean((2, 3, 4))) + noise * .01)


@pytest.mark.parametrize('depth', [18, 10])
def test_actual_loop_sampled_resume_restores_all_state(tmp_path, monkeypatch, depth):
    bundle, cfg, _, _ = make_bundle(tmp_path)
    cfg['training'].update(max_epochs=3, warmup_epochs=1)
    cfg['medicalnet'] = {'depth': depth}
    cfg['training_sampling'] = sampling()
    # Deliberate CPU model double; the production gates run before these seams.
    monkeypatch.setattr(ft, 'validate', lambda c, s: {'depth': depth, 'fixture': 'generated-double'})
    monkeypatch.setattr(ft, 'build', lambda c, s: Tiny())
    full, interrupted, resumed = [tmp_path / p for p in ('full', 'interrupted', 'resumed')]
    training.train(bundle, cfg, full, 17, 'cpu')
    original = training.publish_set
    def stop(*args, **kwargs):
        original(*args, **kwargs)
        if args[1]['epoch'] == 0: raise RuntimeError('owned synthetic boundary interruption')
    with monkeypatch.context() as m:
        m.setattr(training, 'publish_set', stop)
        with pytest.raises(RuntimeError, match='boundary interruption'):
            training.train(bundle, cfg, interrupted, 17, 'cpu')
    before = tree_hashes(interrupted)
    training.train(bundle, cfg, resumed, 17, 'cpu', interrupted / 'last.pt')
    assert tree_hashes(interrupted) == before
    a, b = cp.load_set(full / 'last.pt', {}), cp.load_set(resumed / 'last.pt', {})
    for part in ('last', 'best'):
        for key in ('model', 'optimizer', 'scheduler', 'update_state', 'scaler', 'loader_rng',
                    'torch_rng', 'numpy_rng', 'python_rng', 'cuda_rng', 'epoch',
                    'best_cpm', 'stopping_best', 'stale'):
            assert_same(a[part][key], b[part][key])
    for epoch in (1, 2):
        assert read_json(full/f'sampling_epoch_{epoch:03d}.json') == read_json(resumed/f'sampling_epoch_{epoch:03d}.json')
        assert read_json(full/f'consumption_epoch_{epoch:03d}.json') == read_json(resumed/f'consumption_epoch_{epoch:03d}.json')
    assert a['last']['update_state']['optimizer_updates'] == 6  # incomplete groups included
    assert a['binding']['medicalnet_identity']['depth'] == depth


def test_production_approval_guards_precede_transfer_and_pixels(tmp_path, monkeypatch):
    bundle, cfg, _, _ = make_bundle(tmp_path); bundle['reference_audit_approved'] = False
    cfg['medicalnet'] = {'depth': 18}
    monkeypatch.setattr(ft, 'validate', lambda *a: pytest.fail('guard bypass'))
    with pytest.raises(ValueError, match='Reference audit'):
        training.train(bundle, cfg, tmp_path/'refused', 17)
    assert not (tmp_path/'refused').exists()


def test_failed_validation_does_not_publish_epoch(tmp_path, monkeypatch):
    bundle, cfg, _, _ = make_bundle(tmp_path); cfg['medicalnet'] = {'depth': 10}
    monkeypatch.setattr(ft, 'validate', lambda *a: {'fixture': True})
    monkeypatch.setattr(ft, 'build', lambda *a: Tiny())
    monkeypatch.setattr(training, 'predict', lambda *a: ([], [{'scan_id':'s1','status':'failed','candidates':3,'predictions':0}]))
    out = tmp_path/'failed'
    with pytest.raises(ValueError, match='Incomplete validation'): training.train(bundle,cfg,out,17)
    assert (out/'validation_epoch_000_failure.json').exists()
    assert not (out/'checkpoint_commit.json').exists() and not (out/'selection.json').exists()


@pytest.mark.parametrize('depth', [18, 10])
def test_official_transfer_identity_and_drift(tmp_path, monkeypatch, depth):
    checkpoint = tmp_path/'weights.pth'; checkpoint.write_bytes(b'generated stand-in')
    provenance = tmp_path/'provenance.json'
    record = dict(status='OFFICIAL_BYTES_MATCH', distribution_url='https://drive.google.com/file/d/13tnSvXY7oDIEloNFiGTsjUIYfS3g3BfG/view',
                  archive_sha256='a'*64, method='streamed-official-archive-member-comparison',
                  members={str(d):dict(sha256=h,official_member=f'pretrain/resnet_{d}.pth') for d,h in ft.contract.WEIGHTS.items()})
    provenance.write_text(json.dumps(record)); real = ft.contract.digest
    monkeypatch.setattr(ft.contract,'digest',lambda p: ft.contract.WEIGHTS[depth] if Path(p)==checkpoint else real(p))
    cfg={'medicalnet':dict(depth=depth,checkpoint=str(checkpoint),checkpoint_sha256=ft.contract.WEIGHTS[depth],provenance=str(provenance),provenance_sha256=real(provenance)),
         'training':{'amp':False},'model':{'kind':'hybrid','width':16,'patch_size':48}}
    identity=ft.validate(cfg,17)
    assert identity['transfer']['frozen_bn_statistics'] and identity['transfer']['no_resume'] is False
    cfg['medicalnet']['checkpoint_sha256']='b'*64
    with pytest.raises(ValueError,match='Checkpoint mismatch'):ft.validate(cfg,17)


def test_plan_does_not_import_torch():
    script=Path(__file__).parents[1]/'scripts/medicalnet_finetuning_plan.py'
    r=subprocess.run([sys.executable,'-B',str(script)],capture_output=True,text=True)
    assert r.returncode==0 and json.loads(r.stdout)['torch_imported'] is False
    assert subprocess.run([sys.executable,'-B',str(script),'--execute'],capture_output=True).returncode != 0
