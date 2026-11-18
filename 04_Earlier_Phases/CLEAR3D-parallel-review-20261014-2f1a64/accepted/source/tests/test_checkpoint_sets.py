"""Tiny synthetic CPU checkpoint tests; no production data or GPU runs."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pytest
import torch
from lidc3d import checkpoints as cp, training
from lidc3d.provenance import canonical_hash, read_json, sha256
from test_ingest_and_guards import make_bundle


def fixture_state(epoch=0):
    cfg = {"synthetic": True}
    binding = {"run_id": "synthetic-run-a", "config_hash": canonical_hash(cfg),
               "bundle_hash": "synthetic-bundle", "seed": 17, "code_sha256": "synthetic-code",
               "device_type": "cpu", "amp": False}
    state = {"checkpoint_binding": binding, "config": cfg, "epoch": epoch,
             "best_cpm": .5, "stopping_best": .5, "stale": epoch,
             "scheduler": {"epoch": epoch, "rule": "epoch_lr_v1", "base_lrs": [.001]},
             "model": {"weight": torch.tensor([float(epoch)])}, "optimizer": {"synthetic": True},
             "scaler": {}, **{k: binding[k] for k in ("config_hash", "bundle_hash", "seed")}}
    return state, binding


def tree_hashes(root):
    return {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob('*') if p.is_file()}


def assert_same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_same(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_same(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('stage', ['intent', 'last', 'best', 'sealed', 'root_last', 'root_best', 'committed'])
@pytest.mark.parametrize('initial', [True, False], ids=['first-set', 'later-set'])
@pytest.mark.parametrize('hard', [False, True], ids=['exception', 'process-exit'])
def test_publication_crash_boundaries(tmp_path, monkeypatch, stage, initial, hard):
    out = tmp_path / 'run'; out.mkdir()
    state, binding = fixture_state()
    old = None
    if not initial:
        cp.publish_set(out, state, True, binding)
        old = cp.load_set(out / 'last.pt', binding)
        saved = tree_hashes(old['generation'])
        state, binding = fixture_state(1)
    if hard:
        request = tmp_path / 'request.pt'; torch.save(state, request)
        script = '''import os,sys,torch
from pathlib import Path
from lidc3d import checkpoints as cp
state=torch.load(sys.argv[2],map_location="cpu",weights_only=False)
def crash(stage):
    if stage==sys.argv[3]: os._exit(79)
cp._boundary=crash
cp.publish_set(Path(sys.argv[1]),state,True,state["checkpoint_binding"])
'''
        result = subprocess.run([sys.executable, '-B', '-c', script, str(out), str(request), stage],
                                capture_output=True, text=True)
        assert result.returncode == 79, result.stderr
    else:
        def crash(here):
            if here == stage:
                raise RuntimeError('synthetic publication crash at ' + stage)
        with monkeypatch.context() as patch:
            patch.setattr(cp, '_boundary', crash)
            with pytest.raises(RuntimeError, match='synthetic publication crash'):
                cp.publish_set(out, state, True, binding)
        assert len(list(out.glob('checkpoint_failure-*.json'))) == 1
    pointer = read_json(out / 'checkpoint_commit.json')
    assert pointer['status'] == ('COMMITTED' if stage == 'committed' else 'PUBLISHING')
    if stage == 'committed':
        assert cp.load_set(out / 'last.pt', binding)['last']['epoch'] == state['epoch']
    else:
        with pytest.raises(ValueError, match='Incomplete checkpoint publication'):
            cp.load_set(out / 'last.pt', binding)
        before = tree_hashes(out)
        with pytest.raises(ValueError, match='Incomplete checkpoint publication'):
            cp.publish_set(out, state, True, binding)
        assert tree_hashes(out) == before  # No automatic repair/retry.
    if old:
        assert tree_hashes(old['generation']) == saved
        assert cp.load_set(old['generation'] / 'last.pt', binding)['last']['epoch'] == 0


@pytest.mark.parametrize('name', ['last.pt', 'best.pt', 'commit.json'])
def test_corruption_rejected_before_pickle(tmp_path, monkeypatch, name):
    state, binding = fixture_state(); cp.publish_set(tmp_path, state, True, binding)
    selected = cp.load_set(tmp_path / 'last.pt', binding)
    target = selected['generation'] / name
    target.write_bytes(target.read_bytes()[:13] + b'corrupted')
    def forbidden(*args, **kwargs):
        pytest.fail('corrupt data must be rejected before pickle loading')
    monkeypatch.setattr(cp.torch, 'load', forbidden)
    with pytest.raises(ValueError, match='hash mismatch'):
        cp.load_set(tmp_path / 'last.pt', binding)


def test_unrelated_sibling_best_rejected_before_pickle(tmp_path, monkeypatch):
    a=tmp_path/'a'; b=tmp_path/'b'; a.mkdir(); b.mkdir()
    state, binding = fixture_state(); cp.publish_set(a, state, True, binding)
    other=copy.deepcopy(state); other['checkpoint_binding']['run_id']='other-run'
    cp.publish_set(b, other, True, other['checkpoint_binding'])
    shutil.copyfile(b/'best.pt', a/'best.pt')
    monkeypatch.setattr(cp.torch, 'load', lambda *a, **k: pytest.fail('unrelated sibling loaded'))
    with pytest.raises(ValueError, match='hash mismatch: best.pt'):
        cp.load_set(a/'last.pt', binding)


@pytest.mark.parametrize('field', ['run_id', 'seed', 'config', 'best_cpm', 'epoch', 'scheduler'])
def test_inconsistent_best_rejected_even_with_rehashed_inventory(tmp_path, field):
    state, binding=fixture_state(); cp.publish_set(tmp_path,state,True,binding)
    state,binding=fixture_state(1); cp.publish_set(tmp_path,state,False,binding)
    selected=cp.load_set(tmp_path/'last.pt',binding); gen=selected['generation']; best=selected['best']
    if field=='run_id': best['checkpoint_binding']['run_id']='unrelated'
    elif field=='seed': best['seed']=18
    elif field=='config': best['config']['synthetic']=False
    elif field=='best_cpm': best['best_cpm']=.4
    elif field=='epoch': best['epoch']=2; best['scheduler']['epoch']=2
    else: best['scheduler']['epoch']=9
    torch.save(best,gen/'best.pt'); shutil.copyfile(gen/'best.pt',tmp_path/'best.pt')
    record=read_json(gen/'commit.json'); record['files']['best.pt']=sha256(gen/'best.pt')
    (gen/'commit.json').write_text(json.dumps(record))
    pointer=read_json(tmp_path/'checkpoint_commit.json');pointer['commit_sha256']=sha256(gen/'commit.json')
    (tmp_path/'checkpoint_commit.json').write_text(json.dumps(pointer))
    with pytest.raises(ValueError,match='mismatch|Inconsistent'):
        cp.load_set(tmp_path/'last.pt',binding)


@pytest.mark.parametrize('key', ['config_hash','bundle_hash','seed','code_sha256','device_type','amp','run_id'])
def test_exact_expected_binding_required(tmp_path,key):
    state,binding=fixture_state(); cp.publish_set(tmp_path,state,True,binding)
    wrong={**binding,key:'different'}
    with pytest.raises(ValueError,match='Resume identity mismatch'):
        cp.load_set(tmp_path/'last.pt',wrong)


def test_interrupted_epoch_resume_parity_and_cpu_load(tmp_path,monkeypatch):
    bundle,cfg,_,_=make_bundle(tmp_path)
    cfg['training']['max_epochs']=3
    full=tmp_path/'full'; broken=tmp_path/'broken'; resumed=tmp_path/'resumed'
    loads=[]; real_load=torch.load
    def cpu_load(*args,**kwargs):
        loads.append(kwargs.get('map_location'))
        assert kwargs.get('map_location')=='cpu'
        return real_load(*args,**kwargs)
    monkeypatch.setattr(cp.torch,'load',cpu_load)
    training.train(bundle,cfg,full,17,'cpu')
    real_publish=training.publish_set
    def interrupted(*args,**kwargs):
        real_publish(*args,**kwargs)
        if args[1]['epoch']==0:
            raise RuntimeError('synthetic interruption after committed epoch')
    with monkeypatch.context() as scoped:
        scoped.setattr(training,'publish_set',interrupted)
        with pytest.raises(RuntimeError,match='after committed epoch'):
            training.train(bundle,cfg,broken,17,'cpu')
    assert not (broken/'selection.json').exists()
    assert read_json(broken/'failure.json')['status']=='INCOMPLETE'
    previous=tree_hashes(broken)
    training.train(bundle,cfg,resumed,17,'cpu',broken/'last.pt')
    assert tree_hashes(broken)==previous
    expected=cp.load_set(full/'last.pt',{})
    actual=cp.load_set(resumed/'last.pt',{})
    for part in ('last','best'):
        for key in ('model','optimizer','scaler','epoch','scheduler','loader_rng','torch_rng',
                    'numpy_rng','python_rng','cuda_rng','best_cpm','stopping_best','stale'):
            assert_same(expected[part][key],actual[part][key])
    for key in ('selected_epoch','validation_cpm','operating_threshold'):
        assert read_json(full/'selection.json')[key]==read_json(resumed/'selection.json')[key]
    original=cp.load_set(broken/'last.pt',{})
    assert original['binding']==actual['binding']
    assert expected['binding']['run_id']!=actual['binding']['run_id']
    imported=min((resumed/'checkpoints').iterdir(),key=lambda p:p.name)
    assert read_json(imported/'commit.json')['parent']['commit_sha256']==original['commit_sha256']
    assert loads and set(loads)=={'cpu'}


def test_resume_rejection_keeps_failure_evidence(tmp_path,monkeypatch):
    bundle,cfg,_,_=make_bundle(tmp_path); source=tmp_path/'source'
    training.train(bundle,cfg,source,17,'cpu')
    before=tree_hashes(source)
    identity=training.code_identity()
    monkeypatch.setattr(training,'code_identity',lambda:{**identity,'sha256':'changed-source'})
    output=tmp_path/'rejected'
    with pytest.raises(ValueError,match='Resume identity mismatch'):
        training.train(bundle,cfg,output,17,'cpu',source/'last.pt')
    assert tree_hashes(source)==before
    assert read_json(output/'failure.json')['status']=='INCOMPLETE'
    assert not (output/'selection.json').exists() and not (output/'last.pt').exists()


def test_legacy_uncommitted_checkpoint_is_not_guessed(tmp_path):
    state,binding=fixture_state();torch.save(state,tmp_path/'last.pt');torch.save(state,tmp_path/'best.pt')
    with pytest.raises(FileNotFoundError):
        cp.load_set(tmp_path/'last.pt',binding)
    with pytest.raises(ValueError,match='Uncommitted'):
        cp.publish_set(tmp_path,state,True,binding)
