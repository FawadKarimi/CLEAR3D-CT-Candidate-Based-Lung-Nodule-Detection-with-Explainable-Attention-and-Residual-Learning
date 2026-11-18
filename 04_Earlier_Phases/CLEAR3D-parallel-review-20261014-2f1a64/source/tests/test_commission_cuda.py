"""CPU-only commissioning-harness tests. No CUDA worker is executed."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from lidc3d import training
from lidc3d.provenance import read_json,read_jsonl

SCRIPT=Path(__file__).parents[1]/'scripts/commission_cuda.py'
spec=importlib.util.spec_from_file_location('commission_cuda',SCRIPT)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def test_default_plan_does_not_import_torch_or_launch_worker(tmp_path):
    code='''import importlib.util,sys
s=importlib.util.spec_from_file_location("commission",sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.main([])
assert "torch" not in sys.modules
'''
    p=subprocess.run([sys.executable,'-B','-c',code,str(SCRIPT)],capture_output=True,text=True,check=True)
    plan=json.loads(p.stdout)
    assert plan['reference_release']=='HOLD' and plan['limits']['total_seconds']==900
    assert plan['model'].startswith('unchanged')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('uuid',['0','GPU-0','GPU-abc,other','', 'all'])
def test_implicit_or_multiple_gpu_selection_rejected(uuid):
    with pytest.raises(ValueError):c.plan(uuid)


def test_internal_worker_is_guarded():
    env=os.environ.copy();env.pop('LIDC3D_SYNTHETIC_COMMISSION',None);env['CUDA_VISIBLE_DEVICES']=''
    p=subprocess.run([sys.executable,'-B',str(SCRIPT),'--worker','probes'],env=env,capture_output=True,text=True)
    assert p.returncode!=0 and 'requires explicit parent' in p.stderr


@pytest.mark.parametrize('case',['busy','memory','driver','identity'])
def test_gpu_preflight_fails_closed_without_gpu_calls(monkeypatch,case):
    uuid=c.GPU_UUID
    def fake(args):
        if '--query-compute-apps=pid,used_gpu_memory' in args:return '123, 242\n' if case=='busy' else ''
        return f"{uuid if case!='identity' else 'GPU-other'}, NVIDIA GeForce RTX 2080 Ti, {'535.230.02' if case!='driver' else '999.0'}, 11264, {11000 if case!='memory' else 100}\n"
    monkeypatch.setattr(c,'command',fake)
    with pytest.raises(ValueError):c.gpu_preflight(uuid)


def test_monitor_kills_only_owned_timeout_child(tmp_path):
    result=c.monitor([sys.executable,'-c','import time;time.sleep(10)'],os.environ.copy(),tmp_path/'child.log',.15)
    assert result['limit_failure']=='TIME_LIMIT' and result['exit_status']!=0
    assert result['wall_seconds']<6 and (tmp_path/'child.log').exists()


def test_monitor_rss_limit_is_enforced(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'rss_bytes',lambda pid:c.LIMITS['rss_bytes']+1)
    result=c.monitor([sys.executable,'-c','import time;time.sleep(10)'],os.environ.copy(),tmp_path/'child.log',10)
    assert result['limit_failure']=='RSS_LIMIT' and result['exit_status']!=0


def test_failure_is_retained_and_later_stages_not_run(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(c,'gpu_preflight',lambda uuid,shared_use=False:{'synthetic_preflight':True})
    def failed(cmd,env,log,remaining,gpu_uuid=None):
        calls.append(cmd);Path(log).write_text('synthetic worker failure')
        return {'exit_status':7,'limit_failure':None}
    monkeypatch.setattr(c,'monitor',failed)
    out=tmp_path/'failed'
    with pytest.raises(RuntimeError,match='stage failed'):c.execute(out,c.GPU_UUID)
    r=read_json(out/'commissioning-report.json')
    assert len(calls)==1 and r['status']=='INCOMPLETE' and r['stages']['fixture']['status']=='FAIL'
    assert all(r['stages'][s]['status']=='NOT_RUN' for s in c.STAGES[1:])
    before={p.name:p.read_bytes() for p in out.iterdir()}
    with pytest.raises(FileExistsError):c.execute(out,c.GPU_UUID)
    assert {p.name:p.read_bytes() for p in out.iterdir()}==before


def test_preflight_failure_preserved(tmp_path,monkeypatch):
    def reject(uuid,shared_use=False):raise ValueError('synthetic resident process')
    monkeypatch.setattr(c,'gpu_preflight',reject)
    out=tmp_path/'failed'
    with pytest.raises(ValueError):c.execute(out,c.GPU_UUID)
    r=read_json(out/'commissioning-report.json')
    assert all(s['status']=='NOT_RUN' for s in r['stages'].values())
    assert 'resident process' in r['failure']['reason']


def test_exact_comparison_and_cpu_tensor_checks():
    c.equal({'a':torch.tensor([1.])},{'a':torch.tensor([1.])})
    c.cpu_tensors({'a':[torch.tensor([1.])]})
    with pytest.raises(AssertionError):c.equal(torch.tensor([1.]),torch.tensor([1.00001]))
    with pytest.raises(AssertionError):c.equal([1],[1,2])
    with pytest.raises(AssertionError):c.equal(torch.tensor([1.]),torch.tensor([1.],dtype=torch.float64))


def test_generated_fixture_and_real_runner_cpu(tmp_path):
    bundle,cfg=c.generated_fixture(tmp_path/'fixture',small=True)
    assert bundle['scope']=='SYNTHETIC_TEST_FIXTURE_NOT_A_RELEASE'
    scans=read_jsonl(bundle['scans']);rows=read_jsonl(bundle['candidates'])
    assert len(scans)==3 and len(rows)==6
    assert all(s['scan_id'].startswith('SYNTHETIC-') for s in scans)
    assert all(Path(s['image_path']).is_relative_to(tmp_path) for s in scans)
    out=tmp_path/'cpu'
    training.train(bundle,cfg,out,17,'cpu')
    statuses=read_jsonl(out/'validation_scan_status.jsonl')
    assert statuses[-1]['candidates']==0 and all(s['status']=='ok' for s in statuses)
    assert read_json(out/'selection.json')['status']=='TRAINING_COMPLETE'


def test_runtime_mismatch_rejects_before_cuda_initialization(monkeypatch):
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata,'version',lambda name:'WRONG')
    with pytest.raises(ValueError,match='pin mismatch'):c.runtime_checks()
    assert not torch.cuda.is_initialized()


def test_wheel_lock_preserves_exact_archive_hash_and_rejects_duplicates(tmp_path):
    import hashlib
    import zipfile
    path=SCRIPT.with_name('make_wheel_lock.py')
    spec=importlib.util.spec_from_file_location('wheel_lock',path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    with pytest.raises(ValueError,match='Empty'):helper.lock_lines(tmp_path)
    wheel=tmp_path/'synthetic-1-py3-none-any.whl'
    with zipfile.ZipFile(wheel,'w') as z:z.writestr('synthetic-1.dist-info/METADATA','Name: synthetic\nVersion: 1\n')
    assert helper.lock_lines(tmp_path)==['synthetic==1 --hash=sha256:'+hashlib.sha256(wheel.read_bytes()).hexdigest()]
    (tmp_path/'synthetic-1-1-py3-none-any.whl').write_bytes(wheel.read_bytes())
    with pytest.raises(ValueError,match='Duplicate'):helper.lock_lines(tmp_path)


def test_monitor_exception_is_not_misreported_as_not_run(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'gpu_preflight',lambda uuid,shared_use=False:{})
    def failed(*a,**kw):raise RuntimeError('synthetic monitor error')
    monkeypatch.setattr(c,'monitor',failed)
    out=tmp_path/'failed'
    with pytest.raises(RuntimeError):c.execute(out,c.GPU_UUID)
    report=read_json(out/'commissioning-report.json')
    assert report['stages']['fixture']['status']=='FAIL'
    assert all(report['stages'][s]['status']=='NOT_RUN' for s in c.STAGES[1:])
    assert (out/'fixture-started.json').exists()
