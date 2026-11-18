"""Shared-use controls with mocked GPU queries only; never launch CUDA."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

SCRIPT=Path(__file__).parents[1]/'scripts/commission_cuda.py'
spec=importlib.util.spec_from_file_location('shared_commission',SCRIPT)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
RESIDENTS='3340885, 242\n3477966, 308\n'


def fake_query(monkeypatch,free=10452,model='NVIDIA GeForce RTX 2080 Ti',driver='535.230.02',uuid=None):
    def command(args):
        assert args[0]=='nvidia-smi'
        if '--query-compute-apps=pid,used_gpu_memory' in args:return RESIDENTS
        return f'{uuid or c.GPU_UUID}, {model}, {driver}, 11264, {free}\n'
    monkeypatch.setattr(c,'command',command)


def test_residents_require_explicit_permission_not_inherited_env(monkeypatch):
    fake_query(monkeypatch)
    monkeypatch.setenv('LIDC3D_SHARED_GPU_PERMISSION','1')
    with pytest.raises(ValueError,match='explicit shared-use permission'):
        c.gpu_preflight(c.GPU_UUID)
    result=c.gpu_preflight(c.GPU_UUID,shared_use=True)
    assert result['shared_use_permission_confirmed'] is True
    assert result['compute_processes']==[{'pid':3340885,'used_gpu_memory_mib':242},{'pid':3477966,'used_gpu_memory_mib':308}]


@pytest.mark.parametrize('value',['yes',1,None])
def test_nonboolean_permission_rejected(value):
    with pytest.raises(ValueError,match='explicit boolean'):c.plan(c.GPU_UUID,value)


def test_shared_permission_does_not_cover_another_gpu():
    with pytest.raises(ValueError,match='only the reviewed GPU'):
        c.plan('GPU-00000000-0000-0000-0000-000000000000',True)


@pytest.mark.parametrize('free',[0,10239])
def test_shared_permission_never_bypasses_free_memory(monkeypatch,free):
    fake_query(monkeypatch,free=free)
    with pytest.raises(ValueError,match='Insufficient free'):c.gpu_preflight(c.GPU_UUID,True)


def test_exact_minimum_free_memory_allowed_with_explicit_permission(monkeypatch):
    fake_query(monkeypatch,free=10240)
    assert c.gpu_preflight(c.GPU_UUID,True)['shared_use_permission_confirmed']


@pytest.mark.parametrize('kwargs',[{'driver':'999'},{'model':'other'},{'uuid':'GPU-other'}])
def test_shared_mode_preserves_identity_and_driver_checks(monkeypatch,kwargs):
    fake_query(monkeypatch,**kwargs)
    with pytest.raises(ValueError):c.gpu_preflight(c.GPU_UUID,True)


def test_shared_flag_without_execute_is_plan_only_no_torch(tmp_path):
    code='''import importlib.util,sys,subprocess
s=importlib.util.spec_from_file_location("c",sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.command=lambda *a: (_ for _ in ()).throw(AssertionError("GPU query"))
subprocess.Popen=lambda *a,**k: (_ for _ in ()).throw(AssertionError("worker launch"))
m.main(["--acknowledge-shared-gpu-permission"])
assert "torch" not in sys.modules
'''
    p=subprocess.run([sys.executable,'-B','-c',code,str(SCRIPT)],capture_output=True,text=True,check=True)
    plan=json.loads(p.stdout)
    assert plan['shared_use_permission_confirmed'] is True
    assert plan['measurement_context']=='shared-GPU; not exclusive-device performance'
    assert plan['limits']==c.LIMITS and not list(tmp_path.iterdir())


def test_cli_passes_explicit_permission_to_execute(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(c,'execute',lambda *args:calls.append(args))
    c.main(['--execute-synthetic-cuda','--output',str(tmp_path/'new'),'--acknowledge-shared-gpu-permission'])
    assert calls==[(tmp_path/'new',c.GPU_UUID,True)]


@pytest.mark.parametrize('limit',['TIME_LIMIT','RSS_LIMIT','DEVICE_MEMORY_LIMIT'])
def test_monitor_never_signals_residents_on_any_limit(tmp_path,monkeypatch,limit):
    owned=700001; signals=[]
    class Child:
        pid=owned;returncode=None
        def poll(self):return self.returncode
        def wait(self,timeout=None):self.returncode=-15;return self.returncode
    child=Child()
    def popen(*a,**kw):
        assert kw['start_new_session'] is True
        return child
    monkeypatch.setattr(c.subprocess,'Popen',popen)
    def killpg(pid,sig):
        assert pid==owned and pid not in [3340885,3477966]
        signals.append((pid,sig))
    monkeypatch.setattr(c.os,'killpg',killpg)
    monkeypatch.setattr(c.os,'kill',lambda *a:pytest.fail('direct PID signal forbidden'))
    monkeypatch.setattr(c,'rss_bytes',lambda pid:c.LIMITS['rss_bytes']+1 if limit=='RSS_LIMIT' else 0)
    monkeypatch.setattr(c,'command',lambda args:RESIDENTS+f'{owned}, {9217 if limit=="DEVICE_MEMORY_LIMIT" else 1}\n')
    log=tmp_path/'observations.jsonl'
    result=c.monitor(['fake-worker'],{},tmp_path/'worker.log',-1 if limit=='TIME_LIMIT' else 10,c.GPU_UUID,process_log=log)
    assert result['limit_failure']==limit
    assert signals==[(owned,signal.SIGTERM)]
    records=[json.loads(line) for line in log.read_text().splitlines()]
    assert [x['phase'] for x in records]==['worker-before','during-worker','worker-after']
    assert all(x['processes'][:2]==[{'pid':3340885,'used_gpu_memory_mib':242},{'pid':3477966,'used_gpu_memory_mib':308}] for x in records)


def test_observation_error_cleanup_signals_only_owned_child(tmp_path,monkeypatch):
    owned=700001; signals=[]; queries=[]
    class Child:
        pid=owned;returncode=None
        def poll(self):return self.returncode
        def wait(self,timeout=None):self.returncode=-9;return self.returncode
    monkeypatch.setattr(c.subprocess,'Popen',lambda *a,**kw:Child())
    monkeypatch.setattr(c,'rss_bytes',lambda pid:0)
    def query(args):
        queries.append(args)
        if len(queries)==2:raise RuntimeError('synthetic observation failure')
        return RESIDENTS
    monkeypatch.setattr(c,'command',query)
    def killpg(pid,sig):
        assert pid==owned;signals.append((pid,sig))
    monkeypatch.setattr(c.os,'killpg',killpg)
    with pytest.raises(RuntimeError,match='observation failure'):
        c.monitor(['fake'],{},tmp_path/'worker.log',10,c.GPU_UUID,process_log=tmp_path/'observations.jsonl')
    assert signals==[(owned,signal.SIGKILL)]
    rows=[json.loads(s) for s in (tmp_path/'observations.jsonl').read_text().splitlines()]
    assert 'error' in rows[1] and rows[-1]['phase']=='worker-after'


def test_execution_records_before_during_after_and_stops_on_failure(tmp_path,monkeypatch):
    fake_query(monkeypatch)
    calls=[]
    def failed(cmd,env,log,remaining,gpu_uuid=None,process_log=None):
        calls.append(cmd);assert env['LIDC3D_SHARED_GPU_PERMISSION']=='1'
        c.process_observation(gpu_uuid,process_log,'during-worker',700001)
        return {'exit_status':7,'limit_failure':None}
    monkeypatch.setattr(c,'monitor',failed)
    out=tmp_path/'out'
    with pytest.raises(RuntimeError,match='stage failed'):c.execute(out,c.GPU_UUID,True)
    rows=[json.loads(x) for x in (out/'gpu-process-observations.jsonl').read_text().splitlines()]
    assert [r['phase'] for r in rows]==['before-execution','during-worker','after-execution']
    report=json.loads((out/'commissioning-report.json').read_text())
    assert len(calls)==1 and report['stages']['fixture']['status']=='FAIL'
    assert all(report['stages'][s]['status']=='NOT_RUN' for s in c.STAGES[1:])
    assert report['plan']['measurement_context'].startswith('shared-GPU')


def test_final_observation_failure_is_retained(tmp_path,monkeypatch):
    fake_query(monkeypatch)
    original=c.process_observation
    def observe(uuid,path,phase,owned_pid=None):
        if phase=='after-execution':raise RuntimeError('final observation unavailable')
        return original(uuid,path,phase,owned_pid)
    monkeypatch.setattr(c,'process_observation',observe)
    monkeypatch.setattr(c,'monitor',lambda *a,**kw:{'exit_status':0,'limit_failure':None})
    out=tmp_path/'out'
    with pytest.raises(RuntimeError,match='final observation'):c.execute(out,c.GPU_UUID,True)
    report=json.loads((out/'commissioning-report.json').read_text())
    assert report['status']=='INCOMPLETE' and 'observation_failure' in report
