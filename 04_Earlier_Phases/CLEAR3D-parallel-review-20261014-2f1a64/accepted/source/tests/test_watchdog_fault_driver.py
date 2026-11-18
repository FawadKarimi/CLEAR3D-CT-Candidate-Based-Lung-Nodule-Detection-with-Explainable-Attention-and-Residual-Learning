"""No fault-driver execution, subprocess launch, signalling, cgroup, GPU or data access."""
import copy,json,signal,sys
from pathlib import Path
import pytest
S=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(S))
import watchdog_fault_driver as f
import user_watchdog as w

@pytest.fixture(autouse=True)
def never_execute(monkeypatch):
 def forbidden(*a,**k):raise AssertionError('Unit test attempted actual process or signal operation')
 monkeypatch.setattr(w.subprocess,'Popen',forbidden);monkeypatch.setattr(f.os,'fork',forbidden);monkeypatch.setattr(f.os,'kill',forbidden);monkeypatch.setattr(signal,'pidfd_send_signal',forbidden);monkeypatch.setattr(f,'libc',forbidden)

def test_plan_all_not_run_and_no_execution():
 p=f.planned();assert p['status']=='PLAN_ONLY' and len(p['cases'])==7 and all(x['observed']=='NOT_RUN' for x in p['cases']);assert not p['execution_authorized'];assert p['reference_release']=='HOLD'

def test_default_main_is_plan_only(capsys):
 assert f.main([])==0;assert json.loads(capsys.readouterr().out)['status']=='PLAN_ONLY'

@pytest.mark.parametrize('argv,match',[
 (['--execute'],'acknowledgement'),
 (['--execute','--acknowledge-owned-fault-signals'],'Bindings'),
 (['--role','worker','--case','rss','--out','unused'],'inherited gate')])
def test_refuse_missing_prerequisites_before_process(argv,match):
 with pytest.raises(RuntimeError,match=match):f.main(argv)

@pytest.mark.parametrize('case',f.CASES)
def test_test_limits_leave_execution_defaults_unchanged(case):
 before=copy.deepcopy(w.GENERATED_LIMITS);pilot=copy.deepcopy(w.PILOT_LIMITS);l=f.test_limits(case,20*f.MIB)
 assert w.GENERATED_LIMITS==before and w.PILOT_LIMITS==pilot and l is not w.GENERATED_LIMITS
 assert l['rss_bytes']==(28*f.MIB if case=='rss' else 96*f.MIB)
 assert l['sample_seconds']==.1 and l['total_seconds']==12

@pytest.mark.parametrize('bad',[-1,65*f.MIB])
def test_unexpected_fixture_baseline_rejected(bad):
 with pytest.raises(RuntimeError,match='baseline'):f.test_limits('rss',bad)

def test_exact_threshold_is_safe_but_small_buffer_crosses():
 l=f.test_limits('rss',20*f.MIB);b=w.SampleBudget(0,l)
 assert b.check(0,'training',28*f.MIB,None,2*1024**3) is None
 assert b.check(.1,'training',52*f.MIB,None,2*1024**3)=='SAMPLED_RSS_LIMIT'

def test_sleep_timeout_uses_reviewed_budget():
 l=f.test_limits('timeout',20*f.MIB);b=w.SampleBudget(0,l);assert b.check(0,'training',20*f.MIB,None,2*1024**3) is None
 assert b.check(2.1,'training',20*f.MIB,None,2*1024**3)=='RUN_TIME_LIMIT'

def test_checkpoint_mutation_is_detected(tmp_path):
 original=f.fixture(tmp_path);assert f.preserved(tmp_path,original)
 (tmp_path/'committed-checkpoint.fixture').write_bytes(b'changed');assert not f.preserved(tmp_path,original)

def prep(tmp_path,case):
 checkpoint=f.fixture(tmp_path)
 control={'worker_absent':True,'worker_rescue_signals':[],'deadline_exceeded':False,'case_exit':0,'case_signals':[],'adopted_worker_signal':None}
 reason={'rss':'SAMPLED_RSS_LIMIT','timeout':'RUN_TIME_LIMIT','child_creation':'WORKER_EXIT_42','sigint':'EXTERNAL_TERMINATION','sigterm':'EXTERNAL_TERMINATION'}.get(case)
 if reason:w.save(tmp_path/'watchdog-result.json',{'status':'FAIL','reason':'RuntimeError: '+reason,'cleanup':{'worker_exit':-15}})
 if case=='rss':w.save(tmp_path/'buffer-touched.json',{'bytes':f.BUFFER_BYTES})
 if case=='child_creation':w.save(tmp_path/'child-rejection.json',{'rejected':True})
 if case=='pid_identity':
  w.save(tmp_path/'pid-refusal.json',{'error':'Refuse signal: PID identity changed','signal_calls_during_mismatch':[]});w.save(tmp_path/'case-process-result.json',{'cleanup':{'worker_exit':-15}})
 if case in ['sigint','sigterm']:control['case_signals']=[case.upper()]
 if case=='watchdog_death':control.update(case_exit=-signal.SIGKILL,adopted_worker_signal=signal.SIGKILL,case_signals=['SIGKILL'])
 return checkpoint,control

@pytest.mark.parametrize('case',f.CASES)
def test_expected_fault_is_distinguished_from_success(tmp_path,case):
 checkpoint,control=prep(tmp_path,case);result=f.assess(case,tmp_path,checkpoint,control);assert result['status']=='PASS'
 assert result['expected']==f.EXPECTED[case]

@pytest.mark.parametrize('field,value',[
 ('worker_absent',False),('worker_rescue_signals',['SIGKILL']),('deadline_exceeded',True),('case_exit',2)])
def test_cleanup_or_deadline_failure_cannot_pass(tmp_path,field,value):
 checkpoint,control=prep(tmp_path,'rss');control[field]=value;assert f.assess('rss',tmp_path,checkpoint,control)['status']=='FAIL'

def test_parent_death_requires_actual_adopted_signal_evidence(tmp_path):
 checkpoint,control=prep(tmp_path,'watchdog_death');control['adopted_worker_signal']=None
 assert f.assess('watchdog_death',tmp_path,checkpoint,control)['status']=='FAIL'

def test_invented_watchdog_result_after_death_is_rejected(tmp_path):
 checkpoint,control=prep(tmp_path,'watchdog_death');w.save(tmp_path/'watchdog-result.json',{'status':'PASS'})
 assert f.assess('watchdog_death',tmp_path,checkpoint,control)['status']=='FAIL'

def test_pid_mismatch_calls_reviewed_refusal_without_signalling(monkeypatch):
 class FakeProc:
  pid=4321
  def poll(self):return None
 obj=object.__new__(w.OwnedWorker);obj.proc=FakeProc();obj.identity={'start_ticks':100};obj.fd=9
 monkeypatch.setattr(w,'process_identity',lambda pid:{'pid':pid,'start_ticks':101})
 with pytest.raises(RuntimeError,match='identity changed'):obj.send(signal.SIGTERM)

def test_child_hook_rejects_python_fork_before_actual_os_call():
 with pytest.raises(RuntimeError,match='Single-process'):f.prohibit_processes('os.fork',())

def test_unexpected_signal_invalidates_pid_case(tmp_path):
 checkpoint,control=prep(tmp_path,'pid_identity');w.save(tmp_path/'pid-refusal.json',{'error':'PID identity changed','signal_calls_during_mismatch':[['unexpected']]})
 assert f.assess('pid_identity',tmp_path,checkpoint,control)['status']=='FAIL'

def test_descendant_observation_cannot_pass(tmp_path):
 checkpoint,control=prep(tmp_path,'rss');(tmp_path/'samples.jsonl').write_text(json.dumps({'owned_worker':{'children':[9876]}})+chr(10))
 assert f.assess('rss',tmp_path,checkpoint,control)['status']=='FAIL'

def test_case_failure_artifact_cannot_pass(tmp_path):
 checkpoint,control=prep(tmp_path,'timeout');w.save(tmp_path/'case-process-failure.json',{'error':'fixture failed'})
 assert f.assess('timeout',tmp_path,checkpoint,control)['status']=='FAIL'

def test_worker_command_is_fixed_driver_not_arbitrary_executable(tmp_path):
 argv=f.child_command('worker','rss',tmp_path)
 assert argv[0]==sys.executable and argv[2]==str(Path(f.__file__).resolve()) and argv[-1]==str(tmp_path)

def test_no_replacement_supervisor_and_no_gpu_path():
 import ast
 tree=ast.parse(Path(f.__file__).read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
 assert any(isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='w' and n.func.attr=='supervise' for n in calls)
 assert not any(isinstance(n,ast.ClassDef) for n in ast.walk(tree))
 imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
 assert not any((isinstance(n,ast.Import) and any(a.name.startswith('torch') for a in n.names)) or (isinstance(n,ast.ImportFrom) and (n.module or '').startswith('torch')) for n in imports)

@pytest.mark.parametrize('mutation,fd,allowed',[
 ({},8,True),({'pid':987},8,False),({'start_ticks':901},8,False),({'ppid':999},8,False),({},None,False)])
def test_rescue_requires_exact_adopted_owned_worker(mutation,fd,allowed):
 ready={'pid':123,'start_ticks':900};current={'pid':123,'start_ticks':900,'ppid':456};current.update(mutation)
 assert f.rescue_allowed(ready,current,fd,456)==allowed

def test_controller_error_cannot_be_hidden(tmp_path):
 checkpoint,control=prep(tmp_path,'rss');control['error']='pidfd acquisition failed'
 assert f.assess('rss',tmp_path,checkpoint,control)['status']=='FAIL'
