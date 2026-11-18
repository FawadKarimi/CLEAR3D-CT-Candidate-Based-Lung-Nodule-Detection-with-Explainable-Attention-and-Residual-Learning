"""Generated CPU fixtures only; these results are not CUDA qualification."""
import copy,json,os,subprocess,sys
from pathlib import Path
import pytest,torch
from lidc3d import medicalnet_contract as c,medicalnet_qualification as q,engineering as e
from lidc3d.mixed_engineering import CONTRACT
from lidc3d.optimization import new_update_state

ROOT=Path(__file__).resolve().parents[2]
def adopted():
 return dict(status='ADOPTED',scope='medicalnet-matched-engineering-v1',reference_release='HOLD',source_binding_sha256='s',selection_sha256=c.SELECTION,preparation_sha256=c.PREPARATION,policy=copy.deepcopy(c.POLICY),limits=copy.deepcopy(c.LIMITS),acknowledged_unresolved_holds=e.HOLD_IDS,supplied_by='synthetic fixture only',instruction='generated test fixture, not an adoption',date='synthetic',**{k:True for k in ('label_guided_sampling','prior_exposure_uncertainty','sampled_monitoring','not_scientific_approval','fp32_not_amp_coverage')})

@pytest.mark.parametrize('field',['status','scope','source_binding_sha256','selection_sha256','preparation_sha256','policy','limits','acknowledged_unresolved_holds','supplied_by','label_guided_sampling','prior_exposure_uncertainty','sampled_monitoring','not_scientific_approval','fp32_not_amp_coverage'])
def test_disposition_refuses_missing_or_changed(field):
 d=adopted();assert c.disposition(d,'s')==d;d.pop(field)
 with pytest.raises(ValueError):c.disposition(d,'s')

def test_mirror_is_not_official_approval():
 for d in ({},{'approved':True},{'status':'MIRROR_HASH_MATCH'},{'status':'OFFICIAL_BYTES_MATCH','distribution_url':'mirror'}):
  with pytest.raises(ValueError):c.provenance(d)

@pytest.mark.parametrize('field',['transfer','source_binding_sha256','device','stages','exact_repeat','updates_per_repeat','skips'])
def test_generated_result_gates(field):
 d=dict(status='PASS',depth=18,source_binding_sha256='s',transfer=c.transfer(18),device='cuda',stages=['pool_ties_backward','shortcut_backward','full_hybrid_repeat'],exact_repeat=True,updates_per_repeat=1,skips=0)
 assert c.generated_acceptance(d,18,'s')==d;d.pop(field)
 with pytest.raises(ValueError):c.generated_acceptance(d,18,'s')

def test_plan_no_torch_no_gpu_no_files(tmp_path):
 script=ROOT/'work/scripts/medicalnet_watchdog.py'
 result=subprocess.run([sys.executable,'-B',str(script)],cwd=tmp_path,capture_output=True,text=True,check=True)
 d=json.loads(result.stdout);assert d['status']=='PLAN_ONLY' and not d['torch_imported'];assert not list(tmp_path.iterdir())

def test_operator_probes_cpu_and_exact_comparison():
 events=[];q.operator_probes('cpu',events.append);assert len(events)==8 and all(x['status']=='PASS' for x in events)
 a={'x':torch.tensor([1.]),'v':[1,2]};b=copy.deepcopy(a);assert q.exact(a,b);b['x'][0]+=1;assert not q.exact(a,b)

@pytest.mark.parametrize('depth',[18,10])
def test_real_transfer_repeated_weighted_update_cpu(tmp_path,depth):
 cfg=CONTRACT['config'];binding=c.identity(depth,cfg,'source','fixture','test')
 events=[];a=q.generated_repeat(depth,ROOT/f'weights/resnet_{depth}.pth',cfg,tmp_path/'a','cpu',binding,events.append)
 b=q.generated_repeat(depth,ROOT/f'weights/resnet_{depth}.pth',cfg,tmp_path/'b','cpu',binding,events.append)
 assert q.exact(a,b) and a['update_state']['optimizer_updates']==1 and len(events)==2
 assert all(x['denominator']==11 and x['incomplete_group'] for x in events)
 loaded=e.load_checkpoint(tmp_path/'a',binding);assert all(v.device.type=='cpu' for v in loaded['state']['model'].values())
 alien=copy.deepcopy(binding);alien['transfer']['depth']=10 if depth==18 else 18
 with pytest.raises(ValueError):e.load_checkpoint(tmp_path/'a',alien)

@pytest.mark.parametrize('change',['depth','checkpoint_sha256','bridges','frozen_bn_statistics','shortcut_A','maxpool_backward'])
def test_checkpoint_binds_transfer_semantics(tmp_path,change):
 cfg={'fixture':True};binding=c.identity(18,cfg,'source','selection','run')
 state={'stage':'INITIALIZED','config':cfg,'selection_sha256':'selection','update_state':new_update_state(),'model':{'w':torch.ones(1)}}
 e.publish_checkpoint(tmp_path,state,binding);assert e.load_checkpoint(tmp_path,binding)['state']['stage']=='INITIALIZED'
 other=copy.deepcopy(binding);other['transfer'][change]='changed'
 with pytest.raises(ValueError):e.load_checkpoint(tmp_path,other)

def test_incomplete_transfer_publication_retains_previous(tmp_path,monkeypatch):
 from lidc3d import checkpoints
 cfg={};binding=c.identity(18,cfg,'source','selection','run');state={'stage':'INITIALIZED','config':cfg,'selection_sha256':'selection','update_state':new_update_state()}
 e.publish_checkpoint(tmp_path,state,binding);commits=list((tmp_path/'engineering-checkpoints').glob('*/commit.json'));old=commits[0].read_bytes()
 def crash(name):
  if name=='engineering_state':raise RuntimeError('synthetic interruption')
 monkeypatch.setattr(checkpoints,'_boundary',crash)
 with pytest.raises(RuntimeError):e.publish_checkpoint(tmp_path,{**state,'stage':'TRAINED'},binding)
 with pytest.raises(ValueError):e.load_checkpoint(tmp_path,binding)
 assert commits[0].read_bytes()==old and list(tmp_path.glob('checkpoint-failure-*'))
