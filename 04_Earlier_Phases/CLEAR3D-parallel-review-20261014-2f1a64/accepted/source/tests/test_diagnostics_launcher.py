import copy,json,sys,subprocess,hashlib,os
from pathlib import Path
import pytest,torch
from lidc3d.diagnostics import Diagnostics,POLICY,sample
from lidc3d.optimization import train_epoch,new_update_state
from lidc3d.scoped_epochs import exact
from lidc3d.metadata_index import build_index
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import cap4000_review as launch

class Model(torch.nn.Module):
 def __init__(self):
  super().__init__();self.encoder=torch.nn.Linear(3,4);self.head=torch.nn.Linear(4,2);self.drop=torch.nn.Dropout(.25)
 def forward(self,x):return self.head(self.drop(torch.relu(self.encoder(x))))

def one(path,enabled):
 torch.manual_seed(19);m=Model();o=torch.optim.AdamW(m.parameters(),lr=.002);state=new_update_state();events=[]
 x=torch.arange(21,dtype=torch.float32).reshape(7,3)/10;y=torch.tensor([0,1,1,0,1,0,0]);batches=[{'image':x[i:i+2],'label':y[i:i+2]} for i in range(0,7,2)]
 d=Diagnostics(path) if enabled else None
 loss=train_epoch(m,batches,o,torch.amp.GradScaler('cuda',enabled=False),torch.tensor([1.,3.]),torch.device('cpu'),False,3,1.,state,events.append,0,diagnostics=d)
 return m.state_dict(),o.state_dict(),state,torch.get_rng_state(),loss,d

def test_exact_optimization_and_rng(tmp_path):
 a=one(tmp_path/'off',False);b=one(tmp_path/'on',True)
 assert all(exact(x,y) for x,y in zip(a[:5],b[:5]))
 assert b[-1].before=={} and b[-1].handles==[]
 events=[json.loads(s) for s in (tmp_path/'on').read_text().splitlines()]
 loss=[s for s in events if s['kind']=='class_losses'];assert sum(x['classes']['1']['count'] for x in loss)==3
 for x in loss:assert x['classes']['1']['weighted_loss_sum']==pytest.approx(3*x['classes']['1']['unweighted_loss_sum'])
 boundaries=[s for s in events if s['kind']=='optimizer_boundary'];assert len(boundaries)==2
 assert all(x['actual_updates']==1 and x['groups'][0]['lr']==.002 for x in boundaries)
 assert all(x['gradient_norm_after_clip']<=1.00001 for x in boundaries)
 assert sum(x['sample_count'] for x in boundaries[0]['families_sampled'].values())<=POLICY['parameter_values_total']
 assert any(x['changed_values'] for x in boundaries[0]['families_sampled'].values())
 assert any(x['kind']=='activation_sample' for x in events)

def test_noncontiguous_bounded_samples_no_rng():
 x=torch.arange(60000.).reshape(200,300).t();rng=torch.get_rng_state().clone();v=sample(x,256)
 assert len(v)==256 and torch.equal(rng,torch.get_rng_state()) and v[0]==0

def test_hooks_removed_after_failure(tmp_path):
 m=Model();o=torch.optim.AdamW(m.parameters());d=Diagnostics(tmp_path/'log');s=new_update_state()
 with pytest.raises(FloatingPointError):train_epoch(m,[{'image':torch.full((2,3),float('nan')),'label':torch.tensor([0,1])}],o,torch.amp.GradScaler('cuda',enabled=False),torch.tensor([1.,3.]),torch.device('cpu'),False,2,1,s,lambda e:None,0,d)
 assert not d.handles and not m.encoder._forward_hooks

def test_scoped_resume_with_instrumentation(tmp_path):
 from test_scoped_epochs import setup,Small,Tiny
 from lidc3d import scoped_epochs as e,scoped_epochs_contract as c
 cfg,b,rows,scans,val=setup();cfg['diagnostics']=POLICY;b=c.identity(cfg,'b'*64,'instrumented-fixture','generated')
 def fit(name,**kw):return e.fit(cfg,b,tmp_path/name,'cpu',rows,scans,val,Small,e.generated_validate,Tiny,epochs=2,sampling=c.GENERATED_SAMPLING,**kw)
 fit('full');fit('part',stop_after=0);fit('resume',resume=tmp_path/'part');assert e.compare(tmp_path/'full',tmp_path/'resume',b)['exact']

def test_default_plan_and_label_gate():
 p=Path(launch.__file__);r=subprocess.run([sys.executable,'-B',str(p)],capture_output=True,text=True)
 assert r.returncode==0 and json.loads(r.stdout)['torch_imported'] is False
 r=subprocess.run([sys.executable,'-B',str(p),'--execute','label-index'],capture_output=True,text=True);assert r.returncode!=0

def test_draft_and_changed_disposition_refused():
 d={'status':'DRAFT'}
 with pytest.raises(ValueError):launch.disposition(d,'hash','build')
 d={'status':'ADOPTED','supplied_by':'synthetic','instruction':'fixture','scope':'cap4000-metadata-only-v1','binding_sha256':'hash','preparation_sha256':launch.PREPARATION_SHA256,'policy':launch.POLICY,'limits':launch.LIMITS,'reference_release':'HOLD','policy_status':'NOT_ADOPTED','labels_authorized':False,'actions':['build','index'],'manifest_sha256':'m'}
 launch.disposition(d,'hash','build')
 with pytest.raises(ValueError):launch.disposition(d,'changed','build')
 with pytest.raises(ValueError):launch.disposition(d,'hash','index','wrong')

def test_metadata_index_zero_scan_and_no_labels(tmp_path):
 p=tmp_path/'rows';p.write_text(json.dumps({'id':'s:candidate:000000','scan_id':'s','zyx':[0,0,0]})+'\n');h=hashlib.sha256(p.read_bytes()).hexdigest();o=tmp_path/'index';o.mkdir()
 r=build_index(p,h,[{'scan_id':'s','patient_id':'p','split':'train'},{'scan_id':'zero','patient_id':'q','split':'val'}],o,lambda **kw:None)
 assert r['rows']==1 and r['per_scan']['zero']==0 and not r['labels_evaluated'] and not r['training_ready']

@pytest.mark.parametrize('mode',['hash','duplicate','budget','label','patient'])
def test_index_refusals(tmp_path,mode):
 row={'id':'s:0','scan_id':'s'}
 if mode=='label':row['label']=1
 raw=(json.dumps(row)+'\n')*(2 if mode=='duplicate' else 1);p=tmp_path/'rows';p.write_text(raw);h=hashlib.sha256(p.read_bytes()).hexdigest();o=tmp_path/'index';o.mkdir();scans=[{'scan_id':'s','patient_id':'p','split':'train'}]
 if mode=='hash':h='0'*64
 if mode=='patient':scans.append({'scan_id':'z','patient_id':'p','split':'val'})
 def tick(**kw):
  if mode=='budget':raise ValueError('budget')
 with pytest.raises(Exception):build_index(p,h,scans,o,tick)
 assert not (o/'metadata-index.json').exists()

def test_shell_returns_actual_worker_status(tmp_path):
 root=tmp_path/'fixture';scripts=root/'work/scripts';scripts.mkdir(parents=True);evidence=tmp_path/'evidence';evidence.mkdir()
 source=Path(launch.__file__).with_name('cap4000_command.sh').read_text();source=source.replace('/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/LIDC3D_reviews_and_zips_oct07_09/LIDC3D-offline-cpu-20261007T181841Z-IIGwzD/env/bin/python',sys.executable)
 shell=scripts/'cap4000_command.sh';shell.write_text(source);(scripts/'cap4000_review.py').write_text('raise SystemExit(7)\n')
 result=subprocess.run(['/bin/sh',str(shell)],env=dict(os.environ,CAP4000_EVIDENCE=str(evidence)),capture_output=True)
 assert result.returncode==7 and (evidence/'worker.exit').read_text().strip()=='7'

def test_validation_diagnostic_sums_and_empty_class(tmp_path):
 d=Diagnostics(tmp_path/'log');d.validation([{'id':'a','logits':[0.,0.]},{'id':'b','logits':[0.,0.]}],{'a':1,'b':1},[1.,3.],0)
 r=json.loads((tmp_path/'log').read_text());assert r['rows']==2 and r['classes']['0']['probability_mean'] is None
 assert r['classes']['1']['unweighted_loss_sum']==pytest.approx(2*__import__('math').log(2))
 assert r['classes']['1']['weighted_loss_sum']==pytest.approx(6*__import__('math').log(2))

def test_mocked_launcher_failure_and_freshness(tmp_path,monkeypatch):
 from types import SimpleNamespace
 root=tmp_path/'review';work=root/'work';work.mkdir(parents=True);(root/'runs').mkdir();prepared=tmp_path/'prepared';prepared.mkdir();raw=b'{}';(prepared/'preparation.json').write_bytes(raw);ph=hashlib.sha256(raw).hexdigest()
 identities=[{'scan_id':str(i),'split':'train' if i<708 else 'val','patient_id':str(i)} for i in range(797)];ip=root/'identities.json';ip.write_text(json.dumps({'identities':identities}));rp=root/'receipts.jsonl';rp.write_text('{}\n')
 bound=lambda p:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
 b={'preparation_sha256':ph,'policy':launch.POLICY,'limits':launch.LIMITS,'source_root':str(work),'reference_release':'HOLD','prepared':str(prepared),'identities':bound(ip),'receipts':bound(rp),'outputs':{'build':str(root/'runs/cap4000-build-001')}}
 monkeypatch.setattr(launch,'WORK',work);monkeypatch.setattr(launch,'PREPARATION_SHA256',ph);monkeypatch.setattr(launch.guards,'bindings',lambda *args:b);monkeypatch.setattr(launch,'resources',lambda:(1,100*1024**3));monkeypatch.setattr(launch.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
 monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');monkeypatch.setenv('PYTHONNOUSERSITE','1')
 for k in ('PYTHONHOME','PYTHONPATH','LD_LIBRARY_PATH','LD_PRELOAD'):monkeypatch.delenv(k,raising=False)
 class Budget:
  def __init__(self,out):pass
  def check(self,**kw):pass
  def summary(self):return {'mocked':True}
 monkeypatch.setattr(launch,'Cap4000Budget',Budget)
 d={'status':'ADOPTED','supplied_by':'synthetic','instruction':'fixture','scope':'cap4000-metadata-only-v1','binding_sha256':'h','preparation_sha256':ph,'policy':launch.POLICY,'limits':launch.LIMITS,'reference_release':'HOLD','policy_status':'NOT_ADOPTED','labels_authorized':False,'actions':['build']};dp=root/'adopted.json';dp.write_text(json.dumps(d));dh=hashlib.sha256(dp.read_bytes()).hexdigest()
 def fail(*args,**kw):raise ValueError('generated stream failure')
 monkeypatch.setattr(launch,'build',fail)
 args=['--execute','build','--binding','fixture','--binding-sha256','h','--disposition',str(dp),'--disposition-sha256',dh,'--acknowledge-sampled-risk']
 assert launch.main(args)==1;out=Path(b['outputs']['build']);assert json.loads((out/'launcher-failure.json').read_text())['worker_exit']==1
 before={str(p):p.read_bytes() for p in out.iterdir()};assert launch.main(args)==1;assert before=={str(p):p.read_bytes() for p in out.iterdir()}

def test_opt_in_configuration_preserves_numerical_settings():
 from lidc3d import scoped_epochs_contract as c
 old=c.config(18,'weight','provenance','a'*64);new=c.config(18,'weight','provenance','a'*64,diagnostics=True)
 assert new.pop('diagnostics')==POLICY and new==old
 new['diagnostics']=POLICY;c.check_config(new);new['diagnostics']={**POLICY,'activation_values':1000}
 with pytest.raises(ValueError):c.check_config(new)
