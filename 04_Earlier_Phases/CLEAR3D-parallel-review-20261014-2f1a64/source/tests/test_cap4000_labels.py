"""Generated CPU-only fixtures; no population labels or research pixels."""
import copy,hashlib,json,sqlite3,subprocess,sys,tracemalloc
from pathlib import Path
import pytest
from lidc3d import cap4000_labels as m
from lidc3d.medicalnet_finetuning import TrainingSample
SPEC={'policy':'positive-cycle-negative-hash-v1','positive_draws':240,'negative_draws':720,'max_source_rows':283200,'max_epoch_rows':65536}

def region(center=(0,0,0),strict=True,radius=1):return {'center_lps':list(center),'radius_mm':radius,'strict_radius':strict}
def pool():return [{'scan_id':f's{i}','patient_id':f'p{i}','role':'train' if i<12 else 'val'} for i in range(18)]
def selection():
 return {'entries':[{'scan':{'scan_id':p['scan_id'],'patient_id':p['patient_id'],'split':p['role'],'positive':[region(),region()],'ignored':[region((1000,0,0),False,2)]},'selected':[{'fixture':j} for j in range(64)]} for p in pool()]}

def dbfixture(path,n=4000):
 db=sqlite3.connect(path);db.execute('CREATE TABLE candidates(id TEXT PRIMARY KEY,scan TEXT,role TEXT,ordinal INTEGER,payload TEXT)');db.execute('CREATE INDEX scan_order ON candidates(scan,ordinal)')
 for s in range(12):
  for i in range(n):
   row={'id':f's{s}:{i}','scan_id':f's{s}','center_lps':[0 if i<2 else 1000 if i==2 else 10,0,0]}
   db.execute('INSERT INTO candidates VALUES(?,?,?,?,?)',(row['id'],row['scan_id'],'train',s*n+i,json.dumps(row)+'\n'))
 db.commit();db.close()

@pytest.mark.parametrize('strict,center,expected',[(True,[1,0,0],0),(False,[1,0,0],1),(True,[0,0,0],1),(False,[1002,0,0],-1)])
def test_matching_exact_boundary_precedence_and_outliers(strict,center,expected):
 pos=[region(strict=strict),region(strict=strict)];ign=[region((0,0,0),True,.5),region((1000,0,0),False,2)]
 before=copy.deepcopy((pos,ign));assert m.candidate_label(center,pos,ign)==expected;assert (pos,ign)==before

def test_mask_refusal_before_label_use():
 s={'positive':[dict(region(),mask_path='/never/read.npy')],'ignored':[]}
 with pytest.raises(ValueError,match='Mask'):m.references(s)

@pytest.mark.parametrize('change',['patient','role','order','duplicate'])
def test_fixed_population_separation(change):
 s=selection();p=pool()
 if change=='patient':p[12]['patient_id']='p0';s['entries'][12]['scan']['patient_id']='p0'
 if change=='role':s['entries'][0]['scan']['split']='val'
 if change=='order':s['entries'].reverse()
 if change=='duplicate':s['entries'][1]=copy.deepcopy(s['entries'][0])
 with pytest.raises(ValueError):m.identities(s['entries'],p)

def test_full_cap_generated_disk_build_sampling_and_bounded_memory(tmp_path):
 src=tmp_path/'source.sqlite';dbfixture(src);out=tmp_path/'out';out.mkdir();s=selection();before=copy.deepcopy(s)
 tracemalloc.start();result=m.build(src,s,pool(),out);_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
 assert peak<24*1024**2 and result['rows']==48000 and result['usable']==47988
 assert all(x['labels']=={'-1':1,'0':3997,'1':2} for x in result['per_scan'])
 assert s==before and json.loads((out/'validation-unchanged.json').read_text())==s['entries'][12:]
 rows=m.LabelRows(out/'labels.sqlite',m.hash_file(out/'labels.sqlite'))
 try:
  sample=m.DiskTrainingSample(rows,SPEC,17)
  # Independent original implementation comparison over generated data only.
  original=TrainingSample(rows,SPEC,17)
  for epoch in range(3):
   sample.set_epoch(epoch);original.set_epoch(epoch)
   assert sample.indices==original.indices and sample.receipt==original.receipt
   assert sum(sample[i]['label'] for i in range(960))==240
  assert len(sample.indices)==960
 finally:rows.close()
 (tmp_path/'scale-measurement.json').write_text(json.dumps({'rows':48000,'python_peak_bytes':peak}))

def test_failure_keeps_partial_evidence(tmp_path):
 src=tmp_path/'source';dbfixture(src,1);out=tmp_path/'out';out.mkdir()
 with pytest.raises(ValueError,match='count'):m.build(src,selection(),pool(),out)
 assert (out/'selection-failure.json').exists() and not (out/'selection.json').exists()

def test_resource_refusal_preserved(tmp_path):
 src=tmp_path/'source';dbfixture(src,1);out=tmp_path/'out';out.mkdir()
 def fail():raise ValueError('resource fixture')
 with pytest.raises(ValueError,match='resource'):m.build(src,selection(),pool(),out,fail)
 assert json.loads((out/'selection-failure.json').read_text())['status']=='INCOMPLETE'

def test_sidecar_binding_and_admission(tmp_path):
 p=tmp_path/'labels';p.write_text('invalid')
 with pytest.raises(ValueError,match='hash'):m.LabelRows(p,'0'*64)
 with pytest.raises(ValueError):m.DiskTrainingSample(type('R',(),{'role':'val'})(),SPEC,17)

def test_draft_and_changed_binding_refused():
 d={'status':'ADOPTED','scope':m.SCOPE,'binding_sha256':'a','inputs':{},'limits':m.LIMITS,'reference_release':'HOLD','cap4000_policy':'NOT_ADOPTED','acknowledged_holds':m.HOLDS,'exposure_uncertainty':True,'sampled_monitoring':True,'supplied_by':'generated','instruction':'fixture','date':'fixture'}
 m.gate(d,'a',{})
 for key in ['status','scope','binding_sha256','limits','supplied_by','reference_release']:
  broken=copy.deepcopy(d);broken[key]=None
  with pytest.raises(ValueError):m.gate(broken,'a',{})

def test_default_no_torch_or_input_access():
 script=Path(__file__).parents[1]/'scripts/cap4000_label_selection.py'
 code="import runpy,sys;sys.argv=['plan'];runpy.run_path("+repr(str(script))+",run_name='plan_test');assert 'torch' not in sys.modules"
 assert subprocess.run([sys.executable,'-B','-c',code],capture_output=True).returncode==0
 p=subprocess.run([sys.executable,'-B',str(script)],capture_output=True,text=True)
 assert p.returncode==0 and json.loads(p.stdout)['status']=='PLAN_ONLY'

def test_new_checkpoint_bindings_and_resume_rejection(tmp_path):
 from test_scoped_epochs import run,setup
 from lidc3d import scoped_epoch_checkpoints as cp
 run(tmp_path/'old',stop_after=0);parent=setup()[1]
 new=m.checkpoint_identity(parent,*['a'*64]*6)
 for key in ['label_selection_sha256','label_sidecar_sha256','unchanged_validation_sha256','candidate_manifest_sha256','metadata_index_sha256','source_binding_sha256']:
  changed=copy.deepcopy(new);changed[key]='b'*64
  assert changed!=new
  with pytest.raises(ValueError,match='binding'):cp.load(tmp_path/'old',changed)

def test_disk_sampler_actual_epoch_resume_with_model_double(tmp_path):
 from test_scoped_epochs import Tiny,Small,setup
 from lidc3d import scoped_epochs as e,scoped_epochs_contract as c
 p=tmp_path/'fixture.sqlite';db=sqlite3.connect(p)
 db.execute('CREATE TABLE labels(usable_index INTEGER PRIMARY KEY,label INTEGER,payload TEXT,payload_sha256 TEXT)')
 for i in range(800):
  raw=json.dumps({'id':str(i),'fixture_index':i})
  db.execute('INSERT INTO labels VALUES(?,?,?,?)',(i,int(i<20),raw,hashlib.sha256(raw.encode()).hexdigest()))
 db.commit();db.close();rows=m.LabelRows(p,m.hash_file(p));cfg,_,_,_,_=setup()
 binding=c.identity(cfg,'b'*64,'disk-generated','research','c'*64)
 binding=m.checkpoint_identity(binding,*['a'*64]*6)
 scans=[{'scan_id':f'v{i}'} for i in range(6)];val=[{'id':f'v{i}','scan_id':f'v{i//64}','label':int(i<15)} for i in range(384)]
 def validate(*args):return ([{'id':r['id'],'scan_id':r['scan_id'],'logits':[.1,.2]} for r in val],[{'scan_id':s['scan_id'],'status':'ok','candidates':64,'predictions':64} for s in scans])
 def run(name,**kw):return e.fit(cfg,binding,tmp_path/name,'cpu',rows,scans,val,Small,validate,Tiny,epochs=2,sampling=SPEC,sampling_factory=m.DiskTrainingSample,**kw)
 try:
  run('full');run('part',stop_after=0);run('resume',resume=tmp_path/'part')
  assert e.compare(tmp_path/'full',tmp_path/'resume',binding)['exact']
 finally:rows.close()

def test_loading_uses_existing_mmap_patch_path(tmp_path):
 import numpy as np
 from lidc3d.data import CandidateDataset
 # Generated pixel fixture only; proves sequence is retained without list conversion.
 image=tmp_path/'image.npy';valid=tmp_path/'valid.npy';np.save(image,np.zeros((48,48,48),dtype=np.float32));np.save(valid,np.ones((48,48,48),dtype=bool))
 class Rows:
  validated_index=True
  def __len__(self):return 1
  def __getitem__(self,i):return {'id':'fixture','scan_id':'s','label':1,'zyx':[24,24,24]}
 rows=Rows();dataset=CandidateDataset([{'scan_id':'s','image_path':str(image),'valid_path':str(valid)}],rows,48,False,17)
 assert dataset.rows is rows
 item=dataset[0];assert item['image'].shape==(1,48,48,48) and item['label']==1
 from lidc3d.data import load_array
 assert isinstance(load_array(str(image)),np.memmap)
 # Loader field/geometry conventions remain owned by the unchanged data module;
 # pixel path is covered by existing dataset regressions, not guessed here.

@pytest.mark.parametrize('mode',['oversize','wrong-role','nonfinite'])
def test_payload_admission_and_failure_receipt(tmp_path,mode):
 src=tmp_path/'source';dbfixture(src,1);db=sqlite3.connect(src)
 if mode=='oversize':db.execute("UPDATE candidates SET payload=? WHERE id='s0:0'",('x'*16385,))
 if mode=='wrong-role':db.execute("UPDATE candidates SET role='val' WHERE id='s0:0'")
 if mode=='nonfinite':db.execute("UPDATE candidates SET payload=? WHERE id='s0:0'",(json.dumps({'id':'s0:0','scan_id':'s0','center_lps':[float('nan'),0,0]}),))
 db.commit();db.close();out=tmp_path/'out';out.mkdir()
 with pytest.raises(ValueError):m.build(src,selection(),pool(),out)
 assert (out/'selection-failure.json').exists()

def test_symlink_refusal(tmp_path):
 p=tmp_path/'p';p.write_text('fixture');link=tmp_path/'link';link.symlink_to(p)
 with pytest.raises(ValueError,match='Symlink'):m.hash_file(link)
