import copy,hashlib,json,math,tracemalloc
import pytest,torch
from lidc3d import cap4000 as c,accounting as a
from lidc3d.cap4000_index import CandidateIndex4000,flatten_manifest
from lidc3d.indexed import CandidateIndex
from lidc3d.training import epoch_lr,parameter_groups
from lidc3d.optimization import train_epoch,new_update_state

def stream(p,n):
    with p.open('wb') as f:
        for i in range(n):
            row={'scan_id':'s','id':f's:candidate:{i:06d}','log_response':(i%7)/10,'zyx':[i,0,0],'center_lps':[i,0,0]}
            f.write(c.canonical(row)+b'\n')
    return {'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}

def test_cap4000_matches_stable_sort_and_prefix(tmp_path):
    p=tmp_path/'stream';bound=stream(p,6000);accepted=a.account_stream(p,'s',bound,6000)
    rows,receipt=c.select_stream(p,'s',bound,6000,accepted)
    original=[json.loads(x) for x in p.read_bytes().splitlines()];ordered=sorted(enumerate(original),key=lambda x:(-x[1]['log_response'],x[1]['id'],x[0]))
    assert [x['source_ordinal'] for x in rows]==[i for i,r in ordered if r['log_response']>=.005][:4000]
    assert receipt['accepted_cap400_prefix']==accepted['thresholds']['0.005'] and len(rows)==4000
    assert c.POLICY['cap']==4000 and a.CAP==400

@pytest.mark.parametrize('mode',['hash','prefix','tail','budget'])
def test_complete_stream_refusal(tmp_path,mode):
    p=tmp_path/'stream';bound=stream(p,5000);accepted=a.account_stream(p,'s',bound,5000)
    if mode=='hash':bound['sha256']='0'*64
    if mode=='prefix':accepted['thresholds']['0.005']['ranked_keys_sha256']='0'*64
    if mode=='tail':p.write_bytes(p.read_bytes()+b'bad\n')
    def tick(**kw):
        if mode=='budget':raise ValueError('budget')
    with pytest.raises(ValueError):c.select_stream(p,'s',bound,5000,accepted,tick)

def test_generated_heap_bounded(tmp_path):
    p=tmp_path/'stream';bound=stream(p,24000);accepted=a.account_stream(p,'s',bound,24000)
    tracemalloc.start();rows,_=c.select_stream(p,'s',bound,24000,accepted);peak=tracemalloc.get_traced_memory()[1];tracemalloc.stop()
    assert len(rows)==4000 and peak<40*1024**2
    print('CAP4000_HEAP_PEAK_BYTES',peak)

def test_index_extension_and_original_guard(tmp_path):
    p=tmp_path/'rows';stream(p,4001)
    scans=[{'scan_id':'s','split':'train','positive':[],'ignored':[]},{'scan_id':'zero','split':'val','positive':[],'ignored':[]}]
    old=tmp_path/'old';old.mkdir()
    with pytest.raises(ValueError,match='cap'):CandidateIndex(p,scans,old)
    too=tmp_path/'too';too.mkdir()
    with pytest.raises(ValueError,match='cap'):CandidateIndex4000(p,scans,too)
    p.write_bytes(b''.join(p.read_bytes().splitlines(keepends=True)[:4000]));out=tmp_path/'ok';out.mkdir()
    index=CandidateIndex4000(p,scans,out)
    assert len(index.train)==4000 and len(index.val)==0 and index.receipt['per_scan']['zero']==0
    assert index.receipt['limits']['total']==3188000;index.close()

@pytest.mark.parametrize('sizes',[[2,2,1],[2,2,2,2]])
def test_weighted_accumulation_matches_full_group_gradient(sizes):
    torch.manual_seed(18);model=torch.nn.Linear(3,2).double();reference=copy.deepcopy(model)
    x=torch.randn(sum(sizes),3,dtype=torch.double);y=torch.tensor([i%2 for i in range(sum(sizes))]);weights=torch.tensor([1.,3.],dtype=torch.double)
    batches=[];start=0
    for size in sizes:batches.append({'image':x[start:start+size],'label':y[start:start+size]});start+=size
    opt=torch.optim.SGD(model.parameters(),lr=.1);ref=torch.optim.SGD(reference.parameters(),lr=.1)
    loss=torch.nn.functional.cross_entropy(reference(x),y,weight=weights,reduction='sum')/weights[y].sum();loss.backward();ref.step()
    state=new_update_state();events=[]
    train_epoch(model,batches,opt,torch.amp.GradScaler('cuda',enabled=False),weights,torch.device('cpu'),False,8,1e6,state,events.append,0)
    for p,q in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(p,q,rtol=1e-12,atol=1e-12)
    assert state['optimizer_updates']==1 and events[0]['denominator']==float(weights[y].sum())

def test_scheduler_and_balanced_constant_baseline():
    cfg={'warmup_epochs':1,'max_epochs':3,'min_lr':1e-6}
    assert [epoch_lr(1e-4,e,cfg) for e in range(3)]==pytest.approx([1e-4,5.05e-5,1e-6])
    assert [epoch_lr(5e-5,e,cfg) for e in range(3)]==pytest.approx([5e-5,2.55e-5,1e-6])
    y=torch.tensor([1]*240+[0]*720);w=torch.tensor([1.,3.]);loss=torch.nn.functional.cross_entropy(torch.zeros(960,2),y,weight=w)
    assert loss.item()==pytest.approx(math.log(2));assert w[y].sum()==1440

def test_parameter_family_coverage_and_bn_affine():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__();self.encoder=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.BatchNorm1d(2));self.transformers=torch.nn.Linear(2,2);self.head=torch.nn.Linear(2,2)
    m=M();groups=parameter_groups(m,{'lr_cnn':1e-4,'lr_head':1e-4,'lr_transformer':5e-5,'weight_decay':1e-4})
    assert sum(len(g['params']) for g in groups)==len(list(m.parameters()))
    assert len({id(p) for g in groups for p in g['params']})==len(list(m.parameters()))
    assert next(g for g in groups if 'encoder.1.weight' in g['names'])['weight_decay']==1e-4

def manifest_fixture(tmp_path):
    root=tmp_path/'manifest';root.mkdir();(root/'scans').mkdir();p=tmp_path/'stream';bound=stream(p,5000);accepted=a.account_stream(p,'s',bound,5000)
    rows,receipt=c.select_stream(p,'s',bound,5000,accepted);identity={'scan_id':'s','split':'train','patient_id':'p'}
    scan={'identity':identity,'records':rows,'receipt':receipt};raw=c.canonical(scan);(root/'scans/s.json').write_bytes(raw)
    manifest={'policy':c.POLICY,'status':'COMPLETE_UNFROZEN','reference_release':'HOLD','identities':[identity],'files':{'scans/s.json':{'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}},'candidate_counts':{'train':4000,'val':0}}
    raw=c.canonical(manifest);(root/'candidate-manifest.json').write_bytes(raw)
    return root,hashlib.sha256(raw).hexdigest()

def test_flatten_verified_manifest_and_disk_index(tmp_path):
    root,h=manifest_fixture(tmp_path);receipt=flatten_manifest(root,h,tmp_path/'rows.jsonl')
    assert receipt['rows']==4000 and receipt['labels_evaluated'] is False
    out=tmp_path/'index';out.mkdir();index=CandidateIndex4000(tmp_path/'rows.jsonl',[{'scan_id':'s','split':'train','positive':[],'ignored':[]}],out)
    assert len(index.train)==4000 and index.receipt['source_sha256']==receipt['sha256'];index.close()

@pytest.mark.parametrize('mode',['hash','file','budget'])
def test_flatten_failure_retains_evidence(tmp_path,mode):
    root,h=manifest_fixture(tmp_path)
    if mode=='hash':h='0'*64
    if mode=='file':(root/'scans/s.json').write_bytes(b'{}')
    def tick(**kw):
        if mode=='budget':raise ValueError('resource refusal')
    with pytest.raises(ValueError):flatten_manifest(root,h,tmp_path/'rows.jsonl',tick)
    assert not (tmp_path/'rows.receipt.json').exists()
    if mode!='hash':assert (tmp_path/'rows.failure.json').exists()

def test_index_cooperative_refusal(tmp_path):
    p=tmp_path/'rows';stream(p,100);out=tmp_path/'index';out.mkdir();calls=[]
    def tick(**kw):
        calls.append(1)
        if len(calls)>3:raise ValueError('resource refusal')
    with pytest.raises(ValueError,match='resource refusal'):CandidateIndex4000(p,[{'scan_id':'s','split':'train','positive':[]}],out,tick)
    assert not (out/'candidate-index.json').exists()
