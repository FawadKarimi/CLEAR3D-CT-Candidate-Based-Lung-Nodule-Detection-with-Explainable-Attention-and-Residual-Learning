"""CPU tests. Real CPU GradScaler tests are NOT CUDA compatibility evidence.

Explicit injections/mocks test dispatch and failure gates only.
"""
import copy
import importlib.util
import json
from pathlib import Path
import pytest
import torch
from torch import nn
from lidc3d import optimization as op, training, checkpoints as cp
from lidc3d.provenance import read_json, canonical_hash
from test_checkpoint_sets import fixture_state, assert_same, tree_hashes
from test_ingest_and_guards import make_bundle


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.tensor([1.,2.]))
    def forward(self,x):return x*self.weight


def setup(amp=True):
    model=Tiny();optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    scaler=torch.amp.GradScaler('cpu',enabled=amp)
    return model,optimizer,scaler,op.new_update_state(),[]


def boundary(model,optimizer,scaler,state,events,kind='finite',denominator=1.):
    optimizer.zero_grad(set_to_none=True)
    # Deliberate CPU fp16 autograd fixture: default scale65536 overflows the
    # fp16 derivative, but the underlying unscaled derivative is finite (2).
    if kind=='half':loss=(model.weight.to(torch.float16)*2).float().sum()
    elif kind=='singular':loss=torch.sqrt(model.weight-model.weight.detach()).sum()
    else:loss=model.weight.sum()
    scaler.scale(loss).backward()
    if kind=='injected':model.weight.grad[0]=float('inf')
    return op.finish_accumulation(model,optimizer,scaler,denominator,1.,scaler.is_enabled(),
                                  state,events.append,{'epoch':0,'synthetic_kind':kind})


def test_real_cpu_scaler_skip_backoff_and_recovery():
    m,o,s,c,events=setup();before=m.weight.detach().clone()
    assert s.get_scale()==65536
    assert not boundary(m,o,s,c,events,'half')
    assert torch.equal(m.weight,before) and s.get_scale()==32768
    assert not boundary(m,o,s,c,events,'half')
    assert s.get_scale()==16384
    assert boundary(m,o,s,c,events,'half')
    assert not torch.equal(m.weight,before)
    assert c['boundaries']==3 and c['optimizer_updates']==1 and c['skipped_updates']==2
    assert c['scaler_update_calls']==3 and c['scale_changes']==2 and c['consecutive_skips']==0
    assert [e['optimizer_updates'] for e in events]==[0,0,1]
    assert all(e['cause']=='NOT_ESTABLISHED' for e in events)
    op.validate_update_state(c)
    assert not torch.cuda.is_initialized()


def test_singular_derivative_is_not_mislabeled_loss_scale_cause():
    m,o,s,c,events=setup()
    assert not boundary(m,o,s,c,events,'singular')
    assert events[-1]['classification']=='NONFINITE_INDIVIDUAL_GRADIENTS'
    assert events[-1]['cause']=='NOT_ESTABLISHED'
    assert events[-1]['unscaled_gradients']['nonfinite']


@pytest.mark.parametrize('amp',[False,True])
def test_finite_gradient_norm_overflow_is_fatal_not_scaler_recovery(amp,monkeypatch):
    m,o,s,c,events=setup(amp);s.scale(m.weight.sum()).backward()
    m.weight.grad.fill_(1e20*s.get_scale())
    before=m.weight.detach().clone()
    monkeypatch.setattr(nn.utils,'clip_grad_norm_',lambda *a,**k:pytest.fail('must not clip an invalid norm'))
    with pytest.raises(RuntimeError,match='non-finite'):
        op.finish_accumulation(m,o,s,1.,1.,amp,c,events.append,{})
    assert events[-1]['classification']=='NONFINITE_NORM_WITH_FINITE_GRADIENTS'
    assert not events[-1]['unscaled_gradients']['nonfinite']
    assert torch.isfinite(m.weight.grad).all() and torch.equal(m.weight,before)
    assert c['optimizer_updates']==c['skipped_updates']==c['scaler_update_calls']==0


def test_non_amp_nonfinite_is_fatal_and_never_clipped(monkeypatch):
    m,o,s,c,events=setup(False)
    monkeypatch.setattr(nn.utils,'clip_grad_norm_',lambda *a,**k:pytest.fail('nonfinite clipped'))
    with pytest.raises(FloatingPointError,match='non-AMP'):boundary(m,o,s,c,events,'injected')
    assert events[-1]['status']=='FAIL' and c['boundaries']==0


def test_clipping_guard_retained_on_finite_gradients(monkeypatch):
    real=nn.utils.clip_grad_norm_;calls=[]
    def traced(params,*args,**kwargs):
        params=list(params);assert all(torch.isfinite(p.grad).all() for p in params)
        assert kwargs['error_if_nonfinite'] is True;calls.append(1)
        return real(params,*args,**kwargs)
    monkeypatch.setattr(nn.utils,'clip_grad_norm_',traced)
    m,o,s,c,events=setup();assert boundary(m,o,s,c,events)
    assert calls==[1]


@pytest.mark.parametrize('bound',['consecutive','total'])
def test_skip_limits_are_enforced_before_extra_dispatch(bound):
    m,o,s,c,events=setup()
    if bound=='consecutive':
        for _ in range(op.POLICY['max_consecutive_skips']):assert not boundary(m,o,s,c,events,'injected')
    else:
        for _ in range(op.POLICY['max_total_skips']):
            assert not boundary(m,o,s,c,events,'injected');assert boundary(m,o,s,c,events)
    scale=s.get_scale();old=copy.deepcopy(c)
    with pytest.raises(FloatingPointError,match='exhausted'):boundary(m,o,s,c,events,'injected')
    assert c==old and s.get_scale()==scale
    assert events[-1]['classification']=='AMP_SKIP_BOUND_EXHAUSTED'


def test_scale_growth_is_not_counted_as_a_skip():
    m,o,s,c,events=setup();s=torch.amp.GradScaler('cpu',growth_interval=1)
    assert boundary(m,o,s,c,events)
    assert s.get_scale()==131072 and c['scale_changes']==1 and c['skipped_updates']==0
    assert c['optimizer_updates']==1


@pytest.mark.parametrize('microbatch,accumulation,n',[(2,3,7),(3,2,7),(2,8,5),(1,1,4)])
def test_weighted_accumulation_matches_concatenated_effective_batches(microbatch,accumulation,n):
    torch.manual_seed(11);model=nn.Linear(3,2).double();reference=copy.deepcopy(model)
    x=torch.randn(n,3,dtype=torch.float64);y=torch.arange(n)%2;weights=torch.tensor([1.,3.],dtype=torch.float64)
    loader=[{'image':x[i:i+microbatch],'label':y[i:i+microbatch]} for i in range(0,n,microbatch)]
    opt=torch.optim.SGD(model.parameters(),lr=.03);refopt=torch.optim.SGD(reference.parameters(),lr=.03)
    events=[];state=op.new_update_state()
    op.train_epoch(model,loader,opt,torch.amp.GradScaler('cpu',enabled=False),weights,torch.device('cpu'),False,accumulation,1e6,state,events.append,0)
    expected_denominators=[]
    for i in range(0,n,microbatch*accumulation):
        xx=x[i:i+microbatch*accumulation];yy=y[i:i+microbatch*accumulation]
        refopt.zero_grad();loss=nn.functional.cross_entropy(reference(xx),yy,weight=weights,reduction='sum')/weights[yy].sum()
        loss.backward();nn.utils.clip_grad_norm_(reference.parameters(),1e6,error_if_nonfinite=True);refopt.step()
        expected_denominators.append(float(weights[yy].sum()))
    for a,b in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    assert [e['denominator'] for e in events]==expected_denominators
    assert state['optimizer_updates']==len(expected_denominators)
    assert events[-1]['incomplete_group']==(len(loader)%accumulation!=0)


def test_nonfinite_loss_stops_before_backward_or_dispatch(tmp_path):
    class Bad(nn.Module):
        def __init__(self):super().__init__();self.p=nn.Parameter(torch.ones(2))
        def forward(self,x):return self.p[None].expand(x.shape[0],2)*float('nan')
    m=Bad();o=torch.optim.SGD(m.parameters(),lr=.01);s=torch.amp.GradScaler('cpu',enabled=False);c=op.new_update_state()
    log=tmp_path/'events.jsonl'
    with pytest.raises(FloatingPointError,match='training loss'):
        op.train_epoch(m,[{'image':torch.ones(1,1),'label':torch.tensor([0])}],o,s,torch.tensor([1.,3.]),torch.device('cpu'),False,1,1.,c,lambda e:op.append_diagnostic(log,e),0)
    event=json.loads(log.read_text());assert event['classification']=='NONFINITE_UNSCALED_LOSS'
    assert m.p.grad is None and c['optimizer_updates']==0


def test_zero_update_epoch_blocks_validation_selection_checkpoint(tmp_path,monkeypatch):
    # Dispatch mock, not CUDA evidence: simulate skips at every boundary.
    import lidc3d.optimization as module
    def skipped(model,optimizer,scaler,denominator,max_norm,amp,state,emit,context):
        state['boundaries']+=1;state['skipped_updates']+=1;state['scaler_update_calls']+=1
        state['consecutive_skips']+=1;optimizer.zero_grad(set_to_none=True)
        emit({'kind':'mock_skip','context':context});return False
    monkeypatch.setattr(module,'finish_accumulation',skipped)
    monkeypatch.setattr(training,'predict',lambda *a,**kw:pytest.fail('zero-update validation'))
    bundle,cfg,_,_=make_bundle(tmp_path);out=tmp_path/'run'
    with pytest.raises(FloatingPointError,match='zero optimizer updates'):training.train(bundle,cfg,out,17,'cpu')
    assert read_json(out/'failure.json')['status']=='INCOMPLETE'
    assert not (out/'last.pt').exists() and not (out/'selection.json').exists()
    assert 'ZERO_OPTIMIZER_UPDATES' in (out/'optimization_boundaries.jsonl').read_text()


def test_cpu_scaler_checkpoint_resume_preserves_skip_history(tmp_path):
    m,o,s,c,events=setup()
    for _ in range(3):boundary(m,o,s,c,events,'half')
    state,binding=fixture_state();binding['optimization_policy_sha256']=canonical_hash(op.POLICY)
    state.update(model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(o.state_dict()),scaler=copy.deepcopy(s.state_dict()),update_state=copy.deepcopy(c))
    cp.publish_set(tmp_path,state,True,binding)
    saved=tree_hashes(tmp_path);loaded=cp.load_set(tmp_path/'last.pt',binding)['last']
    m2,o2,s2,c2,ev2=setup();m2.load_state_dict(loaded['model']);o2.load_state_dict(loaded['optimizer']);s2.load_state_dict(loaded['scaler']);c2=op.validate_update_state(loaded['update_state'])
    boundary(m,o,s,c,events,'half');boundary(m2,o2,s2,c2,ev2,'half')
    assert_same(m.state_dict(),m2.state_dict());assert_same(o.state_dict(),o2.state_dict())
    assert s.state_dict()==s2.state_dict() and c==c2 and events[-1]==ev2[-1]
    assert tree_hashes(tmp_path)==saved


@pytest.mark.parametrize('field',['optimizer_updates','skipped_updates','scaler_update_calls','scale_changes','consecutive_skips','policy'])
def test_inconsistent_resume_counters_rejected(field):
    c=op.new_update_state();c[field]=999 if field!='policy' else {}
    with pytest.raises(ValueError):op.validate_update_state(c)


def test_real_cpu_epoch_resume_preserves_new_counters(tmp_path,monkeypatch):
    bundle,cfg,_,_=make_bundle(tmp_path);cfg['training']['max_epochs']=3
    full=tmp_path/'full';broken=tmp_path/'broken';resumed=tmp_path/'resumed'
    training.train(bundle,cfg,full,17,'cpu');real=training.publish_set
    def interrupt(*args,**kw):
        real(*args,**kw)
        if args[1]['epoch']==0:raise InterruptedError('synthetic epoch boundary')
    with monkeypatch.context() as scoped:
        scoped.setattr(training,'publish_set',interrupt)
        with pytest.raises(InterruptedError):training.train(bundle,cfg,broken,17,'cpu')
    before=tree_hashes(broken);training.train(bundle,cfg,resumed,17,'cpu',broken/'last.pt')
    a=cp.load_set(full/'last.pt',{});b=cp.load_set(resumed/'last.pt',{})
    for part in ['last','best']:
        for key in ['model','optimizer','scaler','update_state','torch_rng','loader_rng']:
            assert_same(a[part][key],b[part][key])
    assert tree_hashes(broken)==before


def test_actual_scaled_probe_cpu_dispatch_and_zero_update_guard(tmp_path,monkeypatch):
    path=Path(__file__).parents[1]/'scripts/commission_cuda.py'
    spec=importlib.util.spec_from_file_location('amp_commission',path);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    bundle,cfg=c.generated_fixture(tmp_path/'fixture',small=True)
    result=c.scaled_weighted_probe(bundle,cfg,tmp_path/'probe','cpu')
    assert result['actual_training_path'] and result['update_state']['optimizer_updates']>0
    assert Path(result['boundary_evidence']).exists()
    assert c.plan()['training_candidates']==64
    assert c.LIMITS['total_seconds']==900 and c.LIMITS['worker_seconds']==180
    assert not torch.cuda.is_initialized()


def test_nonfinite_diagnostic_serializes_honestly(tmp_path):
    p=tmp_path/'diag.jsonl';op.append_diagnostic(p,{'value':float('inf'),'nested':[float('nan')]})
    assert json.loads(p.read_text())=={'value':'inf','nested':['nan']}
