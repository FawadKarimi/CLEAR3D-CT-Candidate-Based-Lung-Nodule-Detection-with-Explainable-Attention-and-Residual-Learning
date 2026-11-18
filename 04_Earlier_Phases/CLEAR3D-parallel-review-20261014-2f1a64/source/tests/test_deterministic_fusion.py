"""CPU numerical equivalence: no CUDA allocation or real inputs."""
import importlib.util
import json
from pathlib import Path
import pytest
import torch
import yaml
from torch.nn import functional as F
from lidc3d import models

ROOT = Path(__file__).parents[1]
# Fixed acceptance bounds declared before measurement, not repeat-run tolerances.
TOL = {torch.float64: (2e-12, 2e-12), torch.float32: (2e-5, 6e-6),
       torch.float16: (3e-3, 8e-3), torch.bfloat16: (3e-2, 6e-2)}
CASES = [((2,128,3,3,3),(6,6,6)), ((2,128,6,6,6),(12,12,12)),
         ((2,128,3,3,3),(12,12,12)), ((2,128,12,12,12),(12,12,12)),
         ((1,3,1,2,3),(4,5,7)), ((1,2,5,7,9),(3,4,2)),
         ((1,2,3,4,5),(1,1,1)), ((1,2,1,1,1),(1,3,5)),
         ((1,2,3,5,7),(7,8,11))]

@pytest.mark.parametrize('dtype', list(TOL))
@pytest.mark.parametrize('shape,size', CASES)
def test_values_and_input_gradients(shape,size,dtype):
    torch.manual_seed(81)
    x=torch.randn(shape,dtype=dtype,requires_grad=True)
    original=x.detach().clone().requires_grad_()
    actual=models.trilinear_resize(x,size)
    expected=F.interpolate(original,size=size,mode='trilinear',align_corners=False)
    probe=torch.randn_like(actual)
    actual.backward(probe);expected.backward(probe)
    rtol,atol=TOL[dtype]
    torch.testing.assert_close(actual,expected,rtol=rtol,atol=atol)
    torch.testing.assert_close(x.grad,original.grad,rtol=rtol,atol=atol)
    assert actual.dtype==dtype and actual.shape==(*shape[:2],*size)
    print(json.dumps({'shape':shape,'size':size,'dtype':str(dtype),
        'forward_max_abs':float((actual-expected).abs().max()),
        'gradient_max_abs':float((x.grad-original.grad).abs().max()),'rtol':rtol,'atol':atol}))

@pytest.mark.parametrize('pattern',['constant','ramp','corner','noncontiguous'])
def test_boundary_patterns_and_noncontiguous_gradient(pattern):
    x=torch.arange(24,dtype=torch.float64).reshape(1,1,2,3,4)
    if pattern=='constant':x.fill_(2)
    if pattern=='corner':x.zero_();x[0,0,0,0,0]=1
    if pattern=='noncontiguous':x=x.transpose(-1,-2)
    x=x.detach().requires_grad_();y=x.detach().clone().requires_grad_()
    a=models.trilinear_resize(x,(5,7,9));b=F.interpolate(y,size=(5,7,9),mode='trilinear',align_corners=False)
    grad=torch.linspace(-1,1,a.numel(),dtype=a.dtype).reshape_as(a)
    a.backward(grad);b.backward(grad)
    torch.testing.assert_close(a,b,rtol=2e-12,atol=2e-12)
    torch.testing.assert_close(x.grad,y.grad,rtol=2e-12,atol=2e-12)

@pytest.mark.parametrize('dtype', list(TOL))
@pytest.mark.parametrize('spatial', [(12,12,12),(3,3,3),(1,2,7)])
def test_global_pool_values_and_gradients(dtype,spatial):
    torch.manual_seed(51)
    x=torch.randn(2,128,*spatial,dtype=dtype,requires_grad=True)
    y=x.detach().clone().requires_grad_()
    a=models.global_average_pool3d(x);b=F.adaptive_avg_pool3d(y,1)
    g=torch.randn_like(a);a.backward(g);b.backward(g)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)

@pytest.mark.parametrize('fusion',['hierarchical','concat','gated'])
def test_actual_model_production_forward_and_input_gradient(monkeypatch,fusion):
    cfg=yaml.safe_load((ROOT/'configs/reference.yaml').read_text())['model']
    cfg=dict(cfg,fusion=fusion)
    torch.manual_seed(17);model=models.DetectorClassifier(cfg).eval()
    x=torch.randn(1,1,48,48,48,requires_grad=True)
    a=model(x);a.backward(torch.tensor([[.7,-.2]]));grad=x.grad.clone()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.zero_grad(set_to_none=True)
    y=x.detach().clone().requires_grad_()
    monkeypatch.setattr(models,'trilinear_resize',lambda x,size:F.interpolate(x,size=size,mode='trilinear',align_corners=False))
    monkeypatch.setattr(models,'global_average_pool3d',lambda x:F.adaptive_avg_pool3d(x,1))
    b=model(y);b.backward(torch.tensor([[.7,-.2]]))
    torch.testing.assert_close(a,b,rtol=2e-5,atol=6e-6)
    # Full nonlinear model: allow small amplification of interpolation roundoff.
    torch.testing.assert_close(grad,y.grad,rtol=2e-3,atol=2e-6)
    print(json.dumps({'fusion':fusion,'logits_max_abs':float((a-b).abs().max()),
                     'input_gradient_max_abs':float((grad-y.grad).abs().max())}))


def test_gradcheck_and_strict_mode():
    prior=torch.are_deterministic_algorithms_enabled()
    warn=torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        x=torch.randn(1,1,2,2,2,dtype=torch.float64,requires_grad=True)
        assert torch.autograd.gradcheck(lambda v:models.trilinear_resize(v,(3,4,5)),(x,))
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    finally:torch.use_deterministic_algorithms(prior,warn_only=warn)


def test_backward_probe_order_and_failure_prevents_repeats(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('probe_harness',ROOT/'scripts/commission_cuda.py')
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    assert c.STAGES.index('probes') < c.STAGES.index('full-backward') < c.STAGES.index('repeat-a')
    monkeypatch.setattr(c,'gpu_preflight',lambda *a,**kw:{})
    ran=[]
    def monitor(cmd,env,log,*a,**kw):
        stage=cmd[cmd.index('--worker')+1];ran.append(stage)
        Path(log).write_text('synthetic dispatch only')
        return {'exit_status':1 if stage=='full-backward' else 0,'limit_failure':None}
    monkeypatch.setattr(c,'monitor',monitor)
    with pytest.raises(RuntimeError):c.execute(tmp_path/'out',c.GPU_UUID)
    report=json.loads((tmp_path/'out/commissioning-report.json').read_text())
    assert ran==['fixture','probes','full-backward']
    assert report['stages']['full-backward']['status']=='FAIL'
    assert all(report['stages'][s]['status']=='NOT_RUN' for s in c.STAGES[3:])


def test_production_backward_probe_on_cpu():
    spec=importlib.util.spec_from_file_location('probe_cpu',ROOT/'scripts/commission_cuda.py')
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    cfg=yaml.safe_load((ROOT/'configs/reference.yaml').read_text())
    result=c.full_model_backward_probe(cfg,'cpu')
    assert result['FP32']['input_shape']==[2,1,48,48,48]
    assert result['FP32']['parameter_gradient_tensors']>100
    assert result['FP32']['input_gradient_finite_nonzero']
    assert not torch.cuda.is_initialized()
