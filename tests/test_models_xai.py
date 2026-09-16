from pathlib import Path
import copy
import numpy as np
import pytest
import torch
import yaml
from lidc3d.models import DetectorClassifier, signature, position_3d
from lidc3d.training import parameter_groups, epoch_lr
from lidc3d.xai import integrated_gradients, normalize_map, top_mask, perturbation_curves, localization, gradcam


def config():
    return yaml.safe_load((Path(__file__).parents[1]/"configs/reference.yaml").read_text())


@pytest.mark.parametrize("fusion", ["hierarchical","concat","gated"])
def test_model_shape_gradients_and_received_attention(fusion):
    cfg = config()["model"]
    cfg["fusion"] = fusion
    model = DetectorClassifier(cfg).eval()
    x = torch.rand(1,1,16,16,16)
    result = model(x, explain=True)
    assert result["logits"].shape == (1,2)
    assert set(result["attention"]) == {'0','1','2'}
    result["logits"].sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    # Coarsest 1-token map is necessarily constant at this synthetic size;
    # degenerate-map reporting is intentionally separate from model validity.
    assert torch.allclose(result["attention"]["0"].sum(), torch.tensor(1.), atol=1e-5)
    assert result["attention"]["0"].var() > 0


def test_signature_and_parameter_partition():
    cfg=config()
    model=DetectorClassifier(cfg["model"])
    sig=signature(model,48)
    assert [v[-3:] for v in sig["features"].values()] == [[12,12,12],[6,6,6],[3,3,3]]
    groups=parameter_groups(model,cfg["training"])
    ids=[id(p) for g in groups for p in g["params"]]
    assert len(ids)==len(set(ids))==len(list(model.parameters()))


def test_sinusoidal_axis_information():
    p=position_3d((2,3,4),128,'cpu',torch.float32)
    assert p.shape==(1,24,128)
    assert not torch.allclose(p[0,0],p[0,1])
    assert not torch.allclose(p[0,0],p[0,4])


class LinearMargin(torch.nn.Module):
    def forward(self,x):
        value=x.flatten(1).sum(1)
        return torch.stack([torch.zeros_like(value),value],1)


def test_ig_linear_analytic_completeness():
    x=torch.arange(8,dtype=torch.float32).reshape(1,1,2,2,2)
    result=integrated_gradients(LinearMargin(),x,intervals=(4,8))
    assert result["complete"]
    assert torch.equal(result["signed"],x)
    assert result["residuals"][0]["residual"]==[0.]


def test_perturbation_endpoints_and_identical_ranking():
    model=LinearMargin()
    x=torch.arange(8,dtype=torch.float32).reshape(1,1,2,2,2)
    valid=np.ones((2,2,2),bool)
    result=perturbation_curves(model,x,x[0,0].numpy(),valid,torch.zeros_like(x),block=1,steps=5)
    assert result["insertion"][0]==result["deletion"][-1]==0
    assert result["insertion"][-1]==result["deletion"][0]==28
    assert result["ranking"][0]==[1,1,1]


def test_constant_maps_and_empty_masks_not_success():
    valid=np.ones((2,2,2),bool)
    with pytest.raises(ValueError,match="Degenerate"):
        normalize_map(np.ones((2,2,2)),valid)
    assert localization(np.arange(8).reshape(2,2,2),np.zeros_like(valid),valid)["dice"] is None


def test_gradcam_full_network_gradient_path():
    model=DetectorClassifier(config()["model"]).eval()
    result=gradcam(model,torch.rand(1,1,16,16,16))
    assert result["map"].shape==(1,1,16,16,16)
    assert torch.isfinite(result["map"]).all()


def test_incomplete_weighted_gradient_accumulation_matches_full_batch():
    torch.manual_seed(1)
    first=torch.nn.Linear(3,2)
    second=copy.deepcopy(first)
    x=torch.randn(5,3)
    y=torch.tensor([0,1,1,0,1])
    w=torch.tensor([1.,3.])
    torch.nn.functional.cross_entropy(first(x),y,weight=w).backward()
    for start in [0,2,4]:
        torch.nn.functional.cross_entropy(second(x[start:start+2]),y[start:start+2],weight=w,reduction='sum').backward()
    for p in second.parameters():
        p.grad/=w[y].sum()
    assert all(torch.allclose(a.grad,b.grad,atol=1e-7) for a,b in zip(first.parameters(),second.parameters()))


def test_schedule_reaches_declared_minimum():
    cfg=config()["training"]
    values=[epoch_lr(1e-4,e,cfg) for e in range(100)]
    assert values[4]==1e-4
    assert values[-1]==pytest.approx(1e-6)
    assert all(a>=b for a,b in zip(values[4:],values[5:]))
