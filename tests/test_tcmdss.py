import numpy as np
import pandas as pd
import pytest
import torch
from src.data.protocol import assert_normal, assert_disjoint
from src.data.corruptions import corrupt,FAMILIES
from src.diffusion.model import MaskedUNet
from src.diffusion.process import training_masks,covering_masks,denoising_loss,reconstruct
from src.semantic.features import balanced_memory,cosine_knn,FrozenDINO
from src.spectral.features import window_features,fit_spectral,score_spectral
from src.latent.model import fit_latent,score_latent
from src.fusion.calibration import fit_calibration,fuse,VARIANTS
from src.evaluation.metrics import metrics,bootstrap_auc

torch.set_num_threads(2)


def normal(n=10):
    return pd.DataFrame({'is_anomaly':0,'origin':'hirise_clean','class_id':np.arange(n)%8})


def test_normal_guard():
    frame=normal();assert_normal(frame)
    frame.loc[0,'origin']='test_corruption'
    with pytest.raises(ValueError):assert_normal(frame)
    frame=normal();frame.loc[0,'is_anomaly']=1
    with pytest.raises(ValueError):assert_normal(frame)


def test_masks_and_diffusion():
    torch.manual_seed(42)
    mask=training_masks(6,32)
    assert ((mask.mean((1,2,3))>=.09)&(mask.mean((1,2,3))<=.4)).all()
    assert torch.all(covering_masks(32).sum(0)==1)
    model=MaskedUNet(8);image=torch.rand(2,1,32,32)
    loss=denoising_loss(model,image,20);loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None for p in model.parameters())
    reconstruction,heat=reconstruct(model,image,20,3)
    assert reconstruction.shape==heat.shape==image.shape
    assert torch.allclose(reconstruct(model,image,20,3)[0],reconstruction)
    assert torch.allclose(reconstruct(model,image[:1],20,3)[0],reconstruction[:1],atol=1e-3)


def test_memory_and_knn():
    tokens=torch.nn.functional.normalize(torch.randn(3,6,8),dim=-1)
    bank=balanced_memory(tokens,12)
    assert bank.shape==(12,8)
    assert torch.equal(bank,balanced_memory(tokens,12))
    actual=cosine_knn(tokens,bank,k=5,query_chunk=2,bank_chunk=3)
    expected=(1-(tokens.reshape(-1,8)@bank.T).topk(5,dim=-1).values).mean(-1).reshape(3,6)
    assert torch.allclose(actual,expected,atol=1e-6)


def test_dense_dino_shape_without_network(monkeypatch):
    class Tiny(torch.nn.Module):
        num_prefix_tokens=1
        pretrained_cfg={'input_size':(3,32,32)}
        def forward_features(self,x):
            return torch.ones(len(x),5,8)
    monkeypatch.setattr('src.semantic.features.timm.create_model',lambda *a,**kw:Tiny())
    model=FrozenDINO('test_dino')
    tokens,side=model(torch.rand(2,1,32,32))
    assert tokens.shape==(2,4,8) and side==2
    assert not tokens.requires_grad


def test_spectral_latent():
    images=torch.rand(8,1,32,32)
    features,grid=window_features(images,8)
    assert features.shape==(8,16,8)
    state=fit_spectral(images,windows=(8,16))
    score,heat=score_spectral(images,state)
    assert heat.shape==images.shape and torch.isfinite(score).all()
    latent=np.random.default_rng(42).normal(size=(20,16))
    fitted=fit_latent(latent,8)
    distance=score_latent(latent,fitted)
    assert distance.shape==(20,) and np.isfinite(distance).all()


def test_calibration_normal_only():
    scores=np.random.default_rng(42).normal(size=(10,4))
    state=fit_calibration(normal(),scores)
    final=fuse(scores,state)
    assert np.isclose(np.quantile(final,.99),state['thresholds']['V3_TCMD_SS'])
    assert (final>=0).all()
    assert VARIANTS['V3_TCMD_SS']==[.4,.25,.2,.15]
    frame=normal();frame.loc[0,'is_anomaly']=1
    with pytest.raises(ValueError):fit_calibration(frame,scores)


@pytest.mark.parametrize('kind',list(FAMILIES))
def test_corruptions(kind):
    image=np.random.default_rng(1).random((32,32)).astype(np.float32)
    changed,mask=corrupt(image,kind,.7,42)
    assert np.array_equal(changed,corrupt(image,kind,.7,42)[0])
    assert changed.shape==mask.shape==image.shape and np.isfinite(changed).all()
    assert 0<=changed.min()<=changed.max()<=1 and mask.sum()>0


def test_metrics_and_cluster_bootstrap():
    frame=pd.DataFrame({'base_id':['a','a','b','b'],'is_anomaly':[0,1,0,1],'score':[0,1,0,1]})
    result=metrics(frame.is_anomaly,frame.score,.5)
    assert result['auroc']==1 and result['f1']==1 and result['false_positive_rate']==0
    assert bootstrap_auc(frame,repeats=20)['lower']==1
