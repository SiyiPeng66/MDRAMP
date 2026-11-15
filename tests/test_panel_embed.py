import pytest
import torch
from mdramp.panel_embed import query_chunked_sdpa

@pytest.mark.parametrize('mask_kind',['none','full','broadcast'])
def test_full_keys_match(mask_kind):
    torch.manual_seed(7)
    q=torch.randn(2,3,17,8); k=torch.randn(2,3,19,8); v=torch.randn(2,3,19,8)
    mask=None if mask_kind=='none' else torch.rand(2,1,17 if mask_kind=='full' else 1,19)>.2
    original=torch.nn.functional.scaled_dot_product_attention
    assert torch.allclose(original(q,k,v,mask),query_chunked_sdpa(original,5)(q,k,v,mask),atol=1e-6)

def test_reject_semantics_change():
    q=torch.randn(1,2,5,4)
    f=query_chunked_sdpa(torch.nn.functional.scaled_dot_product_attention,2)
    with pytest.raises(ValueError):f(q,q,q,is_causal=True)
    with pytest.raises(ValueError):f(q,q,q,dropout_p=.1)

def test_frameless_shortcut_preserves_bias_and_valid_frame_fallback(monkeypatch):
    from types import SimpleNamespace
    from esm.layers.geom_attention import GeometricReasoningOriginalImpl
    from mdramp.panel_embed import chunked_runtime
    from src.features import build_esm3_embeddings as build
    original = lambda self,s,*args: s + 17
    monkeypatch.setattr(GeometricReasoningOriginalImpl,'forward',original)
    monkeypatch.setattr(build,'_runtime_fingerprint',lambda weights:{'base':1})
    layer=SimpleNamespace(mask_and_zero_frameless=True,out_proj=torch.nn.Linear(6,4,bias=True))
    s=torch.randn(1,7,4); mask=torch.zeros(1,7,dtype=torch.bool)
    with chunked_runtime(3):
        out=GeometricReasoningOriginalImpl.forward(layer,s,None,mask,None,None)
        assert torch.equal(out,layer.out_proj(torch.zeros(1,7,6)))
        mask[0,0]=True
        assert torch.equal(GeometricReasoningOriginalImpl.forward(layer,s,None,mask,None,None),s+17)
        assert build._runtime_fingerprint(None)['query_chunking']['query_chunk_size']==3
    assert GeometricReasoningOriginalImpl.forward is original
