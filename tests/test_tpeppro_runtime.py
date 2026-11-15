import pytest
import numpy as np
import torch
from pathlib import Path
from src.external.tpeppro_runtime import TAGConv, TPepProRuntime


def test_tagconv_matches_explicit_dgl_message_operations():
    torch.manual_seed(11)
    layer=TAGConv()
    x=torch.randn(4,1024)
    adjacency=torch.tensor([[1,1,0,0],[0,1,1,0],[1,0,1,0],[0,0,0,0]])
    norm=adjacency.sum(0).clamp(min=1).float().pow(-.5)[:,None]
    stack=[x]
    for _ in range(2):
        messages=stack[-1]*norm
        received=torch.zeros_like(x)
        for source,destination in adjacency.nonzero().tolist(): received[destination]+=messages[source]
        stack.append(received*norm)
    expected=layer.lin(torch.cat(stack,dim=1))
    torch.testing.assert_close(layer(x,adjacency),expected,rtol=1e-5,atol=1e-6)


def test_real_checkpoint_and_bundled_pair():
    root=Path(__file__).parents[1]/'otherModels/TPepPro/test_sample_model'
    required = ['save_model_pkl/TPepPro_receptor-peptide(train 19187 pairs)_GAT.pkl', 'data/sample_embeddings.npz', 'data/sample_cmap/1a1m_A.npz', 'data/sample_cmap/1a1m_C.npz']
    if not all((root / name).is_file() for name in required):
        pytest.skip('Optional upstream TPepPro integration fixtures are not distributed')
    model=TPepProRuntime(root/'save_model_pkl/TPepPro_receptor-peptide(train 19187 pairs)_GAT.pkl')
    embeddings=np.load(root/'data/sample_embeddings.npz')
    inputs=[]
    for name in ['1a1m_A','1a1m_C']:
        c=np.load(root/'data/sample_cmap'/f'{name}.npz')
        inputs.extend([embeddings[name][:len(str(c['seq']))],c['contact']])
    result=model(*inputs)
    assert np.isfinite(result) and 0 <= result <= 1
