import pytest
import torch
from mrpro.nn import AxialRoPE
from mrpro.utils import RandomGenerator


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
def test_rope(device: torch.device):
    shape = (10, 10)
    n_heads = 2
    n_channels = 64
    n_embed = int(0.5 * n_channels // n_heads)
    q, k = RandomGenerator(seed=42).float32_tensor((2, 1, *shape, n_channels), low=0.5).to(device)
    rope = AxialRoPE(2, non_embed_fraction=0.5)
    (q_rope, k_rope) = rope(q, k)
    assert q_rope.shape == q.shape
    assert k_rope.shape == k.shape

    # non embedded channels should be the same
    torch.testing.assert_close(
        q.unflatten(-1, (n_heads, -1))[..., n_embed:], q_rope.unflatten(-1, (n_heads, -1))[..., n_embed:]
    )
    torch.testing.assert_close(
        k.unflatten(-1, (n_heads, -1))[..., n_embed:], k_rope.unflatten(-1, (n_heads, -1))[..., n_embed:]
    )

    # other should change
    q_emb = q_rope.unflatten(-1, (n_heads, -1))[..., :n_embed]
    q_orig = q.unflatten(-1, (n_heads, -1))[..., :n_embed]
    k_emb = k_rope.unflatten(-1, (n_heads, -1))[..., :n_embed]
    k_orig = k.unflatten(-1, (n_heads, -1))[..., :n_embed]
    assert not torch.isclose(q_emb, q_orig).all()
    assert not torch.isclose(k_emb, k_orig).all()
    assert not torch.isclose(q_emb, k_emb).all()
