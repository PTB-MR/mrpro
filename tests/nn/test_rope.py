"""Tests for AxialRoPE module."""

import pytest
import torch
from mr2.nn import AxialRoPE
from mr2.utils import RandomGenerator


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
def test_rope(device: torch.device) -> None:
    """Test AxialRoPE rotation and embedding functionality."""
    shape = (10, 10)
    n_heads = 2
    n_channels = 64
    n_embed = int(0.5 * n_channels)
    q, k = RandomGenerator(seed=42).float32_tensor((2, 1, n_heads, *shape, n_channels), low=0.5).to(device)

    rope = AxialRoPE(embed_fraction=0.5)
    (q_rope, k_rope) = rope(q, k)

    assert q_rope.shape == q.shape
    assert k_rope.shape == k.shape

    # non embedded channels should be the same
    torch.testing.assert_close(q[..., n_embed:], q_rope[..., n_embed:])
    torch.testing.assert_close(k[..., n_embed:], k_rope[..., n_embed:])

    # other should change
    assert not torch.isclose(q_rope[..., :n_embed], q[..., :n_embed]).all()
    assert not torch.isclose(k_rope[..., :n_embed], k[..., :n_embed]).all()
