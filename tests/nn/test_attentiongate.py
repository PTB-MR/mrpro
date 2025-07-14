"""Tests for AttentionGate module."""

import pytest
from mrpro.nn.AttentionGate import AttentionGate
from mrpro.utils import RandomGenerator


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    ('dim', 'channels_gate', 'channels_in', 'channels_hidden', 'input_shape', 'gate_shape'),
    [
        (2, 32, 32, 16, (1, 32, 32, 32), (1, 32, 16, 16)),
        (3, 32, 4, 8, (2, 4, 16, 16, 16), (2, 32, 16, 16, 16)),
    ],
)
def test_attention_gate(dim, channels_gate, channels_in, channels_hidden, input_shape, gate_shape, device):
    """Test AttentionGate output shape and backpropagation."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor(input_shape).to(device).requires_grad_(True)
    gate = rng.float32_tensor(gate_shape).to(device).requires_grad_(True)
    attn = AttentionGate(
        n_dim=dim, channels_gate=channels_gate, channels_in=channels_in, channels_hidden=channels_hidden
    ).to(device)
    output = attn(x, gate)
    assert output.shape == x.shape, f'Output shape {output.shape} != input shape {x.shape}'
    output.sum().backward()
    assert x.grad is not None, 'No gradient computed for input'
    assert gate.grad is not None, 'No gradient computed for gate'
    assert not x.isnan().any(), 'NaN values in input'
    assert not gate.isnan().any(), 'NaN values in gate'
    assert not x.grad.isnan().any(), 'NaN values in input gradients'
    assert not gate.grad.isnan().any(), 'NaN values in gate gradients'
    assert attn.project_gate.weight.grad is not None, 'No gradient computed for project_gate'
    assert attn.project_x.weight.grad is not None, 'No gradient computed for project_x'
    assert attn.psi[1].weight.grad is not None, 'No gradient computed for psi'
