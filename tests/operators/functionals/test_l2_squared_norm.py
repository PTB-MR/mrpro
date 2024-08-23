"""Tests for L2-Squared-functional."""

import pytest
import torch
from mrpro.operators.functionals.l2_squared import L2NormSquared


@pytest.mark.parametrize(
    (
        'x',
        'forward_x',
        'prox',
        'prox_complex_conjugate',
    ),
    [
        (
            torch.tensor([1, 1], dtype=torch.complex64),
            torch.tensor((2.0), dtype=torch.float32),
            torch.tensor([1 / 3, 1 / 3], dtype=torch.complex64),
            torch.tensor([2 / 3, 2 / 3], dtype=torch.complex64),
        ),
        (
            torch.tensor([1 + 1j, 1 + 1j], dtype=torch.complex64),
            torch.tensor((4.0), dtype=torch.float32),
            torch.tensor([(1 + 1j) / 3, (1 + 1j) / 3], dtype=torch.complex64),
            torch.tensor([2 * (1 + 1j) / 3, 2 * (1 + 1j) / 3], dtype=torch.complex64),
        ),
        (
            torch.tensor([1 + 0j, 1 + 1j], dtype=torch.complex64),
            torch.tensor((3.0), dtype=torch.float32),
            torch.tensor([1 / 3, (1 + 1j) / 3], dtype=torch.complex64),
            torch.tensor([2 / 3, 2 * (1 + 1j) / 3], dtype=torch.complex64),
        ),
    ],
)
def test_l2_squared_functional(
    x,
    forward_x,
    prox,
    prox_complex_conjugate,
):
    """Test if l2_squared_norm matches expected values."""
    l2_squared_norm = L2NormSquared(weight=1)

    torch.testing.assert_close(l2_squared_norm.forward(x)[0], forward_x, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(l2_squared_norm.prox(x, sigma=1)[0], prox, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        l2_squared_norm.prox_convex_conj(x, sigma=1)[0], prox_complex_conjugate, rtol=1e-3, atol=1e-3
    )
