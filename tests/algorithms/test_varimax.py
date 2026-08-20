"""Tests for Varimax rotation."""

import pytest
import torch
from mrpro.algorithms.varimax import varimax
from mrpro.utils import RandomGenerator


@pytest.mark.parametrize('shape', [(3, 8), (2, 3, 8)])
def test_varimax_shape_dtype(shape: tuple[int, ...]) -> None:
    """Varimax keeps the input shape, dtype, and device."""
    phi = RandomGenerator(seed=0).complex64_tensor(shape)

    rotated = varimax(phi, n_iterations=3)

    assert rotated.shape == phi.shape
    assert rotated.dtype == phi.dtype


def test_varimax_preserves_orthogonality_subspace_and_variance() -> None:
    """Varimax is a rotation and preserves the compression subspace."""
    data = RandomGenerator(seed=0).complex64_tensor((12, 4))
    q, _ = torch.linalg.qr(data, mode='reduced')
    phi = q.mH

    rotated = varimax(phi)

    torch.testing.assert_close(rotated @ rotated.mH, phi @ phi.mH)
    torch.testing.assert_close(rotated.mH @ rotated, phi.mH @ phi)
    torch.testing.assert_close(rotated.abs().square().sum(), phi.abs().square().sum())


def test_varimax_improves_objective_function() -> None:
    """Varimax increases the variance of squared component loadings."""
    data = RandomGenerator(seed=0).complex64_tensor((12, 4))
    q, _ = torch.linalg.qr(data, mode='reduced')
    phi = q.mH

    rotated = varimax(phi)
    objective = phi.abs().square().var(dim=-1).sum()
    rotated_objective = rotated.abs().square().var(dim=-1).sum()

    assert rotated_objective > objective


def test_varimax_batched_matches_individual_rotations() -> None:
    """Batched Varimax gives the same result as rotating each matrix separately."""
    data = RandomGenerator(seed=0).complex64_tensor((3, 12, 4))
    q, _ = torch.linalg.qr(data, mode='reduced')
    phi = q.mH

    batched = varimax(phi)
    individual = torch.stack([varimax(single_phi) for single_phi in phi])

    torch.testing.assert_close(batched, individual)


def test_varimax_zero_iterations_returns_input() -> None:
    """Zero iterations should leave the input unchanged."""
    phi = RandomGenerator(seed=0).complex64_tensor((2, 3, 8))

    rotated = varimax(phi, n_iterations=0)

    torch.testing.assert_close(rotated, phi)
