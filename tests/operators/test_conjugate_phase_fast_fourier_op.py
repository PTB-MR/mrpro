import pytest
import torch
from mrpro.operators.ConjugatePhaseFastFourierOp import ConjugatePhaseFastFourierOp
from mrpro.utils import RandomGenerator

from tests.helper import dotproduct_adjointness_test


@pytest.mark.parametrize('shape', [(1, 1, 1, 4, 4), (2, 1, 1, 8, 8)])
def test_conjugate_phase_fast_fourier_op_forward(shape):
    """Check that the operator works with different shapes."""
    rng = RandomGenerator(seed=2)

    # Creating a field map with a limited number of unique frequencies
    # to keep the number of segments reasonable.
    b0_map = torch.randint(0, 5, shape) * 10.0

    # Generate readout vector
    t_ro = torch.arange(shape[-1]) / 20e3
    op = ConjugatePhaseFastFourierOp(b0_map=b0_map, readout_times=t_ro)

    u = rng.complex64_tensor(shape)
    (v,) = op(u)
    assert v.shape == u.shape
    assert not torch.isnan(v).any()


@pytest.mark.parametrize('shape', [(2, 1, 1, 8, 8), (1, 1, 2, 8, 8)])
def test_conjugate_phase_fast_fourier_op_adjointness(shape):
    """Test adjointness of ConjugatePhaseFastFourierOp."""
    rng = RandomGenerator(seed=0)

    # b0_map can be 3D or matching spatial dims of data
    # The current implementation expects b0_map to be (Z, Y, X) or similar
    # In CPR, it's usually (Z, Y, X)
    spatial_shape = shape[-3:]
    b0_map = torch.randint(0, 3, spatial_shape) * 20.0

    t_ro = torch.arange(shape[-1]) / 20e3
    op = ConjugatePhaseFastFourierOp(b0_map=b0_map, readout_times=t_ro)

    u = rng.complex64_tensor(shape)
    v = rng.complex64_tensor(shape)

    dotproduct_adjointness_test(op, u, v)
