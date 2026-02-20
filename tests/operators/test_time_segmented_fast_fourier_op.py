import pytest
import torch
from mrpro.operators.TimeSegmentedFastFourierOp import TimeSegmentedFastFourierOp
from mrpro.utils import RandomGenerator

from tests.helper import dotproduct_adjointness_test


@pytest.mark.parametrize('n_segments', [5, 10])
def test_time_segmented_fast_fourier_op_forward(n_segments):
    """Check that the operator works with different numbers of segments."""
    rng = RandomGenerator(seed=2)

    shape = (1, 1, 1, 4, 4)
    b0_map = rng.float32_tensor(shape)
    # Generate readout vector with 20 kHz ADC bandwidth
    t_ro = torch.arange(shape[-1]) / 20e3
    op = TimeSegmentedFastFourierOp(b0_map=b0_map, readout_times=t_ro, num_segments=n_segments)

    u = rng.complex64_tensor(shape)
    (v,) = op(u)
    assert v.shape == u.shape
    assert not torch.isnan(v).any()


@pytest.mark.parametrize('num_frequencies', [5, 10, -1])
def test_time_segmented_fast_fourier_op_num_frequencies(num_frequencies):
    """Check that the operator works with different numbers of frequencies."""
    rng = RandomGenerator(seed=2)

    shape = (1, 1, 1, 4, 4)
    b0_map = rng.float32_tensor(shape)
    t_ro = torch.arange(shape[-1]) / 20e3
    op = TimeSegmentedFastFourierOp(b0_map=b0_map, readout_times=t_ro, num_segments=5, num_frequencies=num_frequencies)

    u = rng.complex64_tensor(shape)
    (v,) = op(u)
    assert v.shape == u.shape
    assert not torch.isnan(v).any()


@pytest.mark.parametrize('shape', [(1, 1, 2, 16, 16), (2, 1, 1, 24, 16)])
def test_time_segmented_fast_fourier_op_adjointness(shape):
    """Test adjointness of TimeSegmentedFastFourierOp."""
    rng = RandomGenerator(seed=0)

    # Generate readout vector with 20 kHz ADC bandwidth
    t_ro = torch.arange(shape[-1]) / 20e3
    # Generate field map with random values
    b0_map = rng.float32_tensor(shape)

    op = TimeSegmentedFastFourierOp(b0_map=b0_map, readout_times=t_ro, num_segments=5)

    u = rng.complex64_tensor(shape)
    v = rng.complex64_tensor(shape)

    dotproduct_adjointness_test(op, u, v)
