"""Tests for B0-informed Fourier operators."""

import pytest
import torch
from mrpro.operators.B0InformedFourierOp import ConjugatePhaseFourierOp, MultiFrequencyFourierOp, TimeSegmentedFourierOp
from mrpro.operators.FastFourierOp import FastFourierOp
from mrpro.utils import RandomGenerator
from torch.autograd.gradcheck import gradcheck

from tests import (
    dotproduct_adjointness_test,
    forward_mode_autodiff_of_linear_operator_test,
    gradient_of_linear_operator_test,
)


def create_b0_informed_fourier_op_and_range_domain(
    operator_type: type[MultiFrequencyFourierOp] | type[TimeSegmentedFourierOp] | type[ConjugatePhaseFourierOp],
) -> tuple[MultiFrequencyFourierOp | TimeSegmentedFourierOp | ConjugatePhaseFourierOp, torch.Tensor, torch.Tensor]:
    """Create B0-informed Fourier operator and tensors from domain/range."""
    shape = (2, 3, 8, 10, 12)

    rng = RandomGenerator(seed=0)
    u = rng.complex64_tensor(size=shape)
    v = rng.complex64_tensor(size=shape)

    # Keep frequencies and readout times in a numerically stable range.
    b0_map = rng.float32_tensor(size=shape[-3:], low=-50, high=50)
    readout_times = torch.linspace(0, 5e-3, shape[-1], dtype=torch.float32)

    fourier_op = FastFourierOp(dim=(-3, -2, -1), recon_matrix=shape[-3:], encoding_matrix=shape[-3:])
    operator = operator_type(fourier_op=fourier_op, b0_map=b0_map, readout_times=readout_times)

    return operator, u, v


@pytest.mark.parametrize('operator_type', [MultiFrequencyFourierOp, TimeSegmentedFourierOp, ConjugatePhaseFourierOp])
def test_b0_informed_fourier_op_fwd_adj_property(
    operator_type: type[MultiFrequencyFourierOp] | type[TimeSegmentedFourierOp] | type[ConjugatePhaseFourierOp],
) -> None:
    """Test adjoint property of B0-informed Fourier operators."""
    dotproduct_adjointness_test(*create_b0_informed_fourier_op_and_range_domain(operator_type))


@pytest.mark.parametrize('operator_type', [MultiFrequencyFourierOp, TimeSegmentedFourierOp, ConjugatePhaseFourierOp])
def test_b0_informed_fourier_op_grad(
    operator_type: type[MultiFrequencyFourierOp] | type[TimeSegmentedFourierOp] | type[ConjugatePhaseFourierOp],
) -> None:
    """Test gradient of B0-informed Fourier operators."""
    gradient_of_linear_operator_test(*create_b0_informed_fourier_op_and_range_domain(operator_type))


@pytest.mark.parametrize('operator_type', [MultiFrequencyFourierOp, TimeSegmentedFourierOp, ConjugatePhaseFourierOp])
def test_b0_informed_fourier_op_forward_mode_autodiff(
    operator_type: type[MultiFrequencyFourierOp] | type[TimeSegmentedFourierOp] | type[ConjugatePhaseFourierOp],
) -> None:
    """Test forward-mode autodiff of B0-informed Fourier operators."""
    forward_mode_autodiff_of_linear_operator_test(*create_b0_informed_fourier_op_and_range_domain(operator_type))


def test_multi_frequency_fourier_op_invalid_bins() -> None:
    """Invalid number of bins raises a ValueError."""
    shape = (8, 10, 12)
    b0_map = torch.zeros(shape)
    readout_times = torch.linspace(0, 5e-3, shape[-1], dtype=torch.float32)
    fourier_op = FastFourierOp(dim=(-3, -2, -1), recon_matrix=shape, encoding_matrix=shape)

    with pytest.raises(ValueError, match='strictly positive'):
        MultiFrequencyFourierOp(fourier_op=fourier_op, b0_map=b0_map, readout_times=readout_times, n_bins=0)


def test_time_segmented_fourier_op_invalid_segments() -> None:
    """Invalid number of segments raises a ValueError."""
    shape = (8, 10, 12)
    b0_map = torch.zeros(shape)
    readout_times = torch.linspace(0, 5e-3, shape[-1], dtype=torch.float32)
    fourier_op = FastFourierOp(dim=(-3, -2, -1), recon_matrix=shape, encoding_matrix=shape)

    with pytest.raises(ValueError, match='strictly positive'):
        TimeSegmentedFourierOp(fourier_op=fourier_op, b0_map=b0_map, readout_times=readout_times, n_segments=0)


def test_b0_informed_fourier_op_variants_give_similar_kspace() -> None:
    """MFI, TS, and CP should produce similar k-space data for the same setup."""
    shape = (2, 3, 8, 10, 12)
    rng = RandomGenerator(seed=1)

    img = rng.complex64_tensor(size=shape)
    b0_map = rng.float32_tensor(size=shape[-3:], low=-40, high=40)
    readout_times = torch.linspace(0, 4e-3, shape[-1], dtype=torch.float32)
    fourier_op = FastFourierOp(dim=(-3, -2, -1), recon_matrix=shape[-3:], encoding_matrix=shape[-3:])

    mfi_op = MultiFrequencyFourierOp(fourier_op=fourier_op, b0_map=b0_map, readout_times=readout_times, n_bins=32)
    ts_op = TimeSegmentedFourierOp(
        fourier_op=fourier_op,
        b0_map=b0_map,
        readout_times=readout_times,
        n_segments=32,
        n_design_frequencies=64,
    )
    cp_op = ConjugatePhaseFourierOp(fourier_op=fourier_op, b0_map=b0_map, readout_times=readout_times)

    (k_mfi,) = mfi_op(img)
    (k_ts,) = ts_op(img)
    (k_cp,) = cp_op(img)

    torch.testing.assert_close(k_mfi, k_cp, rtol=8e-2, atol=2e-2)
    torch.testing.assert_close(k_ts, k_cp, rtol=8e-2, atol=2e-2)
    torch.testing.assert_close(k_mfi, k_ts, rtol=8e-2, atol=2e-2)


@pytest.mark.parametrize('operator_type', [TimeSegmentedFourierOp, ConjugatePhaseFourierOp, MultiFrequencyFourierOp])
def test_b0_informed_fourier_op_gradcheck_wrt_b0_map(
    operator_type: type[TimeSegmentedFourierOp] | type[ConjugatePhaseFourierOp] | type[MultiFrequencyFourierOp],
) -> None:
    """Gradient check for forward k-space wrt b0 map."""
    shape = (1, 1, 1, 16, 24)
    rng = RandomGenerator(seed=7)
    img = rng.complex64_tensor(size=shape)
    readout_times = torch.linspace(0, 3e-3, shape[-1], dtype=torch.float32)
    fourier_op = FastFourierOp(dim=(-2, -1), recon_matrix=shape[-2:], encoding_matrix=shape[-2:])

    def forward_from_b0_map(b0_map: torch.Tensor) -> torch.Tensor:
        operator = operator_type(fourier_op=fourier_op, b0_map=b0_map, readout_times=readout_times)
        (kspace,) = operator(img)
        return kspace

    b0_map = rng.float64_tensor(size=shape[-3:], low=-20, high=20).requires_grad_(True)
    gradcheck(forward_from_b0_map, (b0_map,), fast_mode=True, eps=1e-3, atol=5e-2, rtol=5e-2)
