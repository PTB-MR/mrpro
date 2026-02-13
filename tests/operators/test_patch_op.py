"""Tests for Rearrange Operator."""

from collections.abc import Sequence

import pytest
import torch
from mr2.operators.PatchOp import PatchOp
from mr2.utils import RandomGenerator

from tests import autodiff_test, dotproduct_adjointness_test

TESTCASES = pytest.mark.parametrize(
    ('input_shape', 'arguments', 'output_shape'),
    [
        ((3, 4, 5), {'dim': (0, 1), 'patch_size': (1, 3), 'stride': (3, 1), 'dilation': (2, 1)}, (2, 1, 3, 5)),
        ((1, 20), {'dim': -1, 'patch_size': 3, 'stride': 3, 'dilation': 5}, (4, 1, 3)),
    ],
)


@TESTCASES
def test_patch_op_adjointness(
    input_shape: Sequence[int], arguments: dict[str, int | Sequence[int]], output_shape: Sequence[int]
) -> None:
    """Test adjointness and shape of Rearrange Op."""
    rng = RandomGenerator(seed=0)
    u = rng.complex64_tensor(size=input_shape)
    v = rng.complex64_tensor(size=output_shape)
    domain_size = (
        input_shape[arguments['dim']]
        if isinstance(arguments['dim'], int)
        else [input_shape[ax] for ax in arguments['dim']]
    )
    operator = PatchOp(**arguments, domain_size=domain_size)
    dotproduct_adjointness_test(operator, u, v)


@TESTCASES
def test_patch_op_autodiff(
    input_shape: Sequence[int], arguments: dict[str, int | Sequence[int]], output_shape: Sequence[int]
) -> None:
    """Test autodiff works for PatchOp."""
    rng = RandomGenerator(seed=0)
    u = rng.complex64_tensor(size=input_shape)
    domain_size = (
        input_shape[arguments['dim']]
        if isinstance(arguments['dim'], int)
        else [input_shape[ax] for ax in arguments['dim']]
    )
    operator = PatchOp(**arguments, domain_size=domain_size)
    autodiff_test(operator, u)


def test_patch_op_invalid() -> None:
    """Test invalid parameters for PatchOp."""
    with pytest.raises(ValueError, match='must be unique'):
        PatchOp(dim=(0, 0), patch_size=3)

    with pytest.raises(ValueError, match='patch_size must be positive'):
        PatchOp(dim=0, patch_size=-3)

    with pytest.raises(ValueError, match='patch_size must be positive'):
        PatchOp(dim=(0, 1), patch_size=(3, -3))

    with pytest.raises(ValueError, match='stride must be positive'):
        PatchOp(dim=0, patch_size=3, stride=-3)

    with pytest.raises(ValueError, match='stride must be positive'):
        PatchOp(dim=(0, 1), patch_size=3, stride=(3, -3))

    with pytest.raises(ValueError, match='dilation must be positive'):
        PatchOp(dim=0, patch_size=3, dilation=-3)

    with pytest.raises(ValueError, match='dilation must be positive'):
        PatchOp(dim=(0, 1), patch_size=3, dilation=(3, -3))

    with pytest.raises(ValueError, match='domain_size must be positive'):
        PatchOp(dim=0, patch_size=3, domain_size=-1)

    with pytest.raises(ValueError, match='domain_size must be positive'):
        PatchOp(dim=(0, 1), patch_size=3, domain_size=(3, -1))

    with pytest.raises(ValueError, match='Length mismatch'):
        PatchOp(dim=0, patch_size=3, dilation=(3, 1))

    with pytest.raises(ValueError, match='Length mismatch'):
        PatchOp(dim=0, patch_size=(3, 3), stride=3)

    with pytest.raises(ValueError, match='Length mismatch'):
        PatchOp(dim=0, patch_size=(3, 3), dilation=3)

    with pytest.raises(ValueError, match='Length mismatch'):
        PatchOp(dim=(0, 1), patch_size=(1, 2, 3))

    with pytest.raises(ValueError, match='Length mismatch'):
        PatchOp(dim=(1, 2), patch_size=1, domain_size=(1,))

    operator = PatchOp(dim=(0, -1), patch_size=3)
    with pytest.raises(IndexError, match='unique'):
        operator(torch.ones(10))

    operator = PatchOp(dim=0, patch_size=3)
    with pytest.raises(ValueError, match='too small'):
        operator(torch.ones(2))


@pytest.mark.cuda
def test_patch_op_cuda() -> None:
    """Test the cMRF model works on cuda devices."""

    rng = RandomGenerator(8)
    u = rng.complex64_tensor((10, 3, 4))

    # Create on CPU, transfer to GPU and run on GPU
    model = PatchOp(0, 2, 3, 2)
    model.cuda()
    (signal,) = (model.H @ model)(u.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Operator has no tensor parameters, so no need to test creating on GPU
