"""Tests for sensitivity operator."""

from collections.abc import Sequence

import pytest
import torch
from mrpro.data import CsmData, QHeader, SpatialDimension
from mrpro.operators import SensitivityOp

from tests import (
    RandomGenerator,
    dotproduct_adjointness_test,
    forward_mode_autodiff_of_linear_operator_test,
    gradient_of_linear_operator_test,
)


def create_sensitivity_op_and_domain_range() -> tuple[SensitivityOp, torch.Tensor, torch.Tensor]:
    """Create a sensitivity operator and an element from domain and range."""
    random_generator = RandomGenerator(seed=0)

    n_zyx = (2, 3, 4)
    n_other = (5, 6, 7)
    n_coils = 4
    # Generate sensitivity operator
    random_tensor = random_generator.complex64_tensor(size=(*n_other, n_coils, *n_zyx))
    random_csmdata = CsmData(data=random_tensor, header=QHeader(resolution=SpatialDimension(1.0, 1.0, 1.0)))
    sensitivity_op = SensitivityOp(random_csmdata)

    u = random_generator.complex64_tensor(size=(*n_other, 1, *n_zyx))
    v = random_generator.complex64_tensor(size=(*n_other, n_coils, *n_zyx))
    return sensitivity_op, u, v


def test_sensitivity_op_adjointness() -> None:
    """Test Sensitivity operator adjoint property."""
    dotproduct_adjointness_test(*create_sensitivity_op_and_domain_range())


def test_sensitivity_op_grad() -> None:
    """Test gradient of sensitivity operator."""
    gradient_of_linear_operator_test(*create_sensitivity_op_and_domain_range())


def test_sensitivity_op_forward_mode_autodiff() -> None:
    """Test forward-mode autodiff of sensitivity operator."""
    forward_mode_autodiff_of_linear_operator_test(*create_sensitivity_op_and_domain_range())


def test_sensitivity_op_csmdata_tensor() -> None:
    """Test matching result after creation via tensor and CSMData."""

    random_generator = RandomGenerator(seed=0)

    n_zyx = (2, 3, 4)
    n_other = (5, 6, 7)
    n_coils = 4

    # Generate sensitivity operators
    random_tensor = random_generator.complex64_tensor(size=(*n_other, n_coils, *n_zyx))
    random_csmdata = CsmData(data=random_tensor, header=QHeader(resolution=SpatialDimension(1.0, 1.0, 1.0)))
    sensitivity_op_csmdata = SensitivityOp(random_csmdata)
    sensitivity_op_tensor = SensitivityOp(random_tensor)

    # Check equality
    u = random_generator.complex64_tensor(size=(*n_other, 1, *n_zyx))
    v = random_generator.complex64_tensor(size=(*n_other, n_coils, *n_zyx))
    assert torch.equal(*sensitivity_op_csmdata(u), *sensitivity_op_tensor(u))
    assert torch.equal(*sensitivity_op_csmdata.H(v), *sensitivity_op_tensor.H(v))


@pytest.mark.parametrize(('n_other_csm', 'n_other_img'), [((2,), (2,)), ((1, 1), (2, 1)), ((3,), (1, 2, 3))])
def test_sensitivity_op_other_dim_compatibility_pass(n_other_csm: Sequence[int], n_other_img: Sequence[int]) -> None:
    """Test paired-dimensions that have to pass applying the sensitivity
    operator."""

    random_generator = RandomGenerator(seed=0)

    n_zyx = (2, 3, 4)
    n_coils = 4

    # Generate sensitivity operator
    random_tensor = random_generator.complex64_tensor(size=(*n_other_csm, n_coils, *n_zyx))
    random_csmdata = CsmData(data=random_tensor, header=QHeader(resolution=SpatialDimension(1.0, 1.0, 1.0)))
    sensitivity_op = SensitivityOp(random_csmdata)

    # Apply to n_other_img shape
    u = random_generator.complex64_tensor(size=(*n_other_img, 1, *n_zyx))
    v = random_generator.complex64_tensor(size=(*n_other_img, n_coils, *n_zyx))
    dotproduct_adjointness_test(sensitivity_op, u, v)


@pytest.mark.parametrize(('n_other_csm', 'n_other_img'), [(6, 3), (3, 6)])
def test_sensitivity_op_other_dim_compatibility_fail(n_other_csm: int, n_other_img: int) -> None:
    """Test paired-dimensions that have to raise error for the sensitivity
    operator."""

    random_generator = RandomGenerator(seed=0)

    n_zyx = (2, 3, 4)
    n_coils = 4

    # Generate sensitivity operator with n_other_csm shape
    random_tensor = random_generator.complex64_tensor(size=(n_other_csm, n_coils, *n_zyx))
    random_csmdata = CsmData(data=random_tensor, header=QHeader(resolution=SpatialDimension(1.0, 1.0, 1.0)))
    sensitivity_op = SensitivityOp(random_csmdata)

    # Apply to n_other_img shape
    u = random_generator.complex64_tensor(size=(n_other_img, 1, *n_zyx))
    with pytest.raises(RuntimeError, match='The size of tensor'):
        sensitivity_op(u)

    v = random_generator.complex64_tensor(size=(n_other_img, n_coils, *n_zyx))
    with pytest.raises(RuntimeError, match='The size of tensor'):
        sensitivity_op.adjoint(v)


@pytest.mark.cuda
def test_sensitivity_op_cuda() -> None:
    """Test sensitivity operator works on CUDA devices."""
    random_generator = RandomGenerator(seed=0)

    n_zyx = (2, 3, 4)
    n_other = (5, 6, 7)
    n_coils = 4
    # Generate input tensor and Csm data
    random_tensor = random_generator.complex64_tensor(size=(*n_other, n_coils, *n_zyx))
    u = random_generator.complex64_tensor(size=(*n_other, 1, *n_zyx))
    random_csmdata = CsmData(data=random_tensor, header=QHeader(resolution=SpatialDimension(1.0, 1.0, 1.0)))

    # Create on CPU, transfer to GPU, run on GPU
    sensitivity_op = SensitivityOp(random_csmdata)
    sensitivity_op.cuda()
    (result,) = sensitivity_op(u.cuda())
    assert result.is_cuda

    # Create on CPU, run on CPU
    sensitivity_op = SensitivityOp(random_csmdata)
    (result,) = sensitivity_op(u)
    assert result.is_cpu

    # Create on GPU, run on GPU
    sensitivity_op = SensitivityOp(random_csmdata.cuda())
    (result,) = sensitivity_op(u.cuda())
    assert result.is_cuda

    # Create on GPU, transfer to CPU, run on CPU
    sensitivity_op = SensitivityOp(random_csmdata.cuda())
    sensitivity_op.cpu()
    (result,) = sensitivity_op(u)
    assert result.is_cpu
