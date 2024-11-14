"""Tests for sensitivity operator."""

import pytest
import torch
from mrpro.data import CsmData, QHeader, SpatialDimension
from mrpro.operators import SensitivityOp

from tests import RandomGenerator, dotproduct_adjointness_test


def test_sensitivity_op_adjointness():
    """Test Sensitivity operator adjoint property."""

    random_generator = RandomGenerator(seed=0)

    n_zyx = (2, 3, 4)
    n_other = (5, 6, 7)
    n_coils = 4

    # Generate sensitivity operator
    random_tensor = random_generator.complex64_tensor(size=(*n_other, n_coils, *n_zyx))
    random_csmdata = CsmData(data=random_tensor, header=QHeader(fov=SpatialDimension(1.0, 1.0, 1.0)))
    sensitivity_op = SensitivityOp(random_csmdata)

    # Check adjoint property
    u = random_generator.complex64_tensor(size=(*n_other, 1, *n_zyx))
    v = random_generator.complex64_tensor(size=(*n_other, n_coils, *n_zyx))
    dotproduct_adjointness_test(sensitivity_op, u, v)


def test_sensitivity_op_csmdata_tensor():
    """Test matching result after creation via tensor and CSMData."""

    random_generator = RandomGenerator(seed=0)

    n_zyx = (2, 3, 4)
    n_other = (5, 6, 7)
    n_coils = 4

    # Generate sensitivity operators
    random_tensor = random_generator.complex64_tensor(size=(*n_other, n_coils, *n_zyx))
    random_csmdata = CsmData(data=random_tensor, header=QHeader(fov=SpatialDimension(1.0, 1.0, 1.0)))
    sensitivity_op_csmdata = SensitivityOp(random_csmdata)
    sensitivity_op_tensor = SensitivityOp(random_tensor)

    # Check equality
    u = random_generator.complex64_tensor(size=(*n_other, 1, *n_zyx))
    v = random_generator.complex64_tensor(size=(*n_other, n_coils, *n_zyx))
    assert torch.equal(*sensitivity_op_csmdata(u), *sensitivity_op_tensor(u))
    assert torch.equal(*sensitivity_op_csmdata.H(v), *sensitivity_op_tensor.H(v))


@pytest.mark.parametrize(('n_other_csm', 'n_other_img'), [(1, 1), (1, 6), (6, 6)])
def test_sensitivity_op_other_dim_compatibility_pass(n_other_csm, n_other_img):
    """Test paired-dimensions that have to pass applying the sensitivity
    operator."""

    random_generator = RandomGenerator(seed=0)

    n_zyx = (2, 3, 4)
    n_coils = 4

    # Generate sensitivity operator
    random_tensor = random_generator.complex64_tensor(size=(n_other_csm, n_coils, *n_zyx))
    random_csmdata = CsmData(data=random_tensor, header=QHeader(fov=SpatialDimension(1.0, 1.0, 1.0)))
    sensitivity_op = SensitivityOp(random_csmdata)

    # Apply to n_other_img shape
    u = random_generator.complex64_tensor(size=(n_other_img, 1, *n_zyx))
    v = random_generator.complex64_tensor(size=(n_other_img, n_coils, *n_zyx))
    dotproduct_adjointness_test(sensitivity_op, u, v)


@pytest.mark.parametrize(('n_other_csm', 'n_other_img'), [(6, 3), (3, 6)])
def test_sensitivity_op_other_dim_compatibility_fail(n_other_csm, n_other_img):
    """Test paired-dimensions that have to raise error for the sensitivity
    operator."""

    random_generator = RandomGenerator(seed=0)

    n_zyx = (2, 3, 4)
    n_coils = 4

    # Generate sensitivity operator with n_other_csm shape
    random_tensor = random_generator.complex64_tensor(size=(n_other_csm, n_coils, *n_zyx))
    random_csmdata = CsmData(data=random_tensor, header=QHeader(fov=SpatialDimension(1.0, 1.0, 1.0)))
    sensitivity_op = SensitivityOp(random_csmdata)

    # Apply to n_other_img shape
    u = random_generator.complex64_tensor(size=(n_other_img, 1, *n_zyx))
    with pytest.raises(RuntimeError, match='The size of tensor'):
        sensitivity_op(u)

    v = random_generator.complex64_tensor(size=(n_other_img, n_coils, *n_zyx))
    with pytest.raises(RuntimeError, match='The size of tensor'):
        sensitivity_op.adjoint(v)
