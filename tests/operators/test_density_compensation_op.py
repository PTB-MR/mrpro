"""Tests for density compensation operator."""

import torch
from mrpro.data import DcfData
from mrpro.operators import DensityCompensationOp

from tests import RandomGenerator, dotproduct_adjointness_test


def test_density_compensation_op_adjointness():
    """Test density operator adjoint property."""
    random_generator = RandomGenerator(seed=0)

    n_zyx = (2, 3, 4)
    n_other = (5, 6, 7)
    n_coils = 8

    # Generate random dcf and operator
    random_tensor = random_generator.complex64_tensor(size=(*n_other, *n_zyx))
    random_dcf = DcfData(data=random_tensor)
    dcf_op = DensityCompensationOp(random_dcf)

    # Check adjoint property
    u = random_generator.complex64_tensor(size=(*n_other, n_coils, *n_zyx))
    v = random_generator.complex64_tensor(size=(*n_other, n_coils, *n_zyx))
    dotproduct_adjointness_test(dcf_op, u, v)


def test_density_compensation_op_dcfdata_tensor():
    """Test matching result after creation via tensor and DcfData."""
    random_generator = RandomGenerator(seed=0)

    n_zyx = (2, 3, 4)
    n_other = (5, 6, 7)
    n_coils = 8

    # Generate random dcf
    random_tensor = random_generator.complex64_tensor(size=(*n_other, *n_zyx))
    random_dcf = DcfData(data=random_tensor)

    # and operators
    dcf_op_tensor = DensityCompensationOp(random_tensor)
    dcf_op_dcfdata = DensityCompensationOp(random_dcf)
    dcf_op_dcfdata_asop = random_dcf.as_operator()

    # Check equality
    u = random_generator.complex64_tensor(size=(*n_other, n_coils, *n_zyx))
    v = random_generator.complex64_tensor(size=(*n_other, n_coils, *n_zyx))
    assert torch.equal(*dcf_op_tensor(u), *dcf_op_dcfdata(u))
    assert torch.equal(*dcf_op_tensor.H(v), *dcf_op_dcfdata.H(v))
    assert torch.equal(*dcf_op_tensor(u), *dcf_op_dcfdata_asop(u))
    assert torch.equal(*dcf_op_tensor.H(v), *dcf_op_dcfdata_asop.H(v))


def test_density_compensation_op_forward():
    """Test result of forward."""
    random_generator = RandomGenerator(seed=0)
    n_zyx = (2, 3, 4)
    n_other = (5, 6, 7)
    n_coils = 8
    random_tensor = random_generator.complex64_tensor(size=(*n_other, *n_zyx))
    dcf_op = DensityCompensationOp(random_tensor)
    u = random_generator.complex64_tensor(size=(*n_other, n_coils, *n_zyx))
    # forward should be a multiplication with the dcf
    expected = random_tensor.unsqueeze(-4) * u
    (actual,) = dcf_op(u)
    torch.testing.assert_close(actual, expected)
