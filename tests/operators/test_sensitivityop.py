import pytest
import torch

from mrpro.data import CsmData
from mrpro.data import QHeader
from mrpro.data import SpatialDimension
from mrpro.operators import SensitivityOp
from tests import RandomGenerator


def test_sensitivity_op_adjointness():
    """Sensitivity Operator adjoint property."""

    random_generator = RandomGenerator(seed=0)
    Nz, Ny, Nx = 100, 100, 100
    num_coils = 4
    random_tensor = random_generator.complex64_tensor(size=(1, num_coils, Nz, Ny, Nx))

    random_csm = CsmData(data=random_tensor, header=QHeader(fov=SpatialDimension(1.0, 1.0, 1.0)))

    # Generate sensitivity operator
    sensitivity_op = SensitivityOp(random_csm)

    u = random_generator.complex64_tensor(size=(1, 1, Nz, Ny, Nx))
    v = random_generator.complex64_tensor(size=(1, num_coils, Nz, Ny, Nx))

    # Apply forward operator
    forward = sensitivity_op.forward(u)

    # Apply adjoint operator
    adjoint = sensitivity_op.adjoint(v)

    # Check adjoint property
    torch.testing.assert_close(
        torch.vdot(forward.ravel(), v.ravel()), torch.vdot(u.ravel(), adjoint.ravel()), rtol=1e-03, atol=1e-03
    )


@pytest.mark.parametrize(
    'csm_other_dim,img_other_dim',
    [
        (1, 1),
        (1, 6),
        (6, 6),
    ],
)
def test_sensitivity_op_other_dim_compatibility_pass(csm_other_dim, img_other_dim):
    """Test paired-dimensions that have to pass applying the sensitivity
    operator."""

    random_generator = RandomGenerator(seed=0)
    Nz, Ny, Nx = 100, 100, 100
    num_coils = 4
    random_tensor = random_generator.complex64_tensor(size=(csm_other_dim, num_coils, Nz, Ny, Nx))

    random_csm = CsmData(data=random_tensor, header=QHeader(fov=SpatialDimension(1.0, 1.0, 1.0)))

    # Generate sensitivity operator
    sensitivity_op = SensitivityOp(random_csm)

    u = random_generator.complex64_tensor(size=(img_other_dim, 1, Nz, Ny, Nx))
    v = random_generator.complex64_tensor(size=(img_other_dim, num_coils, Nz, Ny, Nx))

    forward = sensitivity_op.forward(u)
    adjoint = sensitivity_op.adjoint(v)

    assert forward is not None
    assert adjoint is not None

    assert forward.shape == (img_other_dim, num_coils, Nz, Ny, Nx)
    assert adjoint.shape == (img_other_dim, 1, Nz, Ny, Nx)


@pytest.mark.parametrize(
    'csm_other_dim,img_other_dim',
    [
        (6, 3),
        (3, 6),
    ],
)
def test_sensitivity_op_other_dim_compatibility_fail(csm_other_dim, img_other_dim):
    """Test paired-dimensions that have to raise error for the sensitivity
    operator."""
    random_generator = RandomGenerator(seed=0)
    Nz, Ny, Nx = 100, 100, 100
    num_coils = 4
    random_tensor = random_generator.complex64_tensor(size=(csm_other_dim, num_coils, Nz, Ny, Nx))

    random_csm = CsmData(data=random_tensor, header=QHeader(fov=SpatialDimension(1.0, 1.0, 1.0)))

    # Generate sensitivity operator
    sensitivity_op = SensitivityOp(random_csm)

    u = random_generator.complex64_tensor(size=(img_other_dim, 1, Nz, Ny, Nx))
    v = random_generator.complex64_tensor(size=(img_other_dim, num_coils, Nz, Ny, Nx))

    with pytest.raises(RuntimeError, match='The size of tensor'):
        forward = sensitivity_op.forward(u)

    with pytest.raises(RuntimeError, match='The size of tensor'):
        adjoint = sensitivity_op.adjoint(v)
