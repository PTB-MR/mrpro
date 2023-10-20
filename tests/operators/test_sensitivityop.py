import pytest
import torch

from mrpro.data import CsmData
from mrpro.data import QHeader
from mrpro.data import SpatialDimension
from mrpro.operators import SensitivityOp
from tests import RandomGenerator


def test_sensitivity_op_adjointness():
    """Sensitivity Operator adjoint property."""

    random_tensor = RandomGenerator(seed=0)
    Nz, Ny, Nx = 100, 100, 100
    num_coils = 4
    random_tensor = random_tensor.complex64_tensor(size=(1, num_coils, Nz, Ny, Nx))

    csm = CsmData(data=random_tensor, header=QHeader(fov=SpatialDimension(1.0, 1.0, 1.0)))

    # Generate sensitivity operator
    sensitivity_op = SensitivityOp(csm)

    u = RandomGenerator(seed=0).complex64_tensor(size=(1, 1, Nz, Ny, Nx))
    v = RandomGenerator(seed=0).complex64_tensor(size=(1, num_coils, Nz, Ny, Nx))

    # Apply forward operator
    forward = sensitivity_op.forward(u)

    # Apply adjoint operator
    adjoint = sensitivity_op.adjoint(v)

    # Check adjoint property
    diff = torch.sum(forward * torch.conj(v)) - torch.sum(u * torch.conj(adjoint))
    assert torch.sqrt(diff.real**2 + diff.imag**2) <= 0.01


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

    random_tensor = RandomGenerator(seed=0)
    Nz, Ny, Nx = 100, 100, 100
    num_coils = 4
    random_tensor = random_tensor.complex64_tensor(size=(csm_other_dim, num_coils, Nz, Ny, Nx))

    csm = CsmData(data=random_tensor, header=QHeader(fov=SpatialDimension(1.0, 1.0, 1.0)))

    # Generate sensitivity operator
    sensitivity_op = SensitivityOp(csm)

    u = RandomGenerator(seed=0).complex64_tensor(size=(img_other_dim, 1, Nz, Ny, Nx))
    v = RandomGenerator(seed=0).complex64_tensor(size=(img_other_dim, num_coils, Nz, Ny, Nx))

    forward = sensitivity_op.forward(u)
    assert forward is not None


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
    random_tensor = RandomGenerator(seed=0)
    Nz, Ny, Nx = 100, 100, 100
    num_coils = 4
    random_tensor = random_tensor.complex64_tensor(size=(csm_other_dim, num_coils, Nz, Ny, Nx))

    csm = CsmData(data=random_tensor, header=QHeader(fov=SpatialDimension(1.0, 1.0, 1.0)))

    # Generate sensitivity operator
    sensitivity_op = SensitivityOp(csm)

    u = RandomGenerator(seed=0).complex64_tensor(size=(img_other_dim, 1, Nz, Ny, Nx))
    v = RandomGenerator(seed=0).complex64_tensor(size=(img_other_dim, num_coils, Nz, Ny, Nx))

    with pytest.raises(RuntimeError, match='The size of tensor'):
        forward = sensitivity_op.forward(u)
