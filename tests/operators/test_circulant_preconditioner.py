"""Tests for circulant preconditioner operator."""

import pytest
import torch
from mrpro.algorithms.optimizers import cg
from mrpro.data import DcfData
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.operators import CirculantPreconditioner, NonUniformFastFourierOp
from mrpro.utils import RandomGenerator

from tests import dotproduct_adjointness_test
from tests.conftest import create_traj


def create_circulant_preconditioner_and_domain_range() -> tuple[
    CirculantPreconditioner,
    NonUniformFastFourierOp,
    DcfData,
    torch.Tensor,
    torch.Tensor,
]:
    """Create circulant preconditioner and random elements from domain and range."""
    rng = RandomGenerator(seed=0)

    img_shape = (1, 1, 1, 24, 24)
    nkx = (1, 1, 1, 12, 24)
    nky = (1, 1, 1, 12, 24)
    nkz = (1, 1, 1, 1, 1)
    traj = create_traj(nkx, nky, nkz, 'non-uniform', 'non-uniform', 'zero')

    recon_matrix = SpatialDimension(img_shape[-3], img_shape[-2], img_shape[-1])
    encoding_matrix = SpatialDimension(
        int(traj.kz.max() - traj.kz.min() + 1),
        int(traj.ky.max() - traj.ky.min() + 1),
        int(traj.kx.max() - traj.kx.min() + 1),
    )
    direction = [d for d, e in zip(('z', 'y', 'x'), encoding_matrix.zyx, strict=False) if e > 1]
    nufft_op = NonUniformFastFourierOp(
        direction=direction,  # type: ignore[arg-type]
        recon_matrix=recon_matrix,
        encoding_matrix=encoding_matrix,
        traj=traj,
    )

    dcf = DcfData.from_traj_voronoi(traj)
    circulant_preconditioner = CirculantPreconditioner(nufft_op, dcf)

    u = rng.complex64_tensor(size=img_shape)
    v = rng.complex64_tensor(size=img_shape)

    return circulant_preconditioner, nufft_op, dcf, u, v


def test_circulant_preconditioner_adjointness() -> None:
    """Test adjoint property of circulant preconditioner."""
    circulant_preconditioner, _, _, u, v = create_circulant_preconditioner_and_domain_range()
    dotproduct_adjointness_test(circulant_preconditioner, u, v)


def test_circulant_preconditioner_cg_iteration() -> None:
    """Test circulant preconditioner in CG iterations."""
    circulant_preconditioner, nufft_op, dcf, _, _ = create_circulant_preconditioner_and_domain_range()

    operator = nufft_op.H @ dcf.as_operator() @ nufft_op
    rng = RandomGenerator(seed=1)
    true_solution = rng.complex64_tensor(size=(1, 1, 1, 24, 24))
    (right_hand_side,) = operator(true_solution)

    initial_value = torch.zeros_like(true_solution)
    initial_residual = torch.linalg.vector_norm(right_hand_side.flatten())

    (solution_without_preconditioner,) = cg(
        operator,
        right_hand_side,
        initial_value=initial_value,
        max_iterations=3,
        tolerance=0,
    )
    residual_without_preconditioner = torch.linalg.vector_norm(
        (operator(solution_without_preconditioner)[0] - right_hand_side).flatten()
    )

    (solution_with_preconditioner,) = cg(
        operator,
        right_hand_side,
        initial_value=initial_value,
        preconditioner_inverse=circulant_preconditioner,
        max_iterations=3,
        tolerance=0,
    )
    residual_with_preconditioner = torch.linalg.vector_norm(
        (operator(solution_with_preconditioner)[0] - right_hand_side).flatten()
    )

    assert residual_without_preconditioner < initial_residual
    assert residual_with_preconditioner < initial_residual


@pytest.mark.cuda
def test_circulant_preconditioner_cuda() -> None:
    """Test circulant preconditioner works on CUDA devices."""
    rng = RandomGenerator(seed=2)
    x = rng.complex64_tensor(size=(1, 1, 1, 24, 24))

    # Create on CPU, transfer to GPU, run on GPU
    preconditioner, _, _, _, _ = create_circulant_preconditioner_and_domain_range()
    preconditioner.cuda()
    (result,) = preconditioner(x.cuda())
    assert result.is_cuda

    # Create on CPU, run on CPU
    preconditioner, _, _, _, _ = create_circulant_preconditioner_and_domain_range()
    (result,) = preconditioner(x)
    assert result.is_cpu

    # Create on GPU, run on GPU
    img_shape = (1, 1, 1, 24, 24)
    nkx = (1, 1, 1, 12, 24)
    nky = (1, 1, 1, 12, 24)
    nkz = (1, 1, 1, 1, 1)
    traj = create_traj(nkx, nky, nkz, 'non-uniform', 'non-uniform', 'zero').cuda()

    recon_matrix = SpatialDimension(img_shape[-3], img_shape[-2], img_shape[-1])
    encoding_matrix = SpatialDimension(
        int(traj.kz.max() - traj.kz.min() + 1),
        int(traj.ky.max() - traj.ky.min() + 1),
        int(traj.kx.max() - traj.kx.min() + 1),
    )
    direction = [d for d, e in zip(('z', 'y', 'x'), encoding_matrix.zyx, strict=False) if e > 1]
    nufft_op = NonUniformFastFourierOp(
        direction=direction,  # type: ignore[arg-type]
        recon_matrix=recon_matrix,
        encoding_matrix=encoding_matrix,
        traj=traj,
    )
    dcf = DcfData.from_traj_voronoi(traj)
    preconditioner = CirculantPreconditioner(nufft_op, dcf)
    (result,) = preconditioner(x.cuda())
    assert result.is_cuda

    # Create on GPU, transfer to CPU, run on CPU
    preconditioner.cpu()
    (result,) = preconditioner(x)
    assert result.is_cpu
