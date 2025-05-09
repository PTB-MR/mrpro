"""Tests for the Cartesian sampling operator."""

from typing import TypeAlias

import pytest
import torch
from einops import rearrange
from mrpro.data import KTrajectory, SpatialDimension
from mrpro.operators import CartesianMaskingOp, CartesianSamplingOp
from mrpro.utils import RandomGenerator
from typing_extensions import Unpack

from tests import (
    dotproduct_adjointness_test,
    forward_mode_autodiff_of_linear_operator_test,
    gradient_of_linear_operator_test,
)
from tests.conftest import create_traj

AtLeast3Ints: TypeAlias = tuple[Unpack[tuple[int, ...]], int, int, int]


def test_cart_sampling_op_data_match() -> None:
    # Create 3D uniform trajectory
    k_shape = (1, 5, 20, 40, 60)
    nkx = (1, 1, 1, 1, 60)
    nky = (1, 1, 1, 40, 1)
    nkz = (1, 1, 20, 1, 1)
    type_kx = 'uniform'
    type_ky = 'uniform'
    type_kz = 'uniform'
    trajectory = create_traj(nkx, nky, nkz, type_kx, type_ky, type_kz)

    # Create matching data
    rng = RandomGenerator(seed=0)
    kdata = rng.complex64_tensor(size=k_shape)

    # Create sampling operator
    encoding_matrix = SpatialDimension(k_shape[-3], k_shape[-2], k_shape[-1])
    sampling_op = CartesianSamplingOp(encoding_matrix=encoding_matrix, traj=trajectory)

    # Subsample data and trajectory
    kdata_sub = kdata[..., ::2, ::4, ::3]
    trajectory_sub = KTrajectory(
        kz=trajectory.kz[..., ::2, :, :],
        ky=trajectory.ky[..., :, ::4, :],
        kx=trajectory.kx[..., :, :, ::3],
    )
    sampling_op_sub = CartesianSamplingOp(encoding_matrix=encoding_matrix, traj=trajectory_sub)

    # Verify that the fully-sampled sampling operator does not do anything because the data is already sorted
    assert not sampling_op._needs_indexing

    # Verify identical shape
    (k,) = sampling_op.adjoint(kdata)
    (k_sub,) = sampling_op_sub.adjoint(kdata_sub)
    assert k.shape == k_sub.shape

    # Verify data is correctly sorted
    torch.testing.assert_close(kdata[..., ::2, ::4, ::3], k_sub[..., ::2, ::4, ::3])


def subsample_traj(
    trajectory: KTrajectory, sampling: str, k_shape: tuple[int, int, int, Unpack[tuple[int, ...]]]
) -> KTrajectory:
    """Subsample trajectory based on sampling type."""
    trajectory_tensor = trajectory.as_tensor()
    # Subsample data and trajectory
    match sampling:
        case 'random':
            random_idx = RandomGenerator(13).randperm(k_shape[-2])
            trajectory = KTrajectory.from_tensor(trajectory_tensor[..., random_idx, :])
        case 'partial_echo':
            trajectory = KTrajectory.from_tensor(trajectory_tensor[..., : k_shape[-1] // 2])
        case 'partial_fourier':
            trajectory = KTrajectory.from_tensor(trajectory_tensor[..., : k_shape[-3] // 2, : k_shape[-2] // 2, :])
        case 'regular_undersampling':
            trajectory = KTrajectory.from_tensor(trajectory_tensor[..., ::3, ::5, :])
        case 'random_undersampling':
            random_idx = RandomGenerator(13).randperm(k_shape[-2])
            trajectory = KTrajectory.from_tensor(trajectory_tensor[..., random_idx[: k_shape[-2] // 2], :])
        case 'different_random_undersampling':
            traj_list = [
                traj_one_other[..., RandomGenerator(13).randperm(k_shape[-2])[: k_shape[-2] // 2], :]
                for traj_one_other in trajectory_tensor.unbind(1)
            ]
            trajectory = KTrajectory.from_tensor(torch.stack(traj_list, dim=1))
        case 'cartesian_and_non_cartesian':
            trajectory = KTrajectory.from_tensor(trajectory_tensor)
        case 'kx_ky_along_k0':
            trajectory_tensor = rearrange(trajectory_tensor, '... k1 k0->... 1 (k1 k0)')
            trajectory = KTrajectory.from_tensor(trajectory_tensor)
        case 'kx_ky_along_k0_undersampling':
            trajectory_tensor = rearrange(trajectory_tensor, '... k1 k0->... 1 (k1 k0)')
            random_idx = RandomGenerator(13).randperm(trajectory_tensor.shape[-1])
            trajectory = KTrajectory.from_tensor(trajectory_tensor[..., random_idx[: trajectory_tensor.shape[-1] // 2]])
        case _:
            raise NotImplementedError(f'Test {sampling} not implemented.')
    return trajectory


def create_cart_sampling_op_and_range_domain(
    sampling: str,
    k_shape: AtLeast3Ints = (2, 5, 20, 40, 60),
    nkx: AtLeast3Ints = (2, 1, 1, 1, 60),
    nky: AtLeast3Ints = (2, 1, 1, 40, 1),
    nkz: AtLeast3Ints = (2, 1, 20, 1, 1),
) -> tuple[CartesianSamplingOp, torch.Tensor, torch.Tensor]:
    type_kx = 'uniform'
    type_ky = 'non-uniform' if sampling == 'cartesian_and_non_cartesian' else 'uniform'
    type_kz = 'non-uniform' if sampling == 'cartesian_and_non_cartesian' else 'uniform'
    trajectory = create_traj(nkx, nky, nkz, type_kx, type_ky, type_kz)
    trajectory = subsample_traj(trajectory, sampling, k_shape)

    encoding_matrix = SpatialDimension(k_shape[-3], k_shape[-2], k_shape[-1])
    sampling_op = CartesianSamplingOp(encoding_matrix=encoding_matrix, traj=trajectory)
    rng = RandomGenerator(seed=0)
    u = rng.complex64_tensor(size=k_shape)
    v = rng.complex64_tensor(size=(*k_shape[:-3], *trajectory.shape[-3:]))
    return sampling_op, u, v


SAMPLING_PARAMETERS = pytest.mark.parametrize(
    'sampling',
    [
        'random',
        'partial_echo',
        'partial_fourier',
        'regular_undersampling',
        'random_undersampling',
        'different_random_undersampling',
        'cartesian_and_non_cartesian',
        'kx_ky_along_k0',
        'kx_ky_along_k0_undersampling',
    ],
)


@SAMPLING_PARAMETERS
def test_cart_sampling_op_fwd_adj(sampling: str) -> None:
    """Test adjoint property of the Cartesian sampling operator."""
    dotproduct_adjointness_test(*create_cart_sampling_op_and_range_domain(sampling))


@SAMPLING_PARAMETERS
def test_cart_sampling_op_grad(sampling: str) -> None:
    """Test the gradient of the Cartesian sampling operator."""
    gradient_of_linear_operator_test(*create_cart_sampling_op_and_range_domain(sampling))


@SAMPLING_PARAMETERS
def test_cart_sampling_op_forward_mode_autodiff(sampling: str) -> None:
    """Test forward-mode autodiff of the Cartesian sampling operator."""
    forward_mode_autodiff_of_linear_operator_test(*create_cart_sampling_op_and_range_domain(sampling))


@SAMPLING_PARAMETERS
def test_cart_sampling_op_gram(sampling: str) -> None:
    """Test adjoint gram of Cartesian sampling operator."""
    sampling_op, u, _ = create_cart_sampling_op_and_range_domain(sampling)
    (expected,) = (sampling_op.H @ sampling_op)(u)
    (actual,) = sampling_op.gram(u)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(('k2_min', 'k2_max'), [(-1, 21), (-21, 1)])
@pytest.mark.parametrize(('k0_min', 'k0_max'), [(-6, 13), (-13, 6)])
def test_cart_sampling_op_oversampling(k0_min: int, k0_max: int, k2_min: int, k2_max: int) -> None:
    """Test trajectory points outside of encoding_matrix."""
    encoding_matrix = SpatialDimension(40, 1, 20)

    # Create kx and kz sampling which are asymmetric and larger than the encoding matrix on one side
    # The indices are inverted to ensure CartesianSamplingOp acts on them
    kx = rearrange(torch.linspace(k0_max, k0_min, 20), 'kx -> 1 1 1 1 kx')
    ky = torch.ones(1, 1, 1, 1, 1)
    kz = rearrange(torch.linspace(k2_max, k2_min, 40), 'kz -> 1 1 kz 1 1')
    kz = torch.concat([kz, -kz], dim=0)  # different kz values for two other elements
    trajectory = KTrajectory(kz=kz, ky=ky, kx=kx)

    with pytest.warns(UserWarning, match='K-space points lie outside of the encoding_matrix'):
        sampling_op = CartesianSamplingOp(encoding_matrix=encoding_matrix, traj=trajectory)

    rng = RandomGenerator(seed=0)
    u = rng.complex64_tensor(size=(3, 2, 5, kz.shape[-3], ky.shape[-2], kx.shape[-1]))
    v = rng.complex64_tensor(size=(3, 2, 5, *encoding_matrix.zyx))

    assert sampling_op.adjoint(u)[0].shape[-3:] == encoding_matrix.zyx
    assert sampling_op(v)[0].shape[-3:] == (kz.shape[-3], ky.shape[-2], kx.shape[-1])


def test_cart_sampling_op_repr():
    """Test the __repr__ method of Cartesian sampling operator."""

    # Create 3D uniform trajectory
    k_shape = (1, 5, 20, 40, 60)
    nkx = (1, 1, 1, 1, 60)
    nky = (1, 1, 1, 40, 1)
    nkz = (1, 1, 20, 1, 1)
    type_kx = 'uniform'
    type_ky = 'uniform'
    type_kz = 'uniform'
    trajectory = create_traj(nkx, nky, nkz, type_kx, type_ky, type_kz)

    encoding_matrix = SpatialDimension(k_shape[-3], k_shape[-2], k_shape[-1])
    sampling_op = CartesianSamplingOp(encoding_matrix=encoding_matrix, traj=trajectory)
    repr_str = repr(sampling_op)

    # Check if the _repr__ string contains expected information
    assert 'CartesianSamplingOp' in repr_str
    assert 'Needs indexing' in repr_str
    assert 'Sorted grid shape' in repr_str
    assert 'device' in repr_str


@pytest.mark.cuda
def test_cart_sampling_op_cuda() -> None:
    """Move trajectory to CUDA memory."""

    # Create 3D uniform trajectory
    k_shape = (1, 5, 20, 40, 60)
    nkx = (1, 1, 1, 1, 60)
    nky = (1, 1, 1, 40, 1)
    nkz = (1, 1, 20, 1, 1)
    type_kx = 'uniform'
    type_ky = 'uniform'
    type_kz = 'uniform'
    trajectory_tensor = create_traj(nkx, nky, nkz, type_kx, type_ky, type_kz).as_tensor()

    traj_list = [
        traj_one_other[..., torch.randperm(k_shape[-2])[: k_shape[-2] // 2], :]
        for traj_one_other in trajectory_tensor.unbind(1)
    ]
    trajectory = KTrajectory.from_tensor(torch.stack(traj_list, dim=1))

    encoding_matrix = SpatialDimension(k_shape[-3], k_shape[-2], k_shape[-1])
    rng = RandomGenerator(seed=0)
    input_data = rng.complex64_tensor(size=k_shape)

    # Create on CPU, transfer to GPU and run on GPU
    sampling_op = CartesianSamplingOp(encoding_matrix=encoding_matrix, traj=trajectory)
    operator = sampling_op.H @ sampling_op
    operator.cuda()
    (result,) = operator(input_data.cuda())
    assert result.is_cuda

    # Create on CPU and run on CPU
    sampling_op = CartesianSamplingOp(encoding_matrix=encoding_matrix, traj=trajectory)
    operator = sampling_op.H @ sampling_op
    (result,) = operator(input_data)
    assert result.is_cpu

    # Create on GPU and run on GPU
    sampling_op = CartesianSamplingOp(encoding_matrix=encoding_matrix, traj=trajectory.cuda())
    operator = sampling_op.H @ sampling_op
    (result,) = operator(input_data.cuda())
    assert result.is_cuda

    # Create on GPU, transfer to CPU and run on CPU
    sampling_op = CartesianSamplingOp(encoding_matrix=encoding_matrix, traj=trajectory.cuda())
    operator = sampling_op.H @ sampling_op
    operator.cpu()
    (result,) = operator(input_data)
    assert result.is_cpu


def test_cart_masking_op_from_mask():
    """Test the CartesianMaskingOp from a mask."""
    rng = RandomGenerator(seed=0)
    mask = rng.bool_tensor(size=(1, 1, 1, 40, 60))
    masking_op = CartesianMaskingOp(mask)
    data = rng.complex64_tensor(size=(1, 1, 1, 40, 60))
    (actual,) = masking_op(data)
    assert torch.allclose(actual, data * mask)


def test_cart_masking_op_adjointness():
    """Test the adjointness of the CartesianMaskingOp."""
    rng = RandomGenerator(seed=0)
    mask = rng.bool_tensor(size=(1, 1, 1, 40, 60))
    masking_op = CartesianMaskingOp(mask)
    u = rng.complex64_tensor(size=(1, 1, 1, 40, 60))
    v = rng.complex64_tensor(size=(1, 1, 1, 40, 60))
    dotproduct_adjointness_test(masking_op, u, v)


@SAMPLING_PARAMETERS
def test_cart_masking_op_from_trajectory(sampling: str) -> None:
    """Test the CartesianMaskingOp creation from a trajectory."""
    type_kx = 'uniform'
    type_ky = 'non-uniform' if sampling == 'cartesian_and_non_cartesian' else 'uniform'
    type_kz = 'non-uniform' if sampling == 'cartesian_and_non_cartesian' else 'uniform'
    k_shape = (2, 5, 20, 40, 60)
    nkx = (2, 1, 1, 1, 60)
    nky = (2, 1, 1, 40, 1)
    nkz = (2, 1, 20, 1, 1)
    trajectory = create_traj(nkx, nky, nkz, type_kx, type_ky, type_kz)
    trajectory = subsample_traj(trajectory, sampling, k_shape)
    encoding_matrix = SpatialDimension(k_shape[-3], k_shape[-2], k_shape[-1])
    sampling_op = CartesianSamplingOp(encoding_matrix=encoding_matrix, traj=trajectory)
    masking_op = CartesianMaskingOp.from_trajectory(trajectory, encoding_matrix)
    rng = RandomGenerator(seed=0)
    u = rng.complex64_tensor(size=k_shape)
    torch.testing.assert_close(masking_op(u), (sampling_op.H @ sampling_op)(u))
