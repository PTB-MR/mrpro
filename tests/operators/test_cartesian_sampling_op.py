"""Tests for the Cartesian sampling operator."""

import pytest
import torch
from mrpro.data import KTrajectory, SpatialDimension
from mrpro.operators import CartesianSamplingOp

from tests import RandomGenerator
from tests.conftest import create_traj
from tests.helper import dotproduct_adjointness_test


def test_cart_sampling_op_data_match():
    # Create 3D uniform trajectory
    k_shape = (1, 5, 20, 40, 60)
    nkx = (1, 1, 1, 60)
    nky = (1, 1, 40, 1)
    nkz = (1, 20, 1, 1)
    sx = 'uf'
    sy = 'uf'
    sz = 'uf'
    trajectory = create_traj(k_shape, nkx, nky, nkz, sx, sy, sz)

    # Create matching data
    random_generator = RandomGenerator(seed=0)
    kdata = random_generator.complex64_tensor(size=k_shape)

    # Create sampling operator
    encoding_matrix = SpatialDimension(k_shape[-3], k_shape[-2], k_shape[-1])
    sampling_op = CartesianSamplingOp(encoding_matrix=encoding_matrix, traj=trajectory)

    # Subsample data and trajectory
    kdata_sub = kdata[:, :, ::2, ::4, ::3]
    trajectory_sub = KTrajectory(
        kz=trajectory.kz[:, ::2, :, :],
        ky=trajectory.ky[:, :, ::4, :],
        kx=trajectory.kx[:, :, :, ::3],
    )
    sampling_op_sub = CartesianSamplingOp(encoding_matrix=encoding_matrix, traj=trajectory_sub)

    # Verify that the fully-sampled sampling operator does not do anything because the data is already sorted
    assert not sampling_op._needs_indexing

    # Verify identical shape
    (k,) = sampling_op.adjoint(kdata)
    (k_sub,) = sampling_op_sub.adjoint(kdata_sub)
    assert k.shape == k_sub.shape

    # Verify data is correctly sorted
    torch.testing.assert_close(kdata[:, :, ::2, ::4, ::3], k_sub[:, :, ::2, ::4, ::3])


@pytest.mark.parametrize(
    'sampling',
    [
        'random',
        'partial_echo',
        'partial_fourier',
        'regular_undersampling',
        'random_undersampling',
        'different_random_undersampling',
    ],
)
def test_cart_sampling_op_fwd_adj(sampling):
    """Test adjoint property of Cartesian sampling operator."""

    # Create 3D uniform trajectory
    k_shape = (2, 5, 20, 40, 60)
    nkx = (2, 1, 1, 60)
    nky = (2, 1, 40, 1)
    nkz = (2, 20, 1, 1)
    sx = 'uf'
    sy = 'uf'
    sz = 'uf'
    trajectory_tensor = create_traj(k_shape, nkx, nky, nkz, sx, sy, sz).as_tensor()

    # Subsample data and trajectory
    match sampling:
        case 'random':
            random_idx = torch.randperm(k_shape[-2])
            trajectory = KTrajectory.from_tensor(trajectory_tensor[..., random_idx, :])
        case 'partial_echo':
            trajectory = KTrajectory.from_tensor(trajectory_tensor[..., : k_shape[-1] // 2])
        case 'partial_fourier':
            trajectory = KTrajectory.from_tensor(trajectory_tensor[..., : k_shape[-3] // 2, : k_shape[-2] // 2, :])
        case 'regular_undersampling':
            trajectory = KTrajectory.from_tensor(trajectory_tensor[..., ::3, ::5, :])
        case 'random_undersampling':
            random_idx = torch.randperm(k_shape[-2])
            trajectory = KTrajectory.from_tensor(trajectory_tensor[..., random_idx[: k_shape[-2] // 2], :])
        case 'different_random_undersampling':
            traj_list = [
                traj_one_other[..., torch.randperm(k_shape[-2])[: k_shape[-2] // 2], :]
                for traj_one_other in trajectory_tensor.unbind(1)
            ]
            trajectory = KTrajectory.from_tensor(torch.stack(traj_list, dim=1))
        case _:
            raise NotImplementedError(f'Test {sampling} not implemented.')

    encoding_matrix = SpatialDimension(k_shape[-3], k_shape[-2], k_shape[-1])
    sampling_op = CartesianSamplingOp(encoding_matrix=encoding_matrix, traj=trajectory)

    # Test adjoint property; i.e. <Fu,v> == <u, F^Hv> for all u,v
    random_generator = RandomGenerator(seed=0)
    u = random_generator.complex64_tensor(size=k_shape)
    v = random_generator.complex64_tensor(size=k_shape[:2] + trajectory.as_tensor().shape[2:])
    dotproduct_adjointness_test(sampling_op, u, v)
