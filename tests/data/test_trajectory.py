"""Tests for KTrajectory class."""

import pytest
import torch
from einops import rearrange
from mrpro.data import KTrajectory
from mrpro.data.enums import TrajType

from tests.conftest import COMMON_MR_TRAJECTORIES, create_traj


def test_trajectory_repeat_detection_tol(cartesian_grid):
    """Test the automatic detection of repeated values."""
    n_k0 = 10
    n_k1 = 20
    n_k2 = 30
    kz_full, ky_full, kx_full = cartesian_grid(n_k2, n_k1, n_k0, jitter=0.1)
    trajectory_sparse = KTrajectory(kz_full, ky_full, kx_full, repeat_detection_tolerance=0.1)

    assert trajectory_sparse.broadcasted_shape == (1, n_k2, n_k1, n_k0)
    assert trajectory_sparse.kx.shape == (1, 1, 1, n_k0)
    assert trajectory_sparse.ky.shape == (1, 1, n_k1, 1)
    assert trajectory_sparse.kz.shape == (1, n_k2, 1, 1)


def test_trajectory_repeat_detection_exact(cartesian_grid):
    """Test the automatic detection of repeated values."""
    n_k0 = 10
    n_k1 = 20
    n_k2 = 30
    kz_full, ky_full, kx_full = cartesian_grid(n_k2, n_k1, n_k0, jitter=0.1)
    trajectory_full = KTrajectory(kz_full, ky_full, kx_full, repeat_detection_tolerance=None)

    assert trajectory_full.broadcasted_shape == (1, n_k2, n_k1, n_k0)
    assert trajectory_full.kx.shape == (1, n_k2, n_k1, n_k0)
    assert trajectory_full.ky.shape == (1, n_k2, n_k1, n_k0)
    assert trajectory_full.kz.shape == (1, n_k2, n_k1, n_k0)


def test_trajectory_tensor_conversion(cartesian_grid):
    n_k0 = 10
    n_k1 = 20
    n_k2 = 30
    kz_full, ky_full, kx_full = cartesian_grid(n_k2, n_k1, n_k0, jitter=0.0)
    trajectory = KTrajectory(kz_full, ky_full, kx_full)
    tensor = torch.stack((kz_full, ky_full, kx_full), dim=0).to(torch.float32)

    tensor_from_traj = trajectory.as_tensor()  # stack_dim=0
    tensor_from_traj_dim2 = rearrange(trajectory.as_tensor(stack_dim=2), 'other k2 dim k1 k0->dim other k2 k1 k0')
    tensor_from_traj_from_tensor_dim3 = KTrajectory.from_tensor(
        rearrange(tensor, 'dim other k2 k1 k0->other k2 k1 dim k0'), stack_dim=3
    ).as_tensor()
    tensor_from_traj_from_tensor = KTrajectory.from_tensor(tensor).as_tensor()  # stack_dim=0

    torch.testing.assert_close(tensor, tensor_from_traj)
    torch.testing.assert_close(tensor, tensor_from_traj_dim2)
    torch.testing.assert_close(tensor, tensor_from_traj_from_tensor)
    torch.testing.assert_close(tensor, tensor_from_traj_from_tensor_dim3)


def test_trajectory_raise_not_broadcastable():
    """Non broadcastable shapes should raise."""
    kx = ky = torch.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4)
    kz = torch.arange(1 * 2 * 3 * 100).reshape(1, 2, 3, 100)
    with pytest.raises(ValueError):
        KTrajectory(kz, ky, kx)


def test_trajectory_raise_wrong_dim():
    """Wrong number of dimensions after broadcasting should raise."""
    kx = ky = kz = torch.arange(1 * 2 * 3).reshape(1, 2, 3)
    with pytest.raises(ValueError):
        KTrajectory(kz, ky, kx)


def test_trajectory_to_float64(cartesian_grid):
    """Change KTrajectory dtype to float64."""
    n_k0 = 10
    n_k1 = 20
    n_k2 = 30
    kz_full, ky_full, kx_full = cartesian_grid(n_k2, n_k1, n_k0, jitter=0.0)
    trajectory = KTrajectory(kz_full, ky_full, kx_full)
    trajectory_float64 = trajectory.to(dtype=torch.float64)
    assert trajectory_float64.kz.dtype == torch.float64
    assert trajectory_float64.ky.dtype == torch.float64
    assert trajectory_float64.kx.dtype == torch.float64
    assert trajectory.kz.dtype == torch.float32
    assert trajectory.ky.dtype == torch.float32
    assert trajectory.kx.dtype == torch.float32


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.int32, torch.int64])
def test_trajectory_floating_dtype(dtype):
    """Test if the trajectory will always be converted to float"""
    ks = torch.ones(3, 1, 1, 1, 1, dtype=dtype)
    traj = KTrajectory(*ks)
    if dtype.is_floating_point:
        # keep as as
        assert traj.kz.dtype == dtype
        assert traj.ky.dtype == dtype
        assert traj.kx.dtype == dtype
    else:
        # convert to float32
        assert traj.kz.dtype == torch.float32
        assert traj.ky.dtype == torch.float32
        assert traj.kx.dtype == torch.float32


@pytest.mark.cuda
def test_trajectory_cuda(cartesian_grid):
    """Move KTrajectory object to CUDA memory."""
    n_k0 = 10
    n_k1 = 20
    n_k2 = 30
    kz_full, ky_full, kx_full = cartesian_grid(n_k2, n_k1, n_k0, jitter=0.0)
    trajectory = KTrajectory(kz_full, ky_full, kx_full)

    trajectory_cuda = trajectory.cuda()
    assert trajectory_cuda.kz.is_cuda
    assert trajectory_cuda.ky.is_cuda
    assert trajectory_cuda.kx.is_cuda

    assert trajectory.kz.is_cpu
    assert trajectory.ky.is_cpu
    assert trajectory.kx.is_cpu

    assert trajectory_cuda.is_cuda
    assert trajectory.is_cpu

    assert not trajectory_cuda.is_cpu
    assert not trajectory.is_cuda


@pytest.mark.cuda
def test_trajectory_cpu(cartesian_grid):
    """Move KTrajectory object to CUDA memory and back to CPU memory."""
    n_k0 = 10
    n_k1 = 20
    n_k2 = 30
    kz_full, ky_full, kx_full = cartesian_grid(n_k2, n_k1, n_k0, jitter=0.0)
    trajectory = KTrajectory(kz_full, ky_full, kx_full)

    trajectory_cpu = trajectory.cuda().cpu()
    assert trajectory_cpu.kz.is_cpu
    assert trajectory_cpu.ky.is_cpu
    assert trajectory_cpu.kx.is_cpu


@COMMON_MR_TRAJECTORIES
def test_ktype_along_kzyx(im_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz, type_k0, type_k1, type_k2):
    """Test identification of traj types."""

    # Generate random k-space trajectories
    trajectory = create_traj(k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)

    # Find out the type of the kz, ky and kz dimensions
    single_value_dims = [d for d, s in zip((-3, -2, -1), (type_kz, type_ky, type_kx), strict=True) if s == 'z']
    on_grid_dims = [d for d, s in zip((-3, -2, -1), (type_kz, type_ky, type_kx), strict=True) if s == 'uf']
    not_on_grid_dims = [d for d, s in zip((-3, -2, -1), (type_kz, type_ky, type_kx), strict=True) if s == 'nuf']

    # check dimensions which are of shape 1 and do not need any transform
    assert all(trajectory.type_along_kzyx[dim] & TrajType.SINGLEVALUE for dim in single_value_dims)

    # Check dimensions which are on a grid and require FFT
    assert all(trajectory.type_along_kzyx[dim] & TrajType.ONGRID for dim in on_grid_dims)

    # Check dimensions which are not on a grid and require NUFFT
    assert all(
        ~(trajectory.type_along_kzyx[dim] & (TrajType.SINGLEVALUE | TrajType.ONGRID)) for dim in not_on_grid_dims
    )


@COMMON_MR_TRAJECTORIES
def test_ktype_along_k210(im_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz, type_k0, type_k1, type_k2):
    """Test identification of traj types."""

    # Generate random k-space trajectories
    trajectory = create_traj(k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)

    # Find out the type of the k2, k1 and k0 dimensions
    single_value_dims = [d for d, s in zip((-3, -2, -1), (type_k2, type_k1, type_k0), strict=True) if s == 'z']
    on_grid_dims = [d for d, s in zip((-3, -2, -1), (type_k2, type_k1, type_k0), strict=True) if s == 'uf']
    not_on_grid_dims = [d for d, s in zip((-3, -2, -1), (type_k2, type_k1, type_k0), strict=True) if s == 'nuf']

    # check dimensions which are of shape 1 and do not need any transform
    assert all(trajectory.type_along_k210[dim] & TrajType.SINGLEVALUE for dim in single_value_dims)

    # Check dimensions which are on a grid and require FFT
    assert all(trajectory.type_along_k210[dim] & TrajType.ONGRID for dim in on_grid_dims)

    # Check dimensions which are not on a grid and require NUFFT
    assert all(
        ~(trajectory.type_along_k210[dim] & (TrajType.SINGLEVALUE | TrajType.ONGRID)) for dim in not_on_grid_dims
    )
