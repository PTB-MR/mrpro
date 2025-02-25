"""Tests for Fourier operator."""

from typing import cast

import pytest
import torch
from mrpro.data import KData, KTrajectory, SpatialDimension
from mrpro.data.enums import TrajType
from mrpro.data.traj_calculators import KTrajectoryCartesian
from mrpro.operators import FourierOp

from tests import RandomGenerator, dotproduct_adjointness_test
from tests.conftest import COMMON_MR_TRAJECTORIES, create_traj


def create_data(im_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz):
    random_generator = RandomGenerator(seed=0)

    # generate random image
    img = random_generator.complex64_tensor(size=im_shape)
    # create random trajectories
    trajectory = create_traj(k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)
    return img, trajectory


class NufftTrajektory(KTrajectory):
    """Always returns non-grid trajectory type."""

    def _traj_types(
        self,
        tolerance: float,
    ) -> tuple[tuple[TrajType, TrajType, TrajType], tuple[TrajType, TrajType, TrajType]]:
        true_types = super()._traj_types(tolerance)
        modified = tuple(tuple(t & (~TrajType.ONGRID) for t in ts) for ts in true_types)
        return cast(tuple[tuple[TrajType, TrajType, TrajType], tuple[TrajType, TrajType, TrajType]], modified)


@COMMON_MR_TRAJECTORIES
def test_fourier_op_fwd_adj_property(
    im_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz, type_k0, type_k1, type_k2
):
    """Test adjoint property of Fourier operator."""

    # generate random images and k-space trajectories
    _, trajectory = create_data(im_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)

    # create operator
    recon_matrix = SpatialDimension(im_shape[-3], im_shape[-2], im_shape[-1])
    encoding_matrix = SpatialDimension(
        int(trajectory.kz.max() - trajectory.kz.min() + 1),
        int(trajectory.ky.max() - trajectory.ky.min() + 1),
        int(trajectory.kx.max() - trajectory.kx.min() + 1),
    )
    fourier_op = FourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix, traj=trajectory)

    # test adjoint property; i.e. <Fu,v> == <u, F^Hv> for all u,v
    random_generator = RandomGenerator(seed=0)
    u = random_generator.complex64_tensor(size=im_shape)
    v = random_generator.complex64_tensor(size=k_shape)
    dotproduct_adjointness_test(fourier_op, u, v)


@COMMON_MR_TRAJECTORIES
def test_fourier_op_gram(im_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz, type_k0, type_k1, type_k2):
    """Test gram of Fourier operator."""
    img, trajectory = create_data(im_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)

    recon_matrix = SpatialDimension(im_shape[-3], im_shape[-2], im_shape[-1])
    encoding_matrix = SpatialDimension(
        int(trajectory.kz.max() - trajectory.kz.min() + 1),
        int(trajectory.ky.max() - trajectory.ky.min() + 1),
        int(trajectory.kx.max() - trajectory.kx.min() + 1),
    )
    fourier_op = FourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix, traj=trajectory)

    (expected,) = (fourier_op.H @ fourier_op)(img)
    (actual,) = fourier_op.gram(img)

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def test_fourier_op_cartesian_sorting(ismrmrd_cart):
    """Verify correct sorting of Cartesian k-space data before FFT."""
    kdata = KData.from_file(ismrmrd_cart.filename, KTrajectoryCartesian())
    ff_op = FourierOp.from_kdata(kdata)
    (img,) = ff_op.adjoint(kdata.data)

    # shuffle the kspace points along k0
    permutation_index = RandomGenerator(13).randperm(kdata.data.shape[-1])
    kdata_unsorted = KData(
        header=kdata.header,
        data=kdata.data[..., permutation_index],
        traj=KTrajectory.from_tensor(kdata.traj.as_tensor()[..., permutation_index]),
    )
    ff_op_unsorted = FourierOp.from_kdata(kdata_unsorted)
    (img_unsorted,) = ff_op_unsorted.adjoint(kdata_unsorted.data)

    torch.testing.assert_close(img, img_unsorted)


@pytest.mark.parametrize(
    ('im_shape', 'k_shape', 'nkx', 'nky', 'nkz', 'type_kx', 'type_ky', 'type_kz'),  # parameter names
    [
        (  # 3d single shot stack of spiral but cartesian FFT dimension in ky and k2
            (1, 2, 96, 4, 128),  # im_shape
            (1, 2, 4, 1, 192),  # k_shape
            (1, 1, 1, 192),  # nkx
            (1, 4, 1, 1),  # nky
            (1, 1, 1, 192),  # nkz
            'non-uniform',  # type_kx
            'uniform',  # type_ky
            'non-uniform',  # type_kz
        ),
        (  # radial phase encoding, cartesian FFT dimension not aligned with corresponding k2, k1, k0 dimensions
            (2, 3, 48, 16, 32),  # im_shape
            (2, 3, 96, 18, 64),  # k_shape
            (2, 1, 18, 64),  # nkx
            (2, 96, 1, 1),  # nky - Cartesian ky dimension defined along k2 rather than k1
            (2, 1, 18, 64),  # nkz
            'non-uniform',  # type_kx
            'uniform',  # type_ky
            'non-uniform',  # type_kz
        ),
    ],
    ids=['3d_single_shot_stack_of_spirals_ky_in_k2', 'cartesian_fft_dims_not_aligned_with_k2_k1_k0_dims'],
)
def test_fourier_op_not_supported_traj(im_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz):
    """Test trajectory not supported by Fourier operator."""

    # generate random images and k-space trajectories
    img, trajectory = create_data(im_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)

    # create operator
    recon_matrix = SpatialDimension(im_shape[-3], im_shape[-2], im_shape[-1])
    encoding_matrix = SpatialDimension(
        int(trajectory.kz.max() - trajectory.kz.min() + 1),
        int(trajectory.ky.max() - trajectory.ky.min() + 1),
        int(trajectory.kx.max() - trajectory.kx.min() + 1),
    )
    with pytest.raises(NotImplementedError, match='Cartesian FFT dims need to be aligned'):
        FourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix, traj=trajectory)


@COMMON_MR_TRAJECTORIES
def test_fourier_op_fft_nufft_forward(
    im_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz, type_k0, type_k1, type_k2
):
    """Test Nufft vs FFT for Fourier operator."""
    if not any(t == 'uniform' for t in [type_kx, type_ky, type_kz]):
        return  # only test for uniform trajectories

    img, trajectory = create_data(im_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)

    recon_matrix = SpatialDimension(im_shape[-3], im_shape[-2], im_shape[-1])
    encoding_matrix = SpatialDimension(
        int(trajectory.kz.max() - trajectory.kz.min() + 1),
        int(trajectory.ky.max() - trajectory.ky.min() + 1),
        int(trajectory.kx.max() - trajectory.kx.min() + 1),
    )
    fourier_op = FourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix, traj=trajectory)

    nufft_fourier_op = FourierOp(
        recon_matrix=recon_matrix,
        encoding_matrix=encoding_matrix,
        traj=NufftTrajektory(trajectory.kz, trajectory.ky, trajectory.kx),
    )

    (result_normal,) = fourier_op(img)
    (result_nufft,) = nufft_fourier_op(img)
    torch.testing.assert_close(result_normal, result_nufft, atol=3e-4, rtol=5e-3)


@COMMON_MR_TRAJECTORIES
def test_fourier_op_fft_nufft_adjoint(
    im_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz, type_k0, type_k1, type_k2
):
    """Test AdjointNufft vs IFFT for Fourier operator."""
    if not any(t == 'uniform' for t in [type_kx, type_ky, type_kz]):
        return  # only test for uniform trajectories
    img, trajectory = create_data(im_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)
    recon_matrix = SpatialDimension(im_shape[-3], im_shape[-2], im_shape[-1])
    encoding_matrix = SpatialDimension(
        int(trajectory.kz.max() - trajectory.kz.min() + 1),
        int(trajectory.ky.max() - trajectory.ky.min() + 1),
        int(trajectory.kx.max() - trajectory.kx.min() + 1),
    )
    fourier_op = FourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix, traj=trajectory)

    nufft_fourier_op = FourierOp(
        recon_matrix=recon_matrix,
        encoding_matrix=encoding_matrix,
        traj=NufftTrajektory(trajectory.kz, trajectory.ky, trajectory.kx),
    )

    (k,) = fourier_op(img)
    (result_normal,) = fourier_op.H(k)
    (result_nufft,) = nufft_fourier_op.H(k)
    torch.testing.assert_close(result_normal, result_nufft, atol=3e-4, rtol=5e-3)


@COMMON_MR_TRAJECTORIES
def test_fourier_op_fft_nufft_gram(
    im_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz, type_k0, type_k1, type_k2
):
    """Test Nufft gram vs FFt gram for Fourier operator."""
    if not any(t == 'uniform' for t in [type_kx, type_ky, type_kz]):
        return  # only test for uniform trajectories
    img, trajectory = create_data(im_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)
    recon_matrix = SpatialDimension(im_shape[-3], im_shape[-2], im_shape[-1])
    encoding_matrix = SpatialDimension(
        int(trajectory.kz.max() - trajectory.kz.min() + 1),
        int(trajectory.ky.max() - trajectory.ky.min() + 1),
        int(trajectory.kx.max() - trajectory.kx.min() + 1),
    )
    fourier_op = FourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix, traj=trajectory)

    nufft_fourier_op = FourierOp(
        recon_matrix=recon_matrix,
        encoding_matrix=encoding_matrix,
        traj=NufftTrajektory(trajectory.kz, trajectory.ky, trajectory.kx),
    )

    (result_normal,) = fourier_op.gram(img)
    (result_nufft,) = nufft_fourier_op.gram(img)
    torch.testing.assert_close(result_normal, result_nufft, atol=3e-4, rtol=5e-3)
