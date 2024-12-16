"""Tests for Fourier operator."""

import torch
from mrpro.data import KData, KTrajectory, SpatialDimension
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


@COMMON_MR_TRAJECTORIES
def test_fourier_op_fwd_adj_property(
    im_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz, type_k0, type_k1, type_k2
):
    """Test adjoint property of Fourier operator."""

    # generate random images and k-space trajectories
    img, trajectory = create_data(im_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)

    # create operator
    recon_matrix = SpatialDimension(im_shape[-3], im_shape[-2], im_shape[-1])
    encoding_matrix = SpatialDimension(
        int(trajectory.kz.max() - trajectory.kz.min() + 1),
        int(trajectory.ky.max() - trajectory.ky.min() + 1),
        int(trajectory.kx.max() - trajectory.kx.min() + 1),
    )
    fourier_op = FourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix, traj=trajectory)

    # apply forward operator
    (kdata,) = fourier_op(img)

    # test adjoint property; i.e. <Fu,v> == <u, F^Hv> for all u,v
    random_generator = RandomGenerator(seed=0)
    u = random_generator.complex64_tensor(size=img.shape)
    v = random_generator.complex64_tensor(size=kdata.shape)
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
