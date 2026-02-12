"""Tests the CsmData class."""

from collections.abc import Callable

import pytest
import torch
from mr2.data import CsmData, KData, KHeader, SpatialDimension
from mr2.data.traj_calculators.KTrajectoryCartesian import KTrajectoryCartesian
from mr2.operators import FourierOp

from tests import relative_image_difference
from tests.algorithms.csm.test_walsh import multi_coil_image
from tests.phantoms import EllipsePhantomTestData


def create_matching_idata_kdata(ismrmrd_cart_single_rep):
    """Create matching idata and kdata for csm tests."""
    kdata = KData.from_file(ismrmrd_cart_single_rep.filename, KTrajectoryCartesian())
    ellipses = EllipsePhantomTestData(n_y=kdata.header.recon_matrix.y, n_x=kdata.header.recon_matrix.x)
    idata, _ = multi_coil_image(n_coils=4, ph_ellipse=ellipses, random_kheader=kdata.header)
    fourier_op = FourierOp.from_kdata(kdata)
    kdata = KData(kdata.header, fourier_op(idata.data)[0], traj=kdata.traj)
    return idata, kdata


@pytest.mark.parametrize('csm_method', [CsmData.from_idata_walsh, CsmData.from_idata_inati])
def test_CsmData_smoothing_width(
    csm_method: Callable, ellipse_phantom: EllipsePhantomTestData, random_kheader: KHeader
) -> None:
    """CsmData SpatialDimension and int for smoothing width."""
    idata, _ = multi_coil_image(n_coils=4, ph_ellipse=ellipse_phantom, random_kheader=random_kheader)

    # Estimate coil sensitivity maps using SpatialDimension for smoothing width
    smoothing_width = SpatialDimension(z=1, y=5, x=5)
    csm_using_spatial_dimension = csm_method(idata, smoothing_width)

    # Estimate coil sensitivity maps using int for smoothing width
    csm_using_int = csm_method(idata, smoothing_width=5)

    # assert that both coil sensitivity maps are equal, not just close
    assert torch.equal(csm_using_spatial_dimension.data, csm_using_int.data)


@pytest.mark.cuda
@pytest.mark.parametrize('csm_method', [CsmData.from_idata_walsh, CsmData.from_idata_inati])
def test_CsmData_cuda(csm_method: Callable, ellipse_phantom: EllipsePhantomTestData, random_kheader: KHeader) -> None:
    """CsmData obtained on GPU in CUDA memory."""
    idata, csm_ref = multi_coil_image(n_coils=4, ph_ellipse=ellipse_phantom, random_kheader=random_kheader)

    # Estimate coil sensitivity maps
    smoothing_width = SpatialDimension(z=1, y=5, x=5)
    csm = csm_method(idata.cuda(), smoothing_width)

    # Phase is only relative in csm calculation, therefore only the abs values are compared.
    assert relative_image_difference(torch.abs(csm.data), torch.abs(csm_ref.cuda())) <= 0.01


@pytest.mark.parametrize('csm_method', [CsmData.from_idata_walsh, CsmData.from_idata_inati])
def test_CsmData_downsampling(
    csm_method: Callable, ellipse_phantom: EllipsePhantomTestData, random_kheader: KHeader
) -> None:
    """CsmData downsampling does not change smooth coil sensitivity maps."""
    idata, _ = multi_coil_image(n_coils=4, ph_ellipse=ellipse_phantom, random_kheader=random_kheader)

    # Estimate coil sensitivity maps on original image size with smoothing width = 3
    smoothing_width = SpatialDimension(z=1, y=3, x=3)
    csm_from_original_image = csm_method(idata, smoothing_width)

    # Estimate coil sensitivity maps on 3x downsampled image with smoothing width = 1
    csm_from_lowres_image = csm_method(idata, downsampled_size=idata.data.shape[-1] // 3, smoothing_width=1)

    # assert that both coil sensitivity maps are equal, not just close
    torch.testing.assert_close(csm_from_original_image.data, csm_from_lowres_image.data, rtol=1e-2, atol=1e-2)


def test_CsmData_kdata_walsh(ismrmrd_cart_single_rep) -> None:
    """CsmData using Walsh method should be the same for idata and kdata."""
    idata, kdata = create_matching_idata_kdata(ismrmrd_cart_single_rep)
    csm_from_kdata = CsmData.from_kdata_walsh(kdata)
    csm_from_idata = CsmData.from_idata_walsh(idata)
    torch.testing.assert_close(csm_from_kdata.data, csm_from_idata.data, rtol=1e-5, atol=1e-5)


def test_CsmData_kdata_inati(ismrmrd_cart_single_rep) -> None:
    """CsmData using Inati method should be the same for idata and kdata."""
    idata, kdata = create_matching_idata_kdata(ismrmrd_cart_single_rep)
    csm_from_kdata = CsmData.from_kdata_inati(kdata)
    csm_from_idata = CsmData.from_idata_inati(idata)
    torch.testing.assert_close(csm_from_kdata.data, csm_from_idata.data, rtol=1e-5, atol=1e-5)
