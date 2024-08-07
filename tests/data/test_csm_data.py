"""Tests the CsmData class."""

import dataclasses

import pytest
import torch
from mrpro.data import CsmData, SpatialDimension

from tests.algorithms.csm.test_iterative_walsh import multi_coil_image
from tests.helper import relative_image_difference


def test_CsmData_is_frozen_dataclass(random_test_data, random_kheader):
    """CsmData inherits frozen dataclass property from QData."""
    csm = CsmData(data=random_test_data, header=random_kheader)
    with pytest.raises(dataclasses.FrozenInstanceError):
        csm.data = random_test_data  # type: ignore[misc]


def test_CsmData_interactive_Walsh_smoothing_width(ellipse_phantom, random_kheader):
    """CsmData from iterative Walsh method using SpatialDimension and int for
    smoothing width."""
    idata, csm_ref = multi_coil_image(n_coils=4, ph_ellipse=ellipse_phantom, random_kheader=random_kheader)

    # Estimate coil sensitivity maps using SpatialDimension for smoothing width
    smoothing_width = SpatialDimension(z=5, y=5, x=5)
    csm_using_spatial_dimension = CsmData.from_idata_walsh(idata, smoothing_width)

    # Estimate coil sensitivity maps using int for smoothing width
    csm_using_int = CsmData.from_idata_walsh(idata, smoothing_width=5)

    # assert that both coil sensitivity maps are equal, not just close
    assert torch.equal(csm_using_spatial_dimension.data, csm_using_int.data)


@pytest.mark.cuda()
def test_CsmData_iterative_Walsh_cuda(ellipse_phantom, random_kheader):
    """CsmData obtained with the iterative Walsh method in CUDA memory."""
    idata, csm_ref = multi_coil_image(n_coils=4, ph_ellipse=ellipse_phantom, random_kheader=random_kheader)

    # Estimate coil sensitivity maps
    smoothing_width = SpatialDimension(z=1, y=5, x=5)
    csm = CsmData.from_idata_walsh(idata.cuda(), smoothing_width)

    # Phase is only relative in csm calculation, therefore only the abs values are compared.
    assert relative_image_difference(torch.abs(csm.data), torch.abs(csm_ref.cuda())) <= 0.01
