"""Tests the iterative Walsh algorithm."""

import torch
from mrpro.algorithms.csm import iterative_walsh
from mrpro.data import SpatialDimension
from tests.algorithms.csm.conftest import multi_coil_image
from tests.helper import relative_image_difference


def test_iterative_Walsh(ellipse_phantom, random_kheader):
    """Test the iterative Walsh method."""
    idata, csm_ref = multi_coil_image(n_coils=4, ph_ellipse=ellipse_phantom, random_kheader=random_kheader)

    # Estimate coil sensitivity maps.
    # iterative_walsh should be applied for each other dimension separately
    smoothing_width = SpatialDimension(z=1, y=5, x=5)
    csm = iterative_walsh(idata.data[0, ...], smoothing_width, power_iterations=3)

    # Phase is only relative in csm calculation, therefore only the abs values are compared.
    assert relative_image_difference(torch.abs(csm), torch.abs(csm_ref[0, ...])) <= 0.01
