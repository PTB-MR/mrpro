"""Tests the iterative Walsh algorithm."""

import torch
from mrpro.algorithms.csm import iterative_walsh
from mrpro.data import IData, SpatialDimension
from mrpro.phantoms.coils import birdcage_2d
from tests.helper import relative_image_difference


def multi_coil_image(n_coils, ph_ellipse, random_kheader):
    """Create multi-coil image."""
    image_dimensions = SpatialDimension(z=1, y=ph_ellipse.n_y, x=ph_ellipse.n_x)

    # Create reference coil sensitivities
    csm_ref = birdcage_2d(n_coils, image_dimensions)

    # Create multi-coil phantom image data
    img = ph_ellipse.phantom.image_space(image_dimensions)
    # +1 to ensure that there is signal everywhere, for voxel == 0 csm cannot be determined.
    img_multi_coil = (img + 1) * csm_ref
    idata = IData.from_tensor_and_kheader(data=img_multi_coil, kheader=random_kheader)
    return (idata, csm_ref)


def test_iterative_Walsh(ellipse_phantom, random_kheader):
    """Test the iterative Walsh method."""
    idata, csm_ref = multi_coil_image(n_coils=4, ph_ellipse=ellipse_phantom, random_kheader=random_kheader)

    # Estimate coil sensitivity maps.
    # iterative_walsh should be applied for each other dimension separately
    smoothing_width = SpatialDimension(z=1, y=5, x=5)
    csm = iterative_walsh(idata.data[0, ...], smoothing_width, power_iterations=3)

    # Phase is only relative in csm calculation, therefore only the abs values are compared.
    assert relative_image_difference(torch.abs(csm), torch.abs(csm_ref[0, ...])) <= 0.01
