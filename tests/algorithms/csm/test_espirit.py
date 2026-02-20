"""Tests the espirit algorithm."""

import torch
from mr2.algorithms.csm import espirit
from tests.algorithms.csm.conftest import multi_coil_image
from tests.helper import relative_image_difference


def test_espirit(ellipse_phantom, random_kheader):
    """Test the espirit algorithm."""
    idata, csm_ref = multi_coil_image(n_coils=4, ph_ellipse=ellipse_phantom, random_kheader=random_kheader)

    # Estimate coil sensitivity maps.
    csm = espirit(idata.data[0, ...], img_shape=idata.data.shape[-3:])
    # Phase is only relative in csm calculation, therefore only the abs values are compared.

    assert relative_image_difference(torch.abs(csm), torch.abs(csm_ref[0, ...])) <= 0.01
