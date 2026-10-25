"""Tests the iterative Walsh algorithm."""

import pytest
import torch
from mrpro.algorithms.csm import inati
from mrpro.data import SpatialDimension
from tests import relative_image_difference
from tests.algorithms.csm.conftest import multi_coil_image


def test_inati(ellipse_phantom, random_kheader):
    """Test the Inati method."""
    idata, csm_ref = multi_coil_image(n_coils=4, ph_ellipse=ellipse_phantom, random_kheader=random_kheader)

    # Estimate coil sensitivity maps.
    # inati should be applied for each other dimension separately
    smoothing_width = SpatialDimension(z=1, y=5, x=5)
    csm = inati(idata.data[0, ...], smoothing_width)

    # Phase is only relative in csm calculation, therefore only the abs values are compared.
    assert relative_image_difference(torch.abs(csm), torch.abs(csm_ref[0, ...])) <= 0.01


def test_inati_requires_positive_iterations() -> None:
    """Test that Inati validates the number of iterations."""
    coil_img = torch.ones(2, 1, 4, 4, dtype=torch.complex64)

    with pytest.raises(ValueError, match='n_iterations must be at least 1'):
        inati(coil_img, smoothing_width=1, n_iterations=0)
