"""Tests for ellipse phantom."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import pytest
import torch

from mrpro.data import SpatialDimension
from mrpro.operators import FastFourierOp
from tests.helper import rel_image_diff


def test_image_space(ph_ellipse):
    """Check if image space has correct shape."""
    im_dim = SpatialDimension(z=1, y=ph_ellipse.ny, x=ph_ellipse.nx)
    im = ph_ellipse.phantom.image_space(im_dim)
    assert im.shape[-2:] == (ph_ellipse.ny, ph_ellipse.nx)


def test_kspace_correct_shape(ph_ellipse):
    """Check if kspace has correct shape."""
    kdat = ph_ellipse.phantom.kspace(ph_ellipse.ky, ph_ellipse.kx)
    assert kdat.shape == (ph_ellipse.ny, ph_ellipse.nx)


def test_kspace_raises_error(ph_ellipse):
    """Check if kspace raises error if kx and ky have different shapes."""
    [kx_, _] = torch.meshgrid(
        torch.linspace(-ph_ellipse.nx // 2, ph_ellipse.nx // 2, ph_ellipse.nx + 1),
        torch.linspace(-ph_ellipse.ny // 2, ph_ellipse.ny // 2 + 1, ph_ellipse.ny),
        indexing='xy',
    )
    with pytest.raises(ValueError):
        ph_ellipse.phantom.kspace(ph_ellipse.ky, kx_)


def test_kspace_image_match(ph_ellipse):
    """Check if fft of kspace matches image."""
    im_dim = SpatialDimension(z=1, y=ph_ellipse.ny, x=ph_ellipse.nx)
    im = ph_ellipse.phantom.image_space(im_dim)
    kdat = ph_ellipse.phantom.kspace(ph_ellipse.ky, ph_ellipse.kx)
    FFOp = FastFourierOp(dim=(-1, -2))
    (irec,) = FFOp.adjoint(kdat)
    # Due to discretisation artifacts the reconstructed image will be different to the reference image. Using standard
    # testing functions such as numpy.testing.assert_almost_equal fails because there are few voxels with high
    # differences along the edges of the elliptic objects.
    assert rel_image_diff(irec, im[0, 0, 0, :, :]) <= 0.05
