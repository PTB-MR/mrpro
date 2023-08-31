"""Tests for Phantoms."""

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

import numpy as np
import pytest

from tests.phantoms._EllipsePhantomTestData import EllipsePhantomTestData
from tests.utils import kspace_to_image
from tests.utils import rel_image_diff


@pytest.fixture(scope='session')
def ph_ellipse():
    return EllipsePhantomTestData()


def test_image_space(ph_ellipse):
    """Check if image space has correct shape."""
    im = ph_ellipse.phantom.image_space(ph_ellipse.nx, ph_ellipse.ny)
    assert im.shape == (ph_ellipse.ny, ph_ellipse.nx)


def test_kspace_correct_shape(ph_ellipse):
    """Check if kspace has correct shape."""
    kdat = ph_ellipse.phantom.kspace(ph_ellipse.kx, ph_ellipse.ky)
    assert kdat.shape == (ph_ellipse.ny, ph_ellipse.nx)


def test_kspace_raises_error(ph_ellipse):
    """Check if kspace raises error if kx and ky have different shapes."""
    [kx_, _] = np.meshgrid(
        range(-ph_ellipse.nx // 2, ph_ellipse.nx // 2), range(-ph_ellipse.ny // 2, ph_ellipse.ny // 2 + 1)
    )
    with pytest.raises(ValueError):
        ph_ellipse.phantom.kspace(kx_, ph_ellipse.ky)


def test_kspace_image_match(ph_ellipse):
    """Check if fft of kspace matches image."""
    im = ph_ellipse.phantom.image_space(ph_ellipse.nx, ph_ellipse.ny)
    kdat = ph_ellipse.phantom.kspace(ph_ellipse.kx, ph_ellipse.ky)
    irec = kspace_to_image(kdat)
    # Due to discretisation artifacts the reconstructed image will be different to the reference image. Using standard
    # testing functions such as numpy.testing.assert_almost_equal fails because there are few voxels with high
    # differences along the edges of the elliptic objects.
    assert rel_image_diff(irec, im) <= 0.05
