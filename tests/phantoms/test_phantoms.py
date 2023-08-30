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

from mrpro.phantoms import EllipsePars
from mrpro.phantoms import EllipsePhantom
from tests.utils import kspace_to_image
from tests.utils import rel_image_diff


def test_EllipsePhantom():
    # Define image size and k-space matrix
    nx = 512
    ny = 256
    [kx, ky] = np.meshgrid(range(-nx // 2, nx // 2), range(-ny // 2, ny // 2))

    # Define five ellipses
    test_ellipses = [
        EllipsePars(center_x=0.1, center_y=0.0, radius_x=0.1, radius_y=0.25, intensity=1),
        EllipsePars(center_x=0.3, center_y=0.3, radius_x=0.1, radius_y=0.1, intensity=2),
        EllipsePars(center_x=0.1, center_y=0.1, radius_x=0.1, radius_y=0.1, intensity=3),
        EllipsePars(center_x=-0.2, center_y=-0.2, radius_x=0.1, radius_y=0.1, intensity=4),
        EllipsePars(center_x=-0.3, center_y=-0.3, radius_x=0.1, radius_y=0.1, intensity=5),
    ]

    # Create phantom
    ph = EllipsePhantom(test_ellipses)

    # Get image and k-space representation of phantom
    im = ph.image_space(nx, ny)
    kdat = ph.kspace(kx, ky)

    # Reconstruct and compare
    irec = kspace_to_image(kdat)

    # Due to discretisation artifacts the reconstructed image will be different to the reference image. Using standard
    # testing functions such as numpy.testing.assert_almost_equal fails because there are few voxels with high
    # differences along the edges of the elliptic objects.
    assert rel_image_diff(irec, im) <= 0.05
