"""Ellipse phantom for testing."""

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


class EllipsePhantomTestData:
    """Create ellipse phantom for testing.

    Parameters
    ----------
    nx
        number of voxels along x
    ny
        number of voxels along y
    """

    def __init__(self, nx: int = 512, ny: int = 256):
        # Define image size and k-space matrix
        self.nx: int = nx
        self.ny: int = ny
        [self.kx, self.ky] = np.meshgrid(range(-self.nx // 2, self.nx // 2), range(-self.ny // 2, self.ny // 2))

        # Define five ellipses
        self.test_ellipses = [
            EllipsePars(center_x=0.1, center_y=0.0, radius_x=0.1, radius_y=0.25, intensity=1),
            EllipsePars(center_x=0.3, center_y=0.3, radius_x=0.1, radius_y=0.1, intensity=2),
            EllipsePars(center_x=0.1, center_y=0.1, radius_x=0.1, radius_y=0.1, intensity=3),
            EllipsePars(center_x=-0.2, center_y=-0.2, radius_x=0.1, radius_y=0.1, intensity=4),
            EllipsePars(center_x=-0.3, center_y=-0.3, radius_x=0.1, radius_y=0.1, intensity=5),
        ]
        self.phantom = EllipsePhantom(self.test_ellipses)
