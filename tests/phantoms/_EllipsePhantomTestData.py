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

import torch
from mrpro.phantoms import EllipseParameters
from mrpro.phantoms import EllipsePhantom


class EllipsePhantomTestData:
    """Create ellipse phantom for testing.

    Parameters
    ----------
    n_y
        number of voxels along y
    n_x
        number of voxels along x
    """

    def __init__(self, n_y: int = 512, n_x: int = 256):
        # Define image size and k-space matrix
        self.n_x: int = n_x
        self.n_y: int = n_y
        [self.kx, self.ky] = torch.meshgrid(
            torch.linspace(-n_x // 2, n_x // 2 - 1, n_x),
            torch.linspace(-n_y // 2, n_y // 2 - 1, n_y),
            indexing='xy',
        )

        # Define five ellipses
        self.test_ellipses = [
            EllipseParameters(center_x=0.1, center_y=0.0, radius_x=0.1, radius_y=0.25, intensity=1),
            EllipseParameters(center_x=0.3, center_y=0.3, radius_x=0.1, radius_y=0.1, intensity=2),
            EllipseParameters(center_x=0.1, center_y=0.1, radius_x=0.1, radius_y=0.1, intensity=3),
            EllipseParameters(center_x=-0.2, center_y=-0.2, radius_x=0.1, radius_y=0.1, intensity=4),
            EllipseParameters(center_x=-0.3, center_y=-0.3, radius_x=0.1, radius_y=0.1, intensity=5),
        ]
        self.phantom = EllipsePhantom(self.test_ellipses)
