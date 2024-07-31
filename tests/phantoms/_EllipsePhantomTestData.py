"""Ellipse phantom for testing."""

import torch
from mrpro.phantoms import EllipseParameters, EllipsePhantom


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
