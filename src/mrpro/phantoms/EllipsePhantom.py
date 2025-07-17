"""Numerical phantom with ellipses."""

from collections.abc import Sequence

import torch
from einops import repeat

from mrpro.data.KData import KData
from mrpro.data.KHeader import KHeader
from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.phantoms.phantom_elements import EllipseParameters


class EllipsePhantom:
    """Numerical phantom as the sum of different ellipses."""

    def __init__(self, ellipses: Sequence[EllipseParameters] | None = None):
        """Initialize ellipse phantom.

        Parameters
        ----------
        ellipses
            Parameters defining the ellipses.
            If `None`, defaults to three ellipses with different parameters.
        """
        if ellipses is None:
            self.ellipses = [
                EllipseParameters(center_x=0.2, center_y=0.2, radius_x=0.1, radius_y=0.25, intensity=1),
                EllipseParameters(center_x=0.1, center_y=-0.1, radius_x=0.3, radius_y=0.1, intensity=2),
                EllipseParameters(center_x=-0.2, center_y=0.2, radius_x=0.18, radius_y=0.25, intensity=4),
            ]
        else:
            self.ellipses = list(ellipses)

    def kspace(self, ky: torch.Tensor, kx: torch.Tensor) -> torch.Tensor:
        """Create 2D analytic kspace data based on given k-space locations.

        For a corresponding image with 256 x 256 voxels, the k-space locations should be defined within ``[-128, 127]``.

        The Fourier representation of ellipses can be analytically described by Bessel functions [KOA2007]_.

        Parameters
        ----------
        ky
            k-space locations in ky (phase encoding direction).
        kx
            k-space locations in kx (frequency encoding direction).

        References
        ----------
        .. [KOA2007] Koay C, Sarlls J, Oezarslan E (2007) Three-dimensional analytical magnetic resonance imaging
           phantom in the Fourier domain. MRM 58(2) https://doi.org/10.1002/mrm.21292
        ..
        """
        # kx and ky should be broadcastable
        try:
            shape = torch.broadcast_shapes(kx.shape, ky.shape)
        except RuntimeError:
            raise ValueError(f'shape mismatch between kx {kx.shape} and ky {ky.shape}') from None
        if kx.device != ky.device:
            raise ValueError(f'device mismatch between kx {kx.device} and ky {ky.device}')

        kdata = torch.zeros(shape, dtype=torch.complex64, device=kx.device)
        for ellipse in self.ellipses:
            arg = torch.sqrt((ellipse.radius_x * 2) ** 2 * kx**2 + (ellipse.radius_y * 2) ** 2 * ky**2)
            arg[arg < 1e-6] = 1e-6  # avoid zeros

            cdata = 2 * 2 * ellipse.radius_x * ellipse.radius_y * 0.5 * torch.special.bessel_j1(torch.pi * arg) / arg
            kdata = kdata + (
                torch.exp(-1j * 2 * torch.pi * (ellipse.center_x * kx + ellipse.center_y * ky))
                * cdata
                * ellipse.intensity
            )

        # Scale k-space data by factor sqrt(number of points) to ensure correct scaling after FFT with
        # normalization "ortho". See e.g. https://docs.scipy.org/doc/scipy/tutorial/fft.html
        kdata = kdata * kdata.numel() ** 0.5
        return kdata

    def image_space(self, image_dimensions: SpatialDimension[int]) -> torch.Tensor:
        """Create image representation of phantom.

        Parameters
        ----------
        image_dimensions
            Number of voxels in the image.
            This is a 2D simulation, so the output will be of shape `(1 1 1 image_dimensions.y image_dimensions.x)`.
        """
        # Calculate image representation of phantom
        ny, nx = image_dimensions.y, image_dimensions.x
        ix, iy = torch.meshgrid(
            torch.linspace(-nx // 2, nx // 2 - 1, nx),
            torch.linspace(-ny // 2, ny // 2 - 1, ny),
            indexing='xy',
        )

        idata = torch.zeros((ny, nx), dtype=torch.complex64)
        for ellipse in self.ellipses:
            in_ellipse = (
                (ix / nx - ellipse.center_x) ** 2 / ellipse.radius_x**2
                + (iy / ny - ellipse.center_y) ** 2 / ellipse.radius_y**2
            ) <= 1
            idata += ellipse.intensity * in_ellipse

        return repeat(idata, 'y x->other coils z y x', other=1, coils=1, z=1)

    def kdata(self, trajectory: KTrajectory, encoding_matrix: SpatialDimension[int]) -> KData:
        """Create k-space data for the phantom.

        Parameters
        ----------
        trajectory
            Trajectory.
        encoding_matrix
            Encoding matrix.
        """
        if (trajectory.kz != 0).any():
            raise ValueError('Only 2D k-space data is supported')

        data = self.kspace(trajectory.ky, trajectory.kx)
        header = KHeader(
            recon_fov=SpatialDimension(z=0.01, y=1, x=1),
            encoding_fov=SpatialDimension(z=0.01, y=1, x=1),
            encoding_matrix=encoding_matrix,
            recon_matrix=encoding_matrix,
        )
        kdata = KData(data=data, header=header, traj=trajectory)
        return kdata
