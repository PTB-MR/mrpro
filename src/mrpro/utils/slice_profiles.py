"""Slice Profiles."""

import abc
from collections.abc import Sequence
from math import log

import numpy as np
import torch

from mrpro.utils.TensorAttributeMixin import TensorAttributeMixin

__all__ = ['SliceGaussian', 'SliceInterpolate', 'SliceProfileBase', 'SliceSmoothedRectangular']


class SliceProfileBase(abc.ABC, TensorAttributeMixin, torch.nn.Module):
    """Base class for slice profiles."""

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the slice profile at a position."""
        raise NotImplementedError

    def random_sample(self, size: Sequence[int]) -> torch.Tensor:
        """Sample `n` random positions from the profile.

        Use the profile as a probability density function to sample positions.

        Parameters
        ----------
        size
            Number of positions to sample

        Returns
        -------
            Sampled positions, shape will be size.
        """
        raise NotImplementedError


class SliceGaussian(SliceProfileBase):
    """Gaussian slice profile."""

    fwhm: torch.Tensor

    def __init__(self, fwhm: float | torch.Tensor):
        """Initialize the Gaussian slice profile.

        Parameters
        ----------
        fwhm
            Full width at half maximum of the Gaussian
        """
        super().__init__()
        self.fwhm = torch.as_tensor(fwhm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the Gaussian slice profile at a position.

        Parameters
        ----------
        x
            Position at which to evaluate the profile

        Returns
        -------
            Value of the profile / intensity at the given position
        """
        return torch.exp(-(x**2) / (0.36 * self.fwhm**2))


class SliceSmoothedRectangular(SliceProfileBase):
    """Rectangular slice profile with smoothed flanks.

    Implemented as a convolution of a rectangular profile
    with a Gaussian.
    """

    def __init__(self, fwhm_rect: float | torch.Tensor, fwhm_gauss: float | torch.Tensor):
        """Initialize the Rectangular slice profile.

        Parameters
        ----------
        fwhm_rect
            Full width at half maximum of the rectangular profile
        fwhm_gauss
            Full width at half maximum of the Gaussian profile.
            Set to zero to disable smoothing.

        Returns
        -------
            Value of the profile / intensity at the given position
        """
        super().__init__()
        self.fwhm_rect = torch.nn.Buffer(torch.as_tensor(fwhm_rect))
        self.fwhm_gauss = torch.nn.Buffer(torch.as_tensor(fwhm_gauss))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the Gaussian slice profile at a position.

        Parameters
        ----------
        x
            Position at which to evaluate the profile

        Returns
        -------
            Value of the profile / intensity at the given position
        """
        scaled = x * 2 / self.fwhm_rect
        if self.fwhm_gauss > 0 and self.fwhm_rect > 0:
            n = (log(2) ** 0.5) * self.fwhm_rect / self.fwhm_gauss
            norm = 1 / (2 * torch.erf(n))
            return (torch.erf(n * (1 - scaled)) + torch.erf(n * (1 + scaled))) * norm
        elif self.fwhm_rect > 0:
            return (scaled.abs() <= 1).float()
        elif self.fwhm_gauss > 0:
            return torch.exp(-4 * log(2) * (x / self.fwhm_gauss) ** 2)
        else:
            raise ValueError('At least one of the widths has to be greater zero.')


class SliceInterpolate(SliceProfileBase):
    """slice profile based on interpolation of measured profile."""

    def __init__(self, positions: torch.Tensor, values: torch.Tensor):
        """Initialize the interpolated slice profile.

        Parameters
        ----------
        positions
            Positions of the measured profile.
        values
            Intensities of the measured profile.
        """
        super().__init__()
        self._xs = positions.detach().cpu().float().numpy()
        self._weights = values.detach().cpu().float().numpy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the interpolated slice profile at a position.

        Parameters
        ----------
        x
            Position at which to evaluate the profile.

        Returns
        -------
            Value of the profile / intensity at the given position.
        """
        if x.requires_grad:
            raise NotImplementedError('Interpolated profile does not support gradients.')
        x_np = x.detach().cpu().numpy()
        y_np = torch.as_tensor(np.interp(x_np, self._xs, self._weights, 0, 0))
        return y_np.to(x.device)
