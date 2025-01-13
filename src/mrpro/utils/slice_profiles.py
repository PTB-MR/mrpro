"""Slice Profiles."""

import abc
from collections.abc import Sequence
from math import log

import numpy as np
import torch
from torch import Tensor

__all__ = ['SliceGaussian', 'SliceInterpolate', 'SliceProfileBase', 'SliceSmoothedRectangular']


class SliceProfileBase(abc.ABC, torch.nn.Module):
    """Base class for slice profiles."""

    @abc.abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the slice profile at a position x."""
        raise NotImplementedError

    def random_sample(self, size: Sequence[int]) -> Tensor:
        """Sample n random positions from the profile.

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
    """Gaussian Slice Profile."""

    def __init__(self, fwhm: float | Tensor):
        """Initialize the Gaussian Slice Profile.

        Parameters
        ----------
        fwhm
            Full width at half maximum of the Gaussian
        """
        super().__init__()
        self.register_buffer('fwhm', torch.as_tensor(fwhm))

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the Gaussian Slice Profile at a position.

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
    """Rectangular Slice Profile with smoothed flanks.

    Implemented as a convolution of a rectangular profile
    with a Gaussian.
    """

    def __init__(self, fwhm_rect: float | Tensor, fwhm_gauss: float | Tensor):
        """Initialize the Rectangular Slice Profile.

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
        self.register_buffer('fwhm_rect', torch.as_tensor(fwhm_rect))
        self.register_buffer('fwhm_gauss', torch.as_tensor(fwhm_gauss))

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the Gaussian Slice Profile at a position.

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
    """Slice Profile based on Interpolation of Measured Profile."""

    def __init__(self, positions: Tensor, values: Tensor):
        """Initialize the Interpolated Slice Profile.

        Parameters
        ----------
        positions
            Positions of the measured profile
        values
            Intensities of the measured profile
        """
        super().__init__()
        self._xs = positions.detach().cpu().float().numpy()
        self._weights = values.detach().cpu().float().numpy()

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the Interpolated Slice Profile at a position.

        Parameters
        ----------
        x
            Position at which to evaluate the profile

        Returns
        -------
            Value of the profile / intensity at the given position
        """
        if x.requires_grad:
            raise NotImplementedError('Interpolated profile does not support gradients.')
        x_np = x.detach().cpu().numpy()
        y_np = torch.as_tensor(np.interp(x_np, self._xs, self._weights, 0, 0))
        return y_np.to(x.device)
