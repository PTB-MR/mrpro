"""Slice Profiles."""

import abc
from collections.abc import Sequence
from math import log

import numpy as np
import torch

from mrpro.utils.TensorAttributeMixin import TensorAttributeMixin
from mrpro.utils.unit_conversion import GYROMAGNETIC_RATIO_PROTON

__all__ = [
    'GaussianRFPulse',
    'SincRFPulse',
    'SliceGaussian',
    'SliceInterpolate',
    'SliceProfileBase',
    'SliceRFPulseBase',
    'SliceSmoothedRectangular',
]


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


class SliceRFPulseBase(abc.ABC, TensorAttributeMixin, torch.nn.Module):
    """Base class for slice-selective RF pulse templates."""

    @abc.abstractmethod
    def forward(
        self, flip_angle: torch.Tensor | float, duration: torch.Tensor | float, dt: torch.Tensor | float
    ) -> torch.Tensor:
        """Create a discrete RF waveform in Tesla."""
        raise NotImplementedError

    def rf_and_phase(
        self,
        flip_angle: torch.Tensor | float,
        duration: torch.Tensor | float,
        dt: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create a discrete RF waveform and phase in rad."""
        rf = self(flip_angle=flip_angle, duration=duration, dt=dt)
        return rf, torch.zeros_like(rf)


def _n_samples(duration: torch.Tensor | float, dt: torch.Tensor | float) -> int:
    duration_value = torch.as_tensor(duration).item()
    dt_value = torch.as_tensor(dt).item()
    if duration_value <= 0 or dt_value <= 0:
        raise ValueError('duration and dt must be positive.')
    samples = round(duration_value / dt_value)
    if samples < 1:
        raise ValueError('duration / dt must produce at least one RF sample.')
    return samples


def _scale_waveform_to_flip_angle(
    waveform: torch.Tensor,
    flip_angle: torch.Tensor | float,
    dt: torch.Tensor | float,
) -> torch.Tensor:
    flip_angle = torch.as_tensor(flip_angle, dtype=waveform.dtype, device=waveform.device)
    dt = torch.as_tensor(dt, dtype=waveform.dtype, device=waveform.device)
    return waveform * (flip_angle / (GYROMAGNETIC_RATIO_PROTON * dt * waveform.sum()))


class GaussianRFPulse(SliceRFPulseBase):
    """Gaussian RF pulse template."""

    fwhm_fraction: torch.Tensor

    def __init__(self, fwhm_fraction: float | torch.Tensor = 0.35):
        """Initialize the Gaussian pulse template.

        Parameters
        ----------
        fwhm_fraction
            RF Gaussian FWHM relative to pulse duration.
        """
        super().__init__()
        self.register_buffer('fwhm_fraction', torch.as_tensor(fwhm_fraction))

    def forward(
        self, flip_angle: torch.Tensor | float, duration: torch.Tensor | float, dt: torch.Tensor | float
    ) -> torch.Tensor:
        """Create a Gaussian RF waveform in Tesla."""
        samples = _n_samples(duration, dt)
        duration = torch.as_tensor(duration)
        time = torch.linspace(-0.5, 0.5, samples, dtype=duration.dtype, device=duration.device)
        sigma = self.fwhm_fraction / (2 * (2 * log(2)) ** 0.5)
        waveform = torch.exp(-0.5 * (time / sigma) ** 2)
        return _scale_waveform_to_flip_angle(waveform, flip_angle, dt)


class SincRFPulse(SliceRFPulseBase):
    """Apodized sinc RF pulse template."""

    time_bandwidth: torch.Tensor
    apodization: torch.Tensor

    def __init__(self, time_bandwidth: float | torch.Tensor = 4.0, apodization: float | torch.Tensor = 0.5):
        """Initialize the sinc pulse template.

        Parameters
        ----------
        time_bandwidth
            Time-bandwidth product of the sinc pulse.
        apodization
            Raised-cosine apodization in ``[0, 1]``.
        """
        super().__init__()
        self.register_buffer('time_bandwidth', torch.as_tensor(time_bandwidth))
        self.register_buffer('apodization', torch.as_tensor(apodization))

    def forward(
        self, flip_angle: torch.Tensor | float, duration: torch.Tensor | float, dt: torch.Tensor | float
    ) -> torch.Tensor:
        """Create an apodized sinc RF waveform in Tesla."""
        samples = _n_samples(duration, dt)
        duration = torch.as_tensor(duration)
        time = torch.linspace(-0.5, 0.5, samples, dtype=duration.dtype, device=duration.device)
        sinc = torch.sinc(self.time_bandwidth.to(time) * time)
        window = 1 - self.apodization.to(time) + self.apodization.to(time) * torch.cos(2 * torch.pi * time)
        waveform = sinc * window
        return _scale_waveform_to_flip_angle(waveform, flip_angle, dt)


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
        self.fwhm_rect: torch.Tensor
        self.fwhm_gauss: torch.Tensor

        self.register_buffer('fwhm_rect', torch.as_tensor(fwhm_rect))
        self.register_buffer('fwhm_gauss', torch.as_tensor(fwhm_gauss))

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
