"""Conversion between different units."""

from typing import TypeVar

import numpy as np
import torch

__all__ = [
    'GYROMAGNETIC_RATIO_PROTON',
    'deg_to_rad',
    'lamor_frequency_to_magnetic_field',
    'm_to_mm',
    'magnetic_field_to_lamor_frequency',
    'mm_to_m',
    'ms_to_s',
    'rad_to_deg',
    's_to_ms',
]

GYROMAGNETIC_RATIO_PROTON = 42.58 * 1e6
r"""The gyromagnetic ratio :math:`\frac{\gamma}{2\pi}` of 1H in H20 in Hz/T"""

# Conversion functions for units
T = TypeVar('T', float, torch.Tensor)


def ms_to_s(ms: T) -> T:
    """Convert ms to s."""
    return ms / 1000


def s_to_ms(s: T) -> T:
    """Convert s to ms."""
    return s * 1000


def mm_to_m(mm: T) -> T:
    """Convert mm to m."""
    return mm / 1000


def m_to_mm(m: T) -> T:
    """Convert m to mm."""
    return m * 1000


def deg_to_rad(deg: T) -> T:
    """Convert degree to radians."""
    if isinstance(deg, torch.Tensor):
        return torch.deg2rad(deg)
    return deg / 180.0 * np.pi


def rad_to_deg(deg: T) -> T:
    """Convert radians to degree."""
    if isinstance(deg, torch.Tensor):
        return torch.rad2deg(deg)
    return deg * 180.0 / np.pi


def lamor_frequency_to_magnetic_field(lamor_frequency: T, gyromagnetic_ratio: float = GYROMAGNETIC_RATIO_PROTON) -> T:
    """Convert the Lamor frequency [Hz] to the magntic field strength [T].

    Parameters
    ----------
    lamor_frequency
        Lamor frequency [Hz]
    gyromagnetic_ratio
        Gyromagnetic ratio [Hz/T], default: gyromagnetic ratio of 1H proton

    Returns
    -------
    Magnetic field strength [T]
    """
    return lamor_frequency / gyromagnetic_ratio


def magnetic_field_to_lamor_frequency(
    magnetic_field_strength: T, gyromagnetic_ratio: float = GYROMAGNETIC_RATIO_PROTON
) -> T:
    """Convert the magntic field strength [T] to Lamor frequency [Hz].

    Parameters
    ----------
    magnetic_field_strength
       Strength of the magnetic field [T]
    gyromagnetic_ratio
        Gyromagnetic ratio [Hz/T], default: gyromagnetic ratio of 1H proton

    Returns
    -------
    Lamor frequency [Hz]
    """
    return magnetic_field_strength * gyromagnetic_ratio
