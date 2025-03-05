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

GYROMAGNETIC_RATIO_PROTON = 42.5764 * 1e6
r"""The gyromagnetic ratio :math:`\frac{\gamma}{2\pi}` of 1H in H2O in Hz/T."""

# Conversion functions for units
T = TypeVar('T', float, torch.Tensor, list[float], tuple[float, ...])


def ms_to_s(ms: T) -> T:
    """Convert ms to s."""
    if isinstance(ms, list):
        return [ms_to_s(x) for x in ms]
    if isinstance(ms, tuple):
        return tuple([ms_to_s(x) for x in ms])
    """Convert ms to s."""
    return ms / 1000


def s_to_ms(s: T) -> T:
    """Convert s to ms."""
    if isinstance(s, list):
        return [s_to_ms(x) for x in s]
    if isinstance(s, tuple):
        return tuple([s_to_ms(x) for x in s])
    return s * 1000


def mm_to_m(mm: T) -> T:
    """Convert mm to m."""
    if isinstance(mm, list):
        return [mm_to_m(x) for x in mm]
    if isinstance(mm, tuple):
        return tuple([mm_to_m(x) for x in mm])
    return mm / 1000


def m_to_mm(m: T) -> T:
    """Convert m to mm."""
    if isinstance(m, list):
        return [m_to_mm(x) for x in m]
    if isinstance(m, tuple):
        return tuple([m_to_mm(x) for x in m])
    return m * 1000


def deg_to_rad(deg: T) -> T:
    """Convert degree to radians."""
    if isinstance(deg, torch.Tensor):
        return torch.deg2rad(deg)
    if isinstance(deg, tuple):
        return tuple([deg_to_rad(x) for x in deg])
    if isinstance(deg, list):
        return [deg_to_rad(x) for x in deg]
    return deg / 180.0 * np.pi


def rad_to_deg(rad: T) -> T:
    """Convert radians to degree."""
    if isinstance(rad, torch.Tensor):
        return torch.rad2deg(rad)
    if isinstance(rad, tuple):
        return tuple([rad_to_deg(x) for x in rad])
    if isinstance(rad, list):
        return [rad_to_deg(x) for x in rad]
    return rad * 180.0 / np.pi


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
    if isinstance(lamor_frequency, list):
        return [lamor_frequency_to_magnetic_field(x) for x in lamor_frequency]
    if isinstance(lamor_frequency, tuple):
        return tuple([lamor_frequency_to_magnetic_field(x) for x in lamor_frequency])
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
    if isinstance(magnetic_field_strength, tuple):
        return tuple([magnetic_field_to_lamor_frequency(x) for x in magnetic_field_strength])
    if isinstance(magnetic_field_strength, list):
        return [magnetic_field_to_lamor_frequency(x) for x in magnetic_field_strength]
    return magnetic_field_strength * gyromagnetic_ratio
