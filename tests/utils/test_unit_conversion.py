"""Tests of unit conversion."""

import numpy as np
import pytest
import torch
from mrpro.utils import RandomGenerator
from mrpro.utils.unit_conversion import (
    deg_to_rad,
    lamor_frequency_to_magnetic_field,
    m_to_micrometer,
    m_to_mm,
    magnetic_field_to_lamor_frequency,
    micrometer_to_m,
    mm_to_m,
    ms_to_s,
    rad_to_deg,
    s_to_ms,
)

# (function, expected result as a function of the input tensor)
CONVERSIONS = pytest.mark.parametrize(
    ('function', 'reference'),
    [
        (mm_to_m, lambda x: x / 1000),
        (m_to_mm, lambda x: x * 1000),
        (micrometer_to_m, lambda x: x / 1.0e6),
        (m_to_micrometer, lambda x: x * 1.0e6),
        (ms_to_s, lambda x: x / 1000),
        (s_to_ms, lambda x: x * 1000),
        (rad_to_deg, torch.rad2deg),
        (deg_to_rad, torch.deg2rad),
    ],
    ids=['mm_to_m', 'm_to_mm', 'micrometer_to_m', 'm_to_micrometer', 'ms_to_s', 's_to_ms', 'rad_to_deg', 'deg_to_rad'],
)


@CONVERSIONS
def test_unit_conversion_tensor(function, reference) -> None:
    """Verify conversion of a tensor."""
    x = RandomGenerator(9).float32_tensor((3, 4, 5))
    torch.testing.assert_close(function(x), reference(x))


@CONVERSIONS
def test_unit_conversion_list(function, reference) -> None:
    """Verify conversion of a list."""
    x = RandomGenerator(9).float32_tensor((7,))
    result = function(list(x))
    torch.testing.assert_close(torch.as_tensor(result), reference(x))


@CONVERSIONS
def test_unit_conversion_tuple(function, reference) -> None:
    """Verify conversion of a tuple."""
    x = RandomGenerator(9).float32_tensor((7,))
    result = function(tuple(x))
    torch.testing.assert_close(torch.as_tensor(result), reference(x))


def test_rad_to_deg_float() -> None:
    """Verify radians to degree conversion."""
    assert rad_to_deg(np.pi / 2) == 90.0


def test_deg_to_rad_float() -> None:
    """Verify degree to radians conversion."""
    assert deg_to_rad(180.0) == np.pi


def test_lamor_frequency_to_magnetic_field() -> None:
    """Verify conversion of lamor frequency to magnetic field."""
    proton_gyromagnetic_ratio = 42.58 * 1e6
    proton_lamor_frequency_at_3tesla = 127.74 * 1e6
    assert lamor_frequency_to_magnetic_field(proton_lamor_frequency_at_3tesla, proton_gyromagnetic_ratio) == 3.0


def test_magnetic_field_to_lamor_frequency() -> None:
    """Verify conversion of magnetic field to lamor frequency."""
    proton_gyromagnetic_ratio = 42.58 * 1e6
    magnetic_field_strength = 3.0
    assert magnetic_field_to_lamor_frequency(magnetic_field_strength, proton_gyromagnetic_ratio) == 127.74 * 1e6
