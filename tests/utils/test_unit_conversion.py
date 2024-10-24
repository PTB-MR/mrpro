"""Tests of unit conversion."""

import numpy as np
import torch
from mrpro.utils.unit_conversion import (
    deg_to_rad,
    lamor_frequency_to_magnetic_field,
    m_to_mm,
    magnetic_field_to_lamor_frequency,
    mm_to_m,
    ms_to_s,
    rad_to_deg,
    s_to_ms,
)

from tests import RandomGenerator


def test_mm_to_m():
    """Verify mm to m conversion."""
    generator = RandomGenerator(seed=0)
    mm_input = generator.float32_tensor((3, 4, 5))
    torch.testing.assert_close(mm_to_m(mm_input), mm_input / 1000.0)


def test_m_to_mm():
    """Verify m to mm conversion."""
    generator = RandomGenerator(seed=0)
    m_input = generator.float32_tensor((3, 4, 5))
    torch.testing.assert_close(m_to_mm(m_input), m_input * 1000.0)


def test_ms_to_s():
    """Verify ms to s conversion."""
    generator = RandomGenerator(seed=0)
    ms_input = generator.float32_tensor((3, 4, 5))
    torch.testing.assert_close(ms_to_s(ms_input), ms_input / 1000.0)


def test_s_to_ms():
    """Verify s to ms conversion."""
    generator = RandomGenerator(seed=0)
    s_input = generator.float32_tensor((3, 4, 5))
    torch.testing.assert_close(s_to_ms(s_input), s_input * 1000.0)


def test_rad_to_deg_tensor():
    """Verify radians to degree conversion."""
    generator = RandomGenerator(seed=0)
    s_input = generator.float32_tensor((3, 4, 5))
    torch.testing.assert_close(rad_to_deg(s_input), torch.rad2deg(s_input))


def test_deg_to_rad_tensor():
    """Verify degree to radians conversion."""
    generator = RandomGenerator(seed=0)
    s_input = generator.float32_tensor((3, 4, 5))
    torch.testing.assert_close(deg_to_rad(s_input), torch.deg2rad(s_input))


def test_rad_to_deg_float():
    """Verify radians to degree conversion."""
    assert rad_to_deg(np.pi / 2) == 90.0


def test_deg_to_rad_float():
    """Verify degree to radians conversion."""
    assert deg_to_rad(180.0) == np.pi


def test_lamor_frequency_to_magnetic_field():
    """Verify conversion of lamor frequency to magnetic field."""
    proton_gyromagnetic_ratio = 42.58 * 1e6
    proton_lamor_frequency_at_3tesla = 127.74 * 1e6
    assert lamor_frequency_to_magnetic_field(proton_lamor_frequency_at_3tesla, proton_gyromagnetic_ratio) == 3.0


def test_magnetic_field_to_lamor_frequency():
    """Verify conversion of magnetic field to lamor frequency."""
    proton_gyromagnetic_ratio = 42.58 * 1e6
    magnetic_field_strength = 3.0
    assert magnetic_field_to_lamor_frequency(magnetic_field_strength, proton_gyromagnetic_ratio) == 127.74 * 1e6
