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


def random_tensor_list_tuple() -> tuple[torch.Tensor, tuple, list]:
    generator = RandomGenerator(9)
    return (generator.float32_tensor((3, 4, 5)), generator.float32_tuple(7), list(generator.float32_tuple(6)))


@pytest.mark.parametrize('data', random_tensor_list_tuple())
def test_mm_to_m(data) -> None:
    """Verify mm to m conversion."""
    rng = RandomGenerator(seed=0)
    mm_input = rng.float32_tensor((3, 4, 5))
    torch.testing.assert_close(mm_to_m(mm_input), mm_input / 1000.0)


@pytest.mark.parametrize('data', random_tensor_list_tuple())
def test_m_to_mm(data) -> None:
    """Verify m to mm conversion."""
    rng = RandomGenerator(seed=0)
    m_input = rng.float32_tensor((3, 4, 5))
    torch.testing.assert_close(m_to_mm(m_input), m_input * 1000.0)


def test_micrometer_to_m():
    """Verify micrometer to m conversion."""
    rng = RandomGenerator(seed=0)
    micrometer_input = rng.float32_tensor((3, 4, 5))
    torch.testing.assert_close(micrometer_to_m(micrometer_input), micrometer_input / 1.0e6)


def test_m_to_micrometer():
    """Verify m to micrometer conversion."""
    rng = RandomGenerator(seed=0)
    m_input = rng.float32_tensor((3, 4, 5))
    torch.testing.assert_close(m_to_micrometer(m_input), m_input * 1.0e6)


def test_ms_to_s():
    """Verify ms to s conversion."""
    rng = RandomGenerator(seed=0)
    ms_input = rng.float32_tensor((3, 4, 5))
    torch.testing.assert_close(ms_to_s(ms_input), ms_input / 1000.0)


@pytest.mark.parametrize('data', random_tensor_list_tuple())
def test_s_to_ms(data) -> None:
    """Verify s to ms conversion."""
    rng = RandomGenerator(seed=0)
    s_input = rng.float32_tensor((3, 4, 5))
    torch.testing.assert_close(s_to_ms(s_input), s_input * 1000.0)


@pytest.mark.parametrize('data', random_tensor_list_tuple())
def test_rad_to_deg_tensor(data) -> None:
    """Verify radians to degree conversion."""
    rng = RandomGenerator(seed=0)
    s_input = rng.float32_tensor((3, 4, 5))
    torch.testing.assert_close(rad_to_deg(s_input), torch.rad2deg(s_input))


@pytest.mark.parametrize('data', random_tensor_list_tuple())
def test_deg_to_rad_tensor(data) -> None:
    """Verify degree to radians conversion."""
    rng = RandomGenerator(seed=0)
    s_input = rng.float32_tensor((3, 4, 5))
    torch.testing.assert_close(deg_to_rad(s_input), torch.deg2rad(s_input))


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
