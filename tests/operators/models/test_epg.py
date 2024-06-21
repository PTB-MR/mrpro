"""Tests for EPG signal models."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import pytest
import torch
from mrpro.operators.models import EpgMrfFisp
from mrpro.operators.models import EpgTse


def test_EpgMrfFisp_parameter_broadcasting():
    """Verify correct broadcasting of values."""
    te = tr = rf_phases = torch.ones((1,))
    flip_angles = torch.ones((20,))
    epg_mrf_fisp = EpgMrfFisp(flip_angles=flip_angles, rf_phases=rf_phases, te=te, tr=tr)
    m0 = t1 = t2 = torch.randn((30,))
    (epg_signal,) = epg_mrf_fisp.forward(m0, t1, t2)
    assert epg_signal.shape == (20, 30)


def test_EpgMrfFisp_parameter_mismatch():
    """Verify error for shape mismatch."""
    flip_angles = rf_phases = tr = torch.ones((1, 2))
    te = torch.ones((1, 3))
    with pytest.raises(ValueError, match='Shapes of flip_angles'):
        EpgMrfFisp(flip_angles=flip_angles, rf_phases=rf_phases, te=te, tr=tr)


def test_EpgTse_parameter_broadcasting():
    """Verify correct broadcasting of values."""
    te = rf_phases = torch.ones((1,))
    flip_angles = torch.ones((20,))
    epg_mrf_fisp = EpgTse(flip_angles=flip_angles, rf_phases=rf_phases, te=te)
    m0 = t1 = t2 = b1 = torch.randn((30,))
    (epg_signal,) = epg_mrf_fisp.forward(m0, t1, t2, b1)
    assert epg_signal.shape == (20, 30)


def test_EpgTse_multi_echo_train():
    """Verify correct shape for multi echo trains."""
    flip_angles = te = rf_phases = torch.ones((20,))
    tr = torch.ones((3,))
    epg_mrf_fisp = EpgTse(flip_angles=flip_angles, rf_phases=rf_phases, te=te, tr=tr)
    m0 = t1 = t2 = b1 = torch.randn((30,))
    (epg_signal,) = epg_mrf_fisp.forward(m0, t1, t2, b1)
    assert epg_signal.shape == (20 * 3, 30)


def test_EpgTse_parameter_mismatch():
    """Verify error for shape mismatch."""
    flip_angles = rf_phases = torch.ones((1, 2))
    te = torch.ones((1, 3))
    with pytest.raises(ValueError, match='Shapes of flip_angles'):
        EpgTse(flip_angles=flip_angles, rf_phases=rf_phases, te=te)
