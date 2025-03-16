"""Tests for EPG signal models."""

from collections.abc import Sequence

import pytest
import torch
from mrpro.operators.models import CardiacFingerprinting
from mrpro.operators.models.EPG import DelayBlock, EPGSequence, FispBlock, InversionBlock, Parameters, T2PrepBlock
from mrpro.operators.SignalModel import SignalModel
from tests import RandomGenerator
from tests.operators.models.conftest import SHAPE_VARIATIONS_SIGNAL_MODELS


class EpgFispModel(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
    def __init__(
        self,
        flip_angles: float | torch.Tensor = torch.pi,
        rf_phases: float | torch.Tensor = 0,
        tr: float | torch.Tensor = 0.01,
        te: float | torch.Tensor = 0.005,
        n_states: int = 10,
    ):
        super().__init__()
        self.sequence = EPGSequence()
        self.sequence.append(FispBlock(flip_angles, rf_phases, tr, te))
        self.n_states = n_states

    def forward(
        self, m0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, b1_relative: torch.Tensor
    ) -> tuple[torch.Tensor]:
        """Simulate the Cardiac MR Fingerprinting signal.

        Parameters
        ----------
        m0
            Steady state magnetization (complex)
        t1
            longitudinal relaxation time T1
        t2
            transversal relaxation time T2
        b1_relative
            relative B1 scaling (complex)


        Returns
        -------
            Signal of Fisp sequence.
        """
        parameters = Parameters(m0, t1, t2, b1_relative)
        _, signals = self.sequence(parameters, n_states=self.n_states)
        signal = torch.stack(list(signals), dim=0)
        return (signal,)


def test_EpgFisp_not_enough_states() -> None:
    """Verify error for less than 2 states."""
    epg_model = EpgFispModel(n_states=1)
    with pytest.raises(ValueError, match='Number of states should be at least 2'):
        epg_model(torch.ones((1,)), torch.ones((1,)), torch.ones((1,)), torch.ones((1,)))


def test_EpgFisp_parameter_broadcasting() -> None:
    """Verify correct broadcasting of values."""
    te = tr = rf_phases = torch.ones((1,))
    flip_angles = torch.ones((20,))
    epg_model = EpgFispModel(flip_angles=flip_angles, rf_phases=rf_phases, te=te, tr=tr)
    m0 = t1 = t2 = b1_relative = torch.randn((30,))
    (epg_signal,) = epg_model(m0, t1, t2, b1_relative)
    assert epg_signal.shape == (20, 30)


def test_EpgFisp_parameter_mismatch() -> None:
    """Verify error for shape mismatch."""
    flip_angles = rf_phases = tr = torch.ones((1, 2))
    te = torch.ones((1, 3))
    with pytest.raises(ValueError, match='Shapes of flip_angles'):
        EpgFispModel(flip_angles=flip_angles, rf_phases=rf_phases, te=te, tr=tr)


@SHAPE_VARIATIONS_SIGNAL_MODELS
def test_EpgFisp_shape(parameter_shape, contrast_dim_shape, signal_shape) -> None:
    """Test correct signal shapes."""
    rng = RandomGenerator(0)
    flip_angles = rng.float32_tensor(contrast_dim_shape, low=1e-5, high=5)
    rf_phases = rng.float32_tensor(contrast_dim_shape, low=1e-5, high=0.5)
    te = rng.float32_tensor(contrast_dim_shape, low=1e-5, high=0.01)
    tr = rng.float32_tensor(contrast_dim_shape, low=1e-5, high=0.05)
    t1 = rng.float32_tensor(parameter_shape, low=1e-5, high=5)
    t2 = rng.float32_tensor(parameter_shape, low=1e-5, high=0.5)
    m0 = rng.complex64_tensor(parameter_shape)
    relative_b1 = rng.complex64_tensor(parameter_shape)

    model_op = EpgFispModel(flip_angles=flip_angles, rf_phases=rf_phases, te=te, tr=tr)
    (signal,) = model_op(m0, t1, t2, relative_b1)
    assert signal.shape == signal_shape


def test_EpgFisp_inversion_recovery() -> None:
    """EPG simulation of single-line inversion recovery sequence.

    Obtaining a single point at different times after an inversion pulse follows a mono-exponential model.
    """
    t1 = torch.as_tensor([100, 200, 300, 400, 500, 1000, 2000, 4000]) * 1e-3
    t2 = torch.as_tensor([20, 80, 160, 320, 20, 80, 160, 320]) * 1e-3
    m0 = torch.ones_like(t1, dtype=torch.complex64)

    # inversion times
    ti_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5]

    # analytical signal
    analytical_signal = m0 * (1 - 2 * torch.exp(-(torch.as_tensor(ti_list)[..., None] / t1)))

    # single readout per inversion block with 90° pulse and very short echo time to avoid T2 effects
    sequence = EPGSequence()
    for ti in ti_list:
        sequence.append(InversionBlock(inversion_time=ti))
        sequence.append(FispBlock(flip_angles=torch.pi / 2, rf_phases=torch.pi / 2, tr=0.007, te=1e-6))
        sequence.append(DelayBlock(delay_time=40))
    parameters = Parameters(m0, t1, t2)
    _, signals = sequence(parameters)
    epg_signal = torch.stack(list(signals), dim=0)

    torch.testing.assert_close(epg_signal, analytical_signal, rtol=1e-3, atol=1e-3)


def test_EpgFisp_t2_preparation() -> None:
    """EPG simulation of single-line T2-prep sequence.

    Obtaining a single point at different TEs of a T2-prep pulse follows a mono-exponential model.
    """
    t1 = torch.as_tensor([100, 200, 300, 400, 500, 1000, 2000, 4000]) * 1e-3
    t2 = torch.as_tensor([20, 80, 160, 320, 20, 80, 160, 320]) * 1e-3
    m0 = torch.ones_like(t1, dtype=torch.complex64)

    # echo times
    t2_prep_te_times = [0, 0.02, 0.04, 0.08, 0.2]

    # analytical signal
    analytical_signal = m0 * torch.exp(-(torch.as_tensor(t2_prep_te_times)[:, None] / t2))

    # single readout per T2-prep block with 90° pulse and very short echo time to avoid T2 effects during acquisition
    sequence = EPGSequence()
    for te in t2_prep_te_times:
        sequence.append(T2PrepBlock(te=te))
        sequence.append(FispBlock(flip_angles=torch.pi / 2, rf_phases=torch.pi / 2, tr=0.007, te=1e-6))
        sequence.append(DelayBlock(delay_time=40))
    parameters = Parameters(m0, t1, t2)
    _, signals = sequence(parameters)
    epg_signal = torch.stack(list(signals), dim=0)

    torch.testing.assert_close(epg_signal, analytical_signal, rtol=1e-3, atol=1e-3)


def test_cmrf_model(parameter_shape: Sequence[int] = (2, 5, 10, 10, 10)) -> None:
    """Test the CMRF model."""
    acquisition_times = torch.linspace(0, 10, 705)
    cmrf_model = CardiacFingerprinting(acquisition_times=acquisition_times, echo_time=0.05)
    rng = RandomGenerator(0)
    t1 = rng.float32_tensor(parameter_shape, low=1e-5, high=5)
    t2 = rng.float32_tensor(parameter_shape, low=1e-5, high=0.5)
    m0 = rng.complex64_tensor(parameter_shape)
    signal = cmrf_model(t1, t2, m0)
    assert signal
