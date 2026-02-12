"""Tests for EPG signal models."""

from collections.abc import Sequence
from typing import Literal

import pytest
import torch
from mr2.operators.models.EPG import (
    AcquisitionBlock,
    DelayBlock,
    EPGSequence,
    FispBlock,
    GradientDephasingBlock,
    InversionBlock,
    Parameters,
    RFBlock,
    T1RhoPrepBlock,
    T2PrepBlock,
    TseBlock,
    initial_state,
)
from mr2.operators.SignalModel import SignalModel
from mr2.utils import RandomGenerator
from tests.operators.models.conftest import SHAPE_VARIATIONS_SIGNAL_MODELS


class BasicEpgModel(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
    """An EPG model which covers all basic EPG blocks for testing.

    A "basic block" is a block which carries out a single operation on the EPG states.
    """

    def __init__(
        self,
        n_states: int = 10,
        device: Literal['cpu', 'cuda'] = 'cpu',
    ):
        super().__init__()
        self.sequence = EPGSequence()
        self.sequence.append(AcquisitionBlock())
        self.sequence.append(DelayBlock(delay_time=torch.tensor(0.01, device=device)))
        self.sequence.append(GradientDephasingBlock())
        self.sequence.append(InversionBlock(inversion_time=torch.tensor(0.02, device=device)))
        self.sequence.append(
            RFBlock(flip_angle=torch.tensor(torch.pi, device=device), phase=torch.tensor(0, device=device))
        )
        self.sequence.append(T1RhoPrepBlock(spin_lock_duration=torch.tensor(0.04, device=device)))
        self.sequence.append(T2PrepBlock(te=torch.tensor(0.1, device=device)))
        self.n_states = n_states

    def forward(
        self, m0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, b1_relative: torch.Tensor, t1_rho: torch.Tensor
    ) -> tuple[torch.Tensor]:
        """Simulate the signal.

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
        t1_rho
            T1 rho relaxation time

        Returns
        -------
            Signal of sequence.
        """
        parameters = Parameters(m0, t1, t2, b1_relative, t1_rho)
        _, signals = self.sequence(parameters, states=self.n_states)
        signal = torch.stack(list(signals), dim=0)
        return (signal,)


class EpgFispModel(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
    """A simple EPG model of a Fisp sequence for testing."""

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
        self.sequence.append(FispBlock(flip_angles, rf_phases, te, tr))
        self.n_states = n_states

    def forward(
        self, m0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, b1_relative: torch.Tensor
    ) -> tuple[torch.Tensor]:
        """Simulate the signal.

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
        _, signals = self.sequence(parameters, states=self.n_states)
        signal = torch.stack(list(signals), dim=0)
        return (signal,)


class EpgTseModel(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
    """A simple EPG model of a Tse sequence for testing."""

    def __init__(
        self,
        refocusing_flip_angles: float | torch.Tensor = torch.pi,
        refocusing_rf_phases: float | torch.Tensor = 0,
        te: float = 0.005,
        n_states: int = 10,
    ):
        super().__init__()
        self.sequence = EPGSequence()
        self.sequence.append(TseBlock(refocusing_flip_angles, refocusing_rf_phases, te))
        self.n_states = n_states

    def forward(
        self, m0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, b1_relative: torch.Tensor
    ) -> tuple[torch.Tensor]:
        """Simulate the signal.

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
            Signal of Tse sequence.
        """
        parameters = Parameters(m0, t1, t2, b1_relative)
        _, signals = self.sequence(parameters, states=self.n_states)
        signal = torch.stack(list(signals), dim=0)
        return (signal,)


@pytest.mark.cuda
def test_BasicEpgModel_cuda(parameter_shape: Sequence[int] = (2,)) -> None:
    """Test basic EPG blocks work on cuda devices."""
    rng = RandomGenerator(8)
    t1 = rng.float32_tensor(parameter_shape, low=1e-5, high=5)
    t2 = rng.float32_tensor(parameter_shape, low=1e-5, high=0.5)
    m0 = rng.complex64_tensor(parameter_shape)
    relative_b1 = rng.complex64_tensor(parameter_shape)
    t1_rho = rng.float32_tensor(parameter_shape, low=1e-5, high=0.5)

    # Create on CPU, transfer to GPU and run on GPU
    model = BasicEpgModel()
    model.cuda()
    (signal,) = model(m0.cuda(), t1.cuda(), t2.cuda(), relative_b1.cuda(), t1_rho.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Create on GPU and run on GPU
    model = BasicEpgModel(device='cuda')
    (signal,) = model(m0.cuda(), t1.cuda(), t2.cuda(), relative_b1.cuda(), t1_rho.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Create on GPU, transfer to CPU and run on CPU
    model = BasicEpgModel(device='cuda')
    model.cpu()
    (signal,) = model(m0, t1, t2, relative_b1, t1_rho)
    assert signal.is_cpu
    assert signal.isfinite().all()


def test_initial_state_not_enough_states() -> None:
    """Verify error for less than 2 states."""
    with pytest.raises(ValueError, match='Number of states should be at least 2'):
        initial_state(shape=(1, 2), n_states=1)


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


def test_EpgFisp_tr_te() -> None:
    """Verify error for tr shorter than te."""
    flip_angles = rf_phases = tr = torch.ones((1, 2))
    with pytest.raises(ValueError, match='should be smaller than repetition time'):
        EpgFispModel(flip_angles=flip_angles, rf_phases=rf_phases, te=tr * 2, tr=tr)


def test_EpgFisp_neg_te() -> None:
    """Verify error for negative te."""
    flip_angles = rf_phases = tr = torch.ones((1, 2))
    with pytest.raises(ValueError, match='Negative echo time'):
        EpgFispModel(flip_angles=flip_angles, rf_phases=rf_phases, te=-tr, tr=tr)


@SHAPE_VARIATIONS_SIGNAL_MODELS
def test_EpgFisp_shape(parameter_shape, contrast_dim_shape, signal_shape) -> None:
    """Test correct signal shapes."""
    rng = RandomGenerator(0)
    flip_angles = rng.float32_tensor(contrast_dim_shape, low=1e-5, high=5)
    rf_phases = rng.float32_tensor(contrast_dim_shape, low=1e-5, high=0.5)
    te = rng.float32_tensor(contrast_dim_shape, low=1e-5, high=0.01)
    tr = rng.float32_tensor(contrast_dim_shape, low=0.01, high=0.05)
    t1 = rng.float32_tensor(parameter_shape, low=1e-5, high=5)
    t2 = rng.float32_tensor(parameter_shape, low=1e-5, high=0.5)
    m0 = rng.complex64_tensor(parameter_shape)
    relative_b1 = rng.complex64_tensor(parameter_shape)

    model_op = EpgFispModel(flip_angles=flip_angles, rf_phases=rf_phases, te=te, tr=tr)
    (signal,) = model_op(m0, t1, t2, relative_b1)
    assert signal.shape == signal_shape


@pytest.mark.cuda
def test_EpgFisp_cuda(parameter_shape: Sequence[int] = (2,)) -> None:
    """Test Fisp model works on cuda devices."""
    rng = RandomGenerator(8)
    flip_angles = rng.float32_tensor(9, low=1e-5, high=5)
    rf_phases = rng.float32_tensor(9, low=1e-5, high=0.5)
    te = rng.float32_tensor(9, low=1e-5, high=0.01)
    tr = rng.float32_tensor(9, low=0.01, high=0.05)
    t1 = rng.float32_tensor(parameter_shape, low=1e-5, high=5)
    t2 = rng.float32_tensor(parameter_shape, low=1e-5, high=0.5)
    m0 = rng.complex64_tensor(parameter_shape)
    relative_b1 = rng.complex64_tensor(parameter_shape)

    # Create on CPU, transfer to GPU and run on GPU
    model = EpgFispModel(flip_angles, rf_phases, tr, te)
    model.cuda()
    (signal,) = model(m0.cuda(), t1.cuda(), t2.cuda(), relative_b1.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Create on GPU and run on GPU
    model = EpgFispModel(flip_angles.cuda(), rf_phases.cuda(), tr, te.cuda())
    (signal,) = model(m0.cuda(), t1.cuda(), t2.cuda(), relative_b1.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Create on GPU, transfer to CPU and run on CPU
    model = EpgFispModel(flip_angles.cuda(), rf_phases.cuda(), tr.cuda(), te)
    model.cpu()
    (signal,) = model(m0, t1, t2, relative_b1)
    assert signal.is_cpu
    assert signal.isfinite().all()


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


def test_epg_tse_mono_exponential_decay():
    """Echo trains should follow monoexpontial model for long TR."""
    t1 = torch.as_tensor([0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 4.00])
    t2 = torch.as_tensor([0.02, 0.08, 0.16, 0.32, 0.02, 0.08, 0.16, 0.32])
    m0 = torch.ones_like(t1, dtype=torch.complex64)

    n_echoes = 10
    te = 0.02

    # analytical signal
    # cumsum because te is the time between refocusing pulses and for the mono-exponential model we start counting
    # from the 90° excitation pulse
    analytical_signal = m0 * torch.exp(-(torch.cumsum(torch.tensor([te] * n_echoes), dim=0)[:, None] / t2))

    # Two TSE trains with long TR in between to ensure full T1 relaxation
    flip_angles = torch.tensor([torch.pi] * n_echoes)
    rf_phases = 0.0
    sequence = EPGSequence()
    sequence.append(TseBlock(refocusing_flip_angles=flip_angles, refocusing_rf_phases=rf_phases, te=te))
    sequence.append(DelayBlock(delay_time=40.0))
    sequence.append(TseBlock(refocusing_flip_angles=flip_angles, refocusing_rf_phases=rf_phases, te=te))
    parameters = Parameters(m0, t1, t2)
    _, signals = sequence(parameters)
    epg_signal = torch.stack(list(signals), dim=0)

    # first TSE train
    torch.testing.assert_close(epg_signal[:n_echoes], analytical_signal, rtol=1e-3, atol=1e-3)
    # second TSE train
    torch.testing.assert_close(epg_signal[n_echoes:], analytical_signal, rtol=1e-3, atol=1e-3)


def test_EpgTse_parameter_mismatch() -> None:
    """Verify error for shape mismatch."""
    refocusing_flip_angles = torch.ones((1, 2))
    refocusing_rf_phases = torch.ones((1, 3))
    with pytest.raises(ValueError, match='Shapes of flip_angles'):
        EpgTseModel(refocusing_flip_angles, refocusing_rf_phases, te=0.1)


def test_EpgTse_parameter_broadcasting() -> None:
    """Verify correct broadcasting of values."""
    refocusing_rf_phases = torch.ones((1,))
    refocusing_flip_angles = torch.ones((20,))
    epg_model = EpgTseModel(refocusing_flip_angles, refocusing_rf_phases, te=0.1)
    m0 = t1 = t2 = b1_relative = torch.randn((30,))
    (epg_signal,) = epg_model(m0, t1, t2, b1_relative)
    assert epg_signal.shape == (20, 30)


def test_EpgTse_neg_te() -> None:
    """Verify error for negative te."""
    refocusing_flip_angles = refocusing_rf_phases = torch.ones((1, 2))
    with pytest.raises(ValueError, match='Negative echo time'):
        EpgTseModel(refocusing_flip_angles=refocusing_flip_angles, refocusing_rf_phases=refocusing_rf_phases, te=-0.1)


@SHAPE_VARIATIONS_SIGNAL_MODELS
def test_EpgTse_shape(parameter_shape, contrast_dim_shape, signal_shape) -> None:
    """Test correct signal shapes."""
    rng = RandomGenerator(0)
    refocusing_flip_angles = rng.float32_tensor(contrast_dim_shape, low=1e-5, high=5)
    refocusing_rf_phases = rng.float32_tensor(contrast_dim_shape, low=1e-5, high=0.5)
    t1 = rng.float32_tensor(parameter_shape, low=1e-5, high=5)
    t2 = rng.float32_tensor(parameter_shape, low=1e-5, high=0.5)
    m0 = rng.complex64_tensor(parameter_shape)
    relative_b1 = rng.complex64_tensor(parameter_shape)

    model_op = EpgTseModel(refocusing_flip_angles, refocusing_rf_phases, te=0.1)
    (signal,) = model_op(m0, t1, t2, relative_b1)
    assert signal.shape == signal_shape


@pytest.mark.cuda
def test_EpgTse_cuda(parameter_shape: Sequence[int] = (2,)) -> None:
    """Test Tse model works on cuda devices."""
    rng = RandomGenerator(8)
    refocusing_flip_angles = rng.float32_tensor(15, low=1e-5, high=5)
    refocusing_rf_phases = rng.float32_tensor(15, low=1e-5, high=0.5)

    t1 = rng.float32_tensor(parameter_shape, low=1e-5, high=5)
    t2 = rng.float32_tensor(parameter_shape, low=1e-5, high=0.5)
    m0 = rng.complex64_tensor(parameter_shape)
    relative_b1 = rng.complex64_tensor(parameter_shape)

    # Create on CPU, transfer to GPU and run on GPU
    model = EpgTseModel(refocusing_flip_angles, refocusing_rf_phases, te=0.1)
    model.cuda()
    (signal,) = model(m0.cuda(), t1.cuda(), t2.cuda(), relative_b1.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Create on GPU and run on GPU
    model = EpgTseModel(refocusing_flip_angles.cuda(), refocusing_rf_phases.cuda(), te=0.1)
    (signal,) = model(m0.cuda(), t1.cuda(), t2.cuda(), relative_b1.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Create on GPU, transfer to CPU and run on CPU
    model = EpgTseModel(refocusing_flip_angles.cuda(), refocusing_rf_phases.cuda(), te=0.1)
    model.cpu()
    (signal,) = model(m0, t1, t2, relative_b1)
    assert signal.is_cpu
    assert signal.isfinite().all()


def test_epg_se_t1_rho_preparation() -> None:
    """EPG simulation of single-line T1-rho-prep sequence.

    Obtaining a single point with different spin-lock durations follows a mono-exponential model.
    """
    t1 = torch.as_tensor([0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 4.00])
    t2 = torch.as_tensor([0.02, 0.08, 0.16, 0.32, 0.02, 0.08, 0.16, 0.32])
    m0 = torch.ones_like(t1, dtype=torch.complex64)
    t1_rho = torch.as_tensor([0.03, 0.03, 0.09, 0.09, 0.13, 0.13, 0.24, 0.24])

    spin_lock_durations = [0, 0.02, 0.04, 0.08, 0.2, 0.4]

    # analytical signal
    analytical_signal = m0 * torch.exp(-(torch.as_tensor(spin_lock_durations)[:, None] / t1_rho))

    # single readout per T2-prep block with 90° pulse and very short echo time to avoid T2 effects during acquisition
    sequence = EPGSequence()
    for sld in spin_lock_durations:
        sequence.append(T1RhoPrepBlock(spin_lock_duration=sld))
        sequence.append(TseBlock(refocusing_flip_angles=torch.pi, refocusing_rf_phases=0.0, te=1e-6))
        sequence.append(DelayBlock(delay_time=40))
    parameters = Parameters(m0, t1, t2, None, t1_rho)
    _, signals = sequence(parameters)
    epg_signal = torch.stack(list(signals), dim=0)

    torch.testing.assert_close(epg_signal, analytical_signal, rtol=1e-3, atol=1e-3)
