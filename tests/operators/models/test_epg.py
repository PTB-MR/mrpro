"""Tests for EPG signal models."""

import pytest
import torch
from mrpro.operators.models import EpgMrfFispWithPreparation, EpgTse
from tests.operators.models.conftest import SHAPE_VARIATIONS_SIGNAL_MODELS, create_parameter_tensor_tuples


def test_EpgMrfFispWithPreparation_parameter_broadcasting():
    """Verify correct broadcasting of values."""
    te = tr = rf_phases = torch.ones((1,))
    flip_angles = torch.ones((20,))
    epg_mrf_fisp = EpgMrfFispWithPreparation(flip_angles=flip_angles, rf_phases=rf_phases, te=te, tr=tr)
    m0 = t1 = t2 = torch.randn((30,))
    (epg_signal,) = epg_mrf_fisp.forward(m0, t1, t2)
    assert epg_signal.shape == (20, 30)


def test_EpgMrfFispWithPreparation_parameter_mismatch():
    """Verify error for shape mismatch."""
    flip_angles = rf_phases = tr = torch.ones((1, 2))
    te = torch.ones((1, 3))
    with pytest.raises(ValueError, match='Shapes of flip_angles'):
        EpgMrfFispWithPreparation(flip_angles=flip_angles, rf_phases=rf_phases, te=te, tr=tr)


def test_EpgMrfFispWithPreparation_inversion_recovery():
    """EPG simulation of single-line inversion recovery sequence.

    Obtaining a single point at different times after an inversion pulse follows a mono-exponential model.
    """
    t1 = torch.as_tensor([100, 200, 300, 400, 500, 1000, 2000, 4000])
    t2 = torch.as_tensor([20, 80, 160, 320, 20, 80, 160, 320])
    m0 = torch.ones_like(t1)

    # inversion times
    ti = [0, 200, 400, 600, 800, 1000, 1500, 2000, 2500]

    # analytical signal
    analytical_signal = m0 * (1 - 2 * torch.exp(-(torch.as_tensor(ti)[..., None] / t1)))

    # constant flip angle, TE and TR
    flip_angles = torch.as_tensor([torch.pi / 2] * len(ti))
    rf_phases = torch.pi / 2
    te = 0.0001  # very short echo time to avoid T2 effects
    tr = 7.0
    inv_prep_ti = ti
    t2_prep_te = None
    n_rf_pulses_per_block = 1
    delay_after_block = 40000
    epg_model = EpgMrfFispWithPreparation(
        flip_angles, rf_phases, te, tr, inv_prep_ti, t2_prep_te, n_rf_pulses_per_block, delay_after_block
    )
    (epg_signal,) = epg_model(m0, t1, t2)

    torch.testing.assert_close(epg_signal.real, analytical_signal, rtol=1e-3, atol=1e-3)


def test_EpgMrfFispWithPreparation_t2_preparation():
    """EPG simulation of single-line T2-prep sequence.

    Obtaining a single point at different TEs of a T2-prep pulse follows a mono-exponential model.
    """
    t1 = torch.as_tensor([100, 200, 300, 400, 500, 1000, 2000, 4000])
    t2 = torch.as_tensor([20, 80, 160, 320, 20, 80, 160, 320])
    m0 = torch.ones_like(t1)

    # echo times
    t2_prep_te_times = [0, 20, 40, 80, 200]

    # analytical signal
    analytical_signal = m0 * torch.exp(-(torch.as_tensor(t2_prep_te_times)[:, None] / t2))

    # constant flip angle, TE and TR
    flip_angles = torch.as_tensor([torch.pi / 2] * len(t2_prep_te_times))
    rf_phases = torch.pi / 2
    te = 0.0001  # very short echo time to avoid T2 effects during acquisition
    tr = 7.0
    inv_prep_ti = None
    t2_prep_te = t2_prep_te_times
    n_rf_pulses_per_block = 1
    delay_after_block = 40000
    epg_model = EpgMrfFispWithPreparation(
        flip_angles, rf_phases, te, tr, inv_prep_ti, t2_prep_te, n_rf_pulses_per_block, delay_after_block
    )
    (epg_signal,) = epg_model(m0, t1, t2)

    torch.testing.assert_close(epg_signal.real, analytical_signal, rtol=1e-3, atol=1e-3)


@SHAPE_VARIATIONS_SIGNAL_MODELS
def test_EpgMrfFispWithPreparation_shape(parameter_shape, contrast_dim_shape, signal_shape):
    """Test correct signal shapes."""
    flip_angles, rf_phases, te, tr = create_parameter_tensor_tuples(contrast_dim_shape, number_of_tensors=4)
    model_op = EpgMrfFispWithPreparation(flip_angles=flip_angles, rf_phases=rf_phases, te=te, tr=tr)
    m0, t1, t2 = create_parameter_tensor_tuples(parameter_shape, number_of_tensors=3)
    (signal,) = model_op.forward(m0, t1, t2)
    assert signal.shape == signal_shape


def test_EpgTse_mono_exponential_decay():
    """Echo trains should follow monoexpontial model for long TR."""
    t1 = torch.as_tensor([100, 200, 300, 400, 500, 1000, 2000, 4000])
    t2 = torch.as_tensor([20, 80, 160, 320, 20, 80, 160, 320])
    m0 = torch.ones_like(t1)
    b1_scaling_factor = torch.ones_like(t1)

    te = torch.as_tensor([20, 20, 80, 20, 100])

    # analytical signal
    # cumsum because te is the time between refocusing pulses and for the mono-exponential model we starting counting
    # from the 90° excitation pulse
    analytical_signal = m0 * torch.exp(-(torch.cumsum(te, dim=0)[:, None] / t2))

    flip_angles = torch.ones_like(te) * torch.pi
    rf_phases = 0
    tr = torch.as_tensor([40000, 40000])
    epg_mrf_fisp = EpgTse(flip_angles=flip_angles, rf_phases=rf_phases, te=te, tr=tr)
    (epg_signal,) = epg_mrf_fisp.forward(m0, t1, t2, b1_scaling_factor)

    # first TSE train
    torch.testing.assert_close(epg_signal.real[: te.shape[0]], analytical_signal, rtol=1e-3, atol=1e-3)
    # second TSE train
    torch.testing.assert_close(epg_signal.real[te.shape[0] :], analytical_signal, rtol=1e-3, atol=1e-3)


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


@SHAPE_VARIATIONS_SIGNAL_MODELS
def test_EpgTse_shape(parameter_shape, contrast_dim_shape, signal_shape):
    """Test correct signal shapes."""
    flip_angles, rf_phases, te = create_parameter_tensor_tuples(contrast_dim_shape, number_of_tensors=3)
    model_op = EpgTse(flip_angles=flip_angles, rf_phases=rf_phases, te=te)
    m0, t1, t2, b1 = create_parameter_tensor_tuples(parameter_shape, number_of_tensors=4)
    (signal,) = model_op.forward(m0, t1, t2, b1)
    assert signal.shape == signal_shape
