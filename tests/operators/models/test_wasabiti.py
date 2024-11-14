"""Tests for the WASABITI signal model."""

import pytest
import torch
from mrpro.operators.models import WASABITI
from tests import autodiff_test
from tests.operators.models.conftest import SHAPE_VARIATIONS_SIGNAL_MODELS, create_parameter_tensor_tuples


def create_data(
    offset_max=500, n_offsets=101, b0_shift=0, rb1=1.0, t1=1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    offsets = torch.linspace(-offset_max, offset_max, n_offsets)
    return offsets, torch.Tensor([b0_shift]), torch.Tensor([rb1]), torch.Tensor([t1])


def test_WASABITI_symmetry():
    """Test symmetry property of complete WASABITI spectra."""
    offsets, b0_shift, rb1, t1 = create_data()
    wasabiti_model = WASABITI(offsets=offsets, trec=torch.ones_like(offsets))
    (signal,) = wasabiti_model(b0_shift, rb1, t1)

    # check that all values are symmetric around the center
    assert torch.allclose(signal, signal.flipud(), rtol=1e-15), 'Result should be symmetric around center'


def test_WASABITI_symmetry_after_shift():
    """Test symmetry property of shifted WASABITI spectra."""
    offsets_shifted, b0_shift, rb1, t1 = create_data(b0_shift=100)
    trec = torch.ones_like(offsets_shifted)
    wasabiti_model = WASABITI(offsets=offsets_shifted, trec=trec)
    (signal_shifted,) = wasabiti_model(b0_shift, rb1, t1)

    lower_index = int((offsets_shifted == -300).nonzero()[0][0])
    upper_index = int((offsets_shifted == 500).nonzero()[0][0])

    assert signal_shifted[lower_index] == signal_shifted[upper_index], 'Result should be symmetric around shift'


def test_WASABITI_asymmetry_for_non_unique_trec():
    """Test symmetry property of WASABITI spectra for non-unique trec values."""
    offsets_unshifted, b0_shift, rb1, t1 = create_data(n_offsets=11)
    trec = torch.ones_like(offsets_unshifted)
    # set first half of trec values to 2.0
    trec[: len(offsets_unshifted) // 2] = 2.0

    wasabiti_model = WASABITI(offsets=offsets_unshifted, trec=trec)
    (signal,) = wasabiti_model(b0_shift, rb1, t1)

    assert not torch.allclose(signal, signal.flipud(), rtol=1e-8), 'Result should not be symmetric around center'


@pytest.mark.parametrize('t1', [(1), (2), (3)])
def test_WASABITI_relaxation_term(t1):
    """Test relaxation term (Mzi) of WASABITI model."""
    offset, b0_shift, rb1, t1 = create_data(offset_max=50000, n_offsets=1, t1=t1)
    trec = torch.ones_like(offset) * t1
    wasabiti_model = WASABITI(offsets=offset, trec=trec)
    sig = wasabiti_model(b0_shift, rb1, t1)

    assert torch.isclose(sig[0], torch.FloatTensor([1 - torch.exp(torch.FloatTensor([-1]))]), rtol=1e-8)


def test_WASABITI_offsets_trec_mismatch():
    """Verify error for shape mismatch."""
    offsets = torch.ones((1, 2))
    trec = torch.ones((1,))
    with pytest.raises(ValueError, match='Shape of trec'):
        WASABITI(offsets=offsets, trec=trec)


@SHAPE_VARIATIONS_SIGNAL_MODELS
def test_WASABITI_shape(parameter_shape, contrast_dim_shape, signal_shape):
    """Test correct signal shapes."""
    ti, trec = create_parameter_tensor_tuples(contrast_dim_shape, number_of_tensors=2)
    model_op = WASABITI(ti, trec)
    b0_shift, rb1, t1 = create_parameter_tensor_tuples(parameter_shape, number_of_tensors=3)
    (signal,) = model_op(b0_shift, rb1, t1)
    assert signal.shape == signal_shape


def test_autodiff_WASABITI():
    """Test autodiff works for WASABITI model."""
    offset, b0_shift, rb1, t1 = create_data(offset_max=300, n_offsets=2)
    trec = torch.ones_like(offset) * t1
    wasabiti_model = WASABITI(offsets=offset, trec=trec)
    autodiff_test(wasabiti_model, b0_shift, rb1, t1)
