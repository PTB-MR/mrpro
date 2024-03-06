import pytest
import torch
from mrpro.operators.models._WASABITI import WASABITI


def create_data(offset_max=250, offset_nr=101, b0_shift_in=0, rb1=1.0, t1=1.0):
    offsets = torch.linspace(-offset_max, offset_max, offset_nr)
    b0_shift = torch.zeros([1, 1, 1, 1, 1])
    b0_shift[0] = b0_shift_in
    rb1 = torch.Tensor([rb1])
    t1 = torch.Tensor([t1])

    return offsets, b0_shift, rb1, t1


@pytest.mark.parametrize(
    ('offset_max', 'offset_nr', 'b0_shift', 'rb1', 't1', 'coils', 'z', 'y', 'x'),
    [
        (250, 101, 0, 1.0, 1.0, 1, 1, 1, 1),
        (200, 101, 0, 0, 1.0, 1, 1, 1, 1),
        (10, 101, 10, 10, 1.0, 1, 1, 1, 1),
    ],
)
def test_WASABITI_signal_model_shape(offset_max, offset_nr, b0_shift, rb1, t1, coils, z, y, x):
    """Test for correct output shape."""
    offsets, b0_shift, rb1, t1 = create_data(offset_max, offset_nr, b0_shift, rb1, t1)
    trec = torch.ones_like(offsets)

    wasabiti_model = WASABITI(offsets=offsets, trec=trec)
    (signal,) = wasabiti_model.forward(b0_shift, rb1, t1)
    assert signal.shape == (offset_nr, coils, z, y, x)


def test_WASABITI_shift_and_symmetry():
    """Test symmetry property of shifted and unshifted WASABITI spectra."""
    offsets_unshifted, b0_shift, rb1, t1 = create_data(offset_max=500, offset_nr=100, b0_shift_in=0)
    trec = torch.ones_like(offsets_unshifted)
    wasabiti_model = WASABITI(offsets=offsets_unshifted, trec=trec)
    (signal,) = wasabiti_model.forward(b0_shift, rb1, t1)

    offsets_shifted, b0_shift, rb1, t1 = create_data(offset_max=500, offset_nr=101, b0_shift_in=100)
    trec = torch.ones_like(offsets_shifted)
    wasabiti_model = WASABITI(offsets=offsets_shifted, trec=trec)
    (signal_shifted,) = wasabiti_model.forward(b0_shift, rb1, t1)

    lower_index = (offsets_shifted == -300).nonzero()[0][0].item()
    upper_index = (offsets_shifted == 500).nonzero()[0][0].item()

    assert signal[0] == signal[-1]
    assert signal_shifted[lower_index] == signal_shifted[upper_index]


@pytest.mark.parametrize('t1', [(1), (2), (3)])
def test_WASABITI_relaxation_term(t1):
    """Test relaxation term (Mzi) of WASABITI model."""
    _, b0_shift, rb1, t1 = create_data(offset_max=300, offset_nr=3, b0_shift_in=0, t1=t1)
    offsets_new = torch.FloatTensor([30000])
    trec = torch.ones_like(offsets_new) * t1
    wasabiti_model = WASABITI(offsets=offsets_new, trec=trec)
    sig = wasabiti_model.forward(b0_shift, rb1, t1)

    assert torch.isclose(sig[0], torch.FloatTensor([1 - torch.exp(torch.FloatTensor([-1]))]), rtol=1e-8)
