import pytest
import torch

from mrpro.operators.models._WASABITI import WASABITI


def create_data(offset_max=250, offset_nr=101, b0_shift=0, rb1=1.0, t1=1.0, p=4, other=1, coils=1, z=1, y=1, x=1):
    offsets = torch.linspace(-offset_max, offset_max, offset_nr)
    qdata = torch.ones(p, other, coils, z, y, x)
    qdata[0, ...] = b0_shift
    qdata[1, ...] = rb1
    qdata[2, ...] = t1

    return offsets, qdata


@pytest.mark.parametrize(
    'offset_max, offset_nr, b0_shift, rb1, t1, p, other, coils, z, y, x',
    [
        (250, 101, 0, 1.0, 1.0, 4, 1, 1, 1, 1, 1),
        (200, 101, 0, 0, 1.0, 4, 1, 1, 1, 1, 1),
        (10, 101, 10, 10, 1.0, 4, 1, 1, 1, 1, 1),
        (200, 101, 0, 0, 1.0, 4, 1, 1, 1, 1, 1),
    ],
)
def test_WASABITI_signal_model_shape(offset_max, offset_nr, b0_shift, rb1, t1, p, other, coils, z, y, x):
    """Test for correct output shape."""
    offsets, qdata = create_data(offset_max, offset_nr, b0_shift, rb1, t1, p, other, coils, z, y, x)
    trec = torch.ones_like(offsets)

    wasabiti_model = WASABITI(offsets=offsets, trec=trec)
    sig = wasabiti_model.forward(qdata)

    signal_shape = torch.Tensor(offset_nr, coils, z, y, x)
    assert sig.shape == signal_shape.shape


def test_WASABITI_shift_and_symmetry():
    """Test symmetry property of shifted and unshifted WASABITI spectra."""
    offsets_unshifted, qdata_unshifted = create_data(offset_max=500, offset_nr=100, b0_shift=0)
    trec = torch.ones_like(offsets_unshifted)
    wasabiti_model = WASABITI(offsets=offsets_unshifted, trec=trec)
    sig_unshifted = wasabiti_model.forward(qdata_unshifted)

    offsets_shifted, qdata_shifted = create_data(offset_max=500, offset_nr=101, b0_shift=100)
    trec = torch.ones_like(offsets_shifted)
    wasabiti_model = WASABITI(offsets=offsets_shifted, trec=trec)
    sig_shifted = wasabiti_model.forward(qdata_shifted)

    lower_index = (offsets_shifted == -300).nonzero()[0][0].item()
    upper_index = (offsets_shifted == 500).nonzero()[0][0].item()

    assert sig_unshifted[0] == sig_unshifted[-1]
    assert sig_shifted[lower_index] == sig_shifted[upper_index]


@pytest.mark.parametrize(
    't1',
    [(1), (2), (3)],
)
def test_WASABITI_relaxation_term(t1):
    """Test relaxation term (Mzi) of WASABITI model."""
    _, qdata = create_data(offset_max=300, offset_nr=3, b0_shift=0, t1=t1)
    offsets_new = torch.FloatTensor([30000])
    trec = torch.ones_like(offsets_new) * t1
    wasabiti_model = WASABITI(offsets=offsets_new, trec=trec)
    sig = wasabiti_model.forward(qdata)

    assert torch.isclose(sig[0], torch.FloatTensor([1 - torch.exp(torch.FloatTensor([-1]))]), rtol=1e-8)
