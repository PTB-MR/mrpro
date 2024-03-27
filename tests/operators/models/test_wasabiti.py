import pytest
import torch
from mrpro.operators.models import WASABITI


def create_data(offset_max=500, n_offsets=101, b0_shift=0, rb1=1.0, t1=1.0):
    offsets = torch.linspace(-offset_max, offset_max, n_offsets)
    return offsets, torch.Tensor([b0_shift]), torch.Tensor([rb1]), torch.Tensor([t1])


def test_WASABITI_shift_and_symmetry():
    """Test symmetry property of shifted and unshifted WASABITI spectra."""
    offsets_unshifted, b0_shift, rb1, t1 = create_data()
    # note that the symmetry is only guaranteed for identical trec values
    trec = torch.ones_like(offsets_unshifted)
    wasabiti_model = WASABITI(offsets=offsets_unshifted, trec=trec)
    (signal,) = wasabiti_model.forward(b0_shift, rb1, t1)

    offsets_shifted, b0_shift, rb1, t1 = create_data(b0_shift=100)
    trec = torch.ones_like(offsets_shifted)
    wasabiti_model = WASABITI(offsets=offsets_shifted, trec=trec)
    (signal_shifted,) = wasabiti_model.forward(b0_shift, rb1, t1)

    lower_index = (offsets_shifted == -300).nonzero()[0][0].item()
    upper_index = (offsets_shifted == 500).nonzero()[0][0].item()

    assert signal[0] == signal[-1], 'Result should be symmetric around center'
    assert signal_shifted[lower_index] == signal_shifted[upper_index], 'Result should be symmetric around shift'


@pytest.mark.parametrize('t1', [(1), (2), (3)])
def test_WASABITI_relaxation_term(t1):
    """Test relaxation term (Mzi) of WASABITI model."""
    offset, b0_shift, rb1, t1 = create_data(offset_max=50000, n_offsets=1, t1=t1)
    trec = torch.ones_like(offset) * t1
    wasabiti_model = WASABITI(offsets=offset, trec=trec)
    sig = wasabiti_model.forward(b0_shift, rb1, t1)

    assert torch.isclose(sig[0], torch.FloatTensor([1 - torch.exp(torch.FloatTensor([-1]))]), rtol=1e-8)


def test_WASABITI_offsets_trec_mismatch():
    """Verify error for shape mismatch."""
    offsets = torch.ones((1, 2))
    trec = torch.ones((1,))
    with pytest.raises(ValueError, match='Shape of trec'):
        WASABITI(offsets=offsets, trec=trec)
