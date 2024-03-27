import torch
from mrpro.operators.models import WASABI


def create_data(offset_max=500, n_offsets=101, b0_shift=0, rb1=1.0, c=1.0, d=2.0):
    offsets = torch.linspace(-offset_max, offset_max, n_offsets)
    return offsets, torch.Tensor([b0_shift]), torch.Tensor([rb1]), torch.Tensor([c]), torch.Tensor([d])


def test_WASABI_shift():
    """Test symmetry property of shifted and unshifted WASABI spectra."""
    offsets_unshifted, b0_shift, rb1, c, d = create_data()
    wasabi_model = WASABI(offsets=offsets_unshifted)
    (signal,) = wasabi_model.forward(b0_shift, rb1, c, d)

    offsets_shifted, b0_shift, rb1, c, d = create_data(b0_shift=100)
    wasabi_model = WASABI(offsets=offsets_shifted)
    (signal_shifted,) = wasabi_model.forward(b0_shift, rb1, c, d)

    lower_index = (offsets_shifted == -300).nonzero()[0][0].item()
    upper_index = (offsets_shifted == 500).nonzero()[0][0].item()

    assert signal[0] == signal[-1], 'Result should be symmetric around center'
    assert signal_shifted[lower_index] == signal_shifted[upper_index], 'Result should be symmetric around shift'


def test_WASABI_extreme_offset():
    offset, b0_shift, rb1, c, d = create_data(offset_max=30000, n_offsets=1)
    wasabi_model = WASABI(offsets=offset)
    (signal,) = wasabi_model.forward(b0_shift, rb1, c, d)

    assert torch.isclose(signal, torch.tensor([1.0])), 'For an extreme offset, the signal should be unattenuated'
