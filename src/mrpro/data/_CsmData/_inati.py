import torch

from mrpro.utils.window import sliding_window


def coil_map_study_2d_Inati(data: torch.Tensor, ks: int, power: int, padding_mode='circular'):
    """Coil sensitivity maps using the method described in Inati et al. 2004.

    Parameters
    ----------
    data: Images of shape (coil, E1, E0)
    ks: kernel size
    power: number of iterations
    padding_mode: padding mode for the sliding window
    """

    if ks % 2 != 1:
        raise ValueError('ks must be odd')
    if power < 1:
        raise ValueError('power must be at least 1')

    halfKs = ks // 2
    # adding another dimension before padding is a workaround for https://github.com/pytorch/pytorch/issues/95320
    padded = torch.nn.functional.pad(data[None], (halfKs, halfKs, halfKs, halfKs), mode=padding_mode)[0]
    D = sliding_window(padded, (ks, ks), axis=(-1, -2)).flatten(-2)  # coil E1, E0, ks*ks
    DH_D = torch.einsum('i...j,k...j->...ik', D, D.conj())  # E1,E0,coil,coil
    singular_vector = torch.sum(D, dim=-1)  # coil, E1, E0
    singular_vector /= singular_vector.abs().square().sum(0, keepdim=True).sqrt()
    for _ in range(power):
        singular_vector = torch.einsum('...ij,j...->i...', DH_D, singular_vector)  # coil, E1, E0
        singular_vector /= singular_vector.abs().square().sum(0, keepdim=True).sqrt()
    singular_value = torch.einsum('i...j,i...->...j', D, singular_vector)  # E1, E0, ks*ks
    phase = singular_value.sum(-1)
    phase /= phase.abs()  # E1, E0
    csm = singular_vector.conj() * phase[None, ...]
    return csm
