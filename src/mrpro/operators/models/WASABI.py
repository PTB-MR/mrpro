from mrpro.operators import Operator
from einops import rearrange

import torch

"""

    :param x: frequency offsets [Hz]
    :param b0_shift: B0 shift [Hz]
    :param b1: B1 peak amplitude [µT]
    :param c: free fit parameter
    :param d: free fit parameter
    :param tp: duration of the WASABI pulse [s]
    :param gamma: gyromagnetic ratio [MHz/T]
    :return fit function
"""


class WASABI(Operator):
    def __init__(
        self,
        offsets: torch.Tensor,
        tp: float = 0.005,
        b1_nom: float = 3.70,
        gamma: float = 42.5764,
        freq: float = 127.7292,
    ) -> None:
        """WASABI function for simultaneous determination of B1 and B0.

        For more details see: https://doi.org/10.1002/mrm.26133

        Parameters
        ----------
        offsets
            frequency offsets [Hz]
        tp, optional
            RF pulse duration [s], by default 0.005
        b1_nom, optional
            nominal B1 amplitude [µT], by default 3.70
        gamma, optional
            gyromagnetic ratio [MHz/T], by default 42.5764
        freq, optional
            larmor frequency [MHz], by default 127.7292
        """
        self.offsets = offsets
        self.tp = tp
        self.b1_nom = b1_nom
        self.gamma = gamma
        self.freq = freq

    def forward(self, qdata: torch.Tensor) -> torch.Tensor:
        b0_shift = qdata[0, ...]
        rb1 = qdata[1, ...]
        c = qdata[2, ...]
        d = qdata[3, ...]

        b1 = self.b1_nom * rb1
        offsets = rearrange(self.offsets, 'x -> x 1 1 1 1 1')

        res = c - d * (torch.pi * b1 * self.gamma * self.tp) ** 2 * torch.sinc(
            self.tp * torch.sqrt((b1 * self.gamma) ** 2 + ((offsets - b0_shift) * self.freq) ** 2) ** 2
        )

        return rearrange(res, 'p other c z y x -> (p other) c z y x')
