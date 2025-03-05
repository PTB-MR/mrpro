"""Modified Look-Locker inversion recovery (MOLLI) signal model for T1 mapping."""

from collections.abc import Sequence

import torch

from mrpro.operators.SignalModel import SignalModel
from mrpro.utils import unsqueeze_right


class MOLLI(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor]):
    r"""Signal model for Modified Look-Locker inversion recovery (MOLLI).

    This model describes
    :math:`M_z(t) = a(1 - (1 + c) e^{-t / T_1^*})` with :math:`T_1^* = T_1 / c`.

    This is a small modification from the original MOLLI signal model [MESS2004]_:
    :math:`M_z(t) = a - be^{(-t / T_1^*)}` with :math:`T_1^* = T1 / (b/a - 1)`.

    For a meaningful result chose :math:`c \in R_{>0}`, :math:`t \in R_{>0}`, and :math:`T_1 \in R_{>0}`

    .. [MESS2004] Messroghli DR, Radjenovic A, Kozerke S, Higgins DM, Sivananthan MU, Ridgway JP (2004) Modified
      look-locker inversion recovery (MOLLI) for high-resolution T 1 mapping of the heart. MRM, 52(1).
      https://doi.org/10.1002/mrm.20110
    """

    def __init__(self, ti: float | torch.Tensor | Sequence[float]):
        """Initialize MOLLI signal model for T1 mapping.

        Parameters
        ----------
        ti
            inversion times
            with shape (time, ...)
        """
        super().__init__()
        ti = torch.as_tensor(ti)
        self.ti = torch.nn.Parameter(ti, requires_grad=ti.requires_grad)

    def forward(self, a: torch.Tensor, c: torch.Tensor, t1: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply MOLLI signal model.

        Parameters
        ----------
        a
            parameter a in MOLLI signal model
            with shape `(*other, coils, z, y, x)`
        c
            parameter c = b/a in MOLLI signal model
            with shape `(*other, coils, z, y, x)`
        t1
            longitudinal relaxation time T1
            with shape `(*other, coils, z, y, x)`

        Returns
        -------
            signal with shape `(time, *other, coils, z, y, x)`
        """
        ndim = max(a.ndim, c.ndim, t1.ndim)
        ti = unsqueeze_right(self.ti, ndim - self.ti.ndim + 1)  # leftmost is time
        signal = a * (1 - (1 + c) * torch.exp(-ti / t1 * c))
        return (signal,)
