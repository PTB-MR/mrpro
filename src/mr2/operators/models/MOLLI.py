"""Modified Look-Locker inversion recovery (MOLLI) signal model for T1 mapping."""

from collections.abc import Sequence

import torch

from mr2.operators.SignalModel import SignalModel
from mr2.utils.reshape import unsqueeze_right


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

    def __call__(self, a: torch.Tensor, c: torch.Tensor, t1: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the Modified Look-Locker Inversion recovery (MOLLI) signal model.

        Calculates the signal based on the formula:
        :math:`S(TI) = a(1 - (1 + c) e^{-TI c / T1})`,
        where `TI` are the inversion times.

        Parameters
        ----------
        a
            Parameter 'a' in the MOLLI signal model.
            Shape `(...)`, for example `(*other, coils, z, y, x)` or `(samples)`.
        c
            Parameter 'c = b/a' in the MOLLI signal model.
            Shape `(...)`, for example `(*other, coils, z, y, x)` or `(samples)`.
        t1
            Longitudinal relaxation time T1.
            Shape `(...)`, for example `(*other, coils, z, y, x)` or `(samples)`.

        Returns
        -------
            Shape `(times ...)`, for example `(times, *other, coils, z, y, x)`, or `(times, samples)`
            where `times` is the number of inversion times.
        """
        return super().__call__(a, c, t1)

    def forward(self, a: torch.Tensor, c: torch.Tensor, t1: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of MOLLI.

        .. note::
            Prefer calling the instance of the MOLLI as ``operator(x)`` over directly calling this method.
            See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        ndim = max(a.ndim, c.ndim, t1.ndim)
        ti = unsqueeze_right(self.ti, ndim - self.ti.ndim + 1)  # leftmost is time
        signal = a * (1 - (1 + c) * torch.exp(-ti / t1 * c))
        return (signal,)
