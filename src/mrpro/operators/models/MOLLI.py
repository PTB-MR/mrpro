"""Modified Look-Locker inversion recovery (MOLLI) signal model for T1 mapping."""

import torch

from mrpro.operators.SignalModel import SignalModel


class MOLLI(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor]):
    """Signal model for Modified Look-Locker inversion recovery (MOLLI)."""

    def __init__(self, ti: float | torch.Tensor):
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

    def forward(self, a: torch.Tensor, b: torch.Tensor, t1: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply MOLLI signal model.

        Parameters
        ----------
        a
            parameter a in MOLLI signal model
            with shape (... other, coils, z, y, x)
        b
            parameter b in MOLLI signal model
            with shape (... other, coils, z, y, x)
        t1
            longitudinal relaxation time T1
            with shape (... other, coils, z, y, x)

        Returns
        -------
            signal
            with shape (time ... other, coils, z, y, x)
        """
        delta_ndim = a.ndim - (self.ti.ndim - 1)  # -1 for time
        ti = self.ti[..., *[None] * (delta_ndim)] if delta_ndim > 0 else self.ti
        c = b / torch.where(a == 0, 1e-10, a)
        t1 = torch.where(t1 == 0, t1 + 1e-10, t1)
        signal = a * (1 - c * torch.exp(ti / t1 * (1 - c)))
        return (signal,)
