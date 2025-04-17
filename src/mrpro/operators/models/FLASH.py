"""FLASH/Spoiled gradient echo signal model."""

import torch

from mrpro.operators.SignalModel import SignalModel
from mrpro.utils.reshape import unsqueeze_right


class FLASH(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor]):
    """FLASH/Spoiled gradient echo signal model."""

    def __init__(
        self, flip_angle: float | torch.Tensor, echo_time: float | torch.Tensor, repetition_time: float | torch.Tensor
    ) -> None:
        r"""Initialize FLASH/Spoiled gradient echo signal model.

        The model is defined as:
        :math:`S = M_0 e^{-t_i / T_{2*}}  \frac{\sin(\alpha)(1 - e^{-t_e / T_1})}{(1 - \cos(\alpha) e^{-t_e / T_1})}`

        where :math:`M_0` is the magnetization at the beginning of the sequence, :math:`\alpha` is the flip angle,
        :math:`t_e` is the echo time, and :math:`t_i` is the repetition time.

        Parameters
        ----------
        flip_angle
            Flip angle in radians.
        echo_time
            Echo time.
        repetition_time
            Repetition time.
        """
        super().__init__()
        self.flip_angle = torch.nn.Parameter(torch.as_tensor(flip_angle))
        self.echo_time = torch.nn.Parameter(torch.as_tensor(echo_time))
        self.repetition_time = torch.nn.Parameter(torch.as_tensor(repetition_time))

    def forward(self, m0: torch.Tensor, t1: torch.Tensor, t2star: torch.Tensor) -> tuple[torch.Tensor,]:
        """Calculate Signal.

        Parameters
        ----------
        m0
            Magnetization at the beginning of the sequence.
        t1
            T1 relaxation time.
        t2star
            T2* relaxation time.
        """
        ndim = max(m0.ndim, t1.ndim, t2star.ndim) + 1
        flip_angle = unsqueeze_right(self.flip_angle, ndim - self.flip_angle.ndim)
        echo_time = unsqueeze_right(self.echo_time, ndim - self.echo_time.ndim)
        repetition_time = unsqueeze_right(self.repetition_time, ndim - self.repetition_time.ndim)
        e1 = torch.exp(-repetition_time / t1)
        e2 = torch.exp(-echo_time / t2star)
        signal = m0 * torch.sin(flip_angle) * (1 - e1) * e2 / (1 - torch.cos(flip_angle) * e1)
        return (signal,)
