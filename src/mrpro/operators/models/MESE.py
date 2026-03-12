"""Multi-echo spin echo (MESE) signal model (EPG)."""

import torch

from mrpro.operators.models.EPG import EPGSequence, Parameters, TseBlock
from mrpro.operators.SignalModel import SignalModel


class MultiEchoSpinEcho(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]):
    """Multi-echo spin echo (MESE) signal model.

    This model simulates a multi-echo spin echo sequence using the extended
    phase graph (`~mr2.operators.models.EPG`) formalism.

    The sequence consists of an excitation pulse followed by a train of
    refocusing pulses, each producing a spin echo [Meiboom]_:

    .. code-block:: text

     Excitation    α₁                    α₂
        | --TE/2-- | --TE/2-- | --TE/2-- | --TE/2-- | ...
                 Echo 1                Echo 2

    When all refocusing flip angles are 180° and
    ``relative_b1 = 1``, the echo amplitudes follow a pure T2 decay.
    If the effective flip angles deviate from 180° (reduced flip angles
    or B1 inhomogeneity), stimulated-echo pathways arise and the signal becomes
    dependent on both T1 and T2.

    References
    ----------
    .. [Meiboom] Meiboom S, Gill D. Modified spin-echo method for measuring nuclear relaxation times.
           Rev Sci Instrum. 1958
    """

    def __init__(self, flip_angles: torch.Tensor, rf_phases: torch.Tensor, echo_time: float = 0.02) -> None:
        """Initialize the multi-echo spin echo signal model.

        Parameters
        ----------
        flip_angles
            Flip angles of the refocusing pulses in rad.
            Shape `(n_echoes,)`.
        rf_phases
            RF phases of the refocusing pulses in rad.
            Shape `(n_echoes,)`.
        echo_time
            Echo time in seconds.

        """
        super().__init__()
        tse = TseBlock(
            refocusing_flip_angles=flip_angles,
            refocusing_rf_phases=rf_phases,
            te=echo_time,
        )
        self.sequence = EPGSequence((tse,))

    def __call__(
        self, m0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, relative_b1: torch.Tensor | None = None
    ) -> tuple[torch.Tensor]:
        """Simulate the multi-echo spin echo signal.

        All input tensors are broadcast together to determine the output shape.

        Parameters
        ----------
        m0
            Equilibrium magnetization / proton density.
            Shape `(...)`, for example `(*other, coils, z, y, x)` or `(samples)`.
        t1
            Longitudinal (T1) relaxation time in seconds.
            Shape `(...)`, for example `(*other, coils, z, y, x)` or`(samples)`.
        t2
            Transversal (T2) relaxation time in seconds.
            Shape `(...)`, for example `(*other, coils, z, y, x)` or `(samples)`.
        relative_b1
            Relative B1 amplitude scaling factor. If None, no B1 inhomogeneity is applied.
            Shape `(...)`, for example `(*other, coils, z, y, x)` or `(samples)`.

        Returns
        -------
            Simulated multi-echo spin echo signal.
            Shape `(echoes, ...)`, for example `(echoes, *other, coils, z, y, x)`
            or `(echoes, samples)`, where `echoes` corresponds to the number of refocusing pulses in the
            echo train.
        """
        return super().__call__(m0, t1, t2, relative_b1)

    def forward(
        self, m0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, relative_b1: torch.Tensor | None = None
    ) -> tuple[torch.Tensor]:
        """Apply forward of MESE.

        .. note::
            Prefer calling the instance of the MESE as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        parameters = Parameters(m0, t1, t2, relative_b1)
        _, signals = self.sequence(parameters)
        shape = torch.broadcast_shapes(m0.shape, t1.shape, t2.shape)
        if relative_b1 is not None:
            shape = torch.broadcast_shapes(shape, relative_b1.shape)
        signal = torch.stack(signals, dim=0).broadcast_to(-1, *shape)
        return (signal,)
