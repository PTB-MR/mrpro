"""NeuroMRF (EPG) signal model."""

import torch

from mrpro.operators.models.EPG import EPGSequence, FispBlock, InversionBlock, Parameters
from mrpro.operators.SignalModel import SignalModel


def _delics_flipangles() -> torch.Tensor:
    """Generate flip angles for the DeliCS dataset."""

    def triangular_segment(n_rise: int, n_flat: int, n_fall: int, peak: float) -> torch.Tensor:
        rise = torch.linspace(0, peak, n_rise + 1)[1:]
        flat = torch.full((n_flat,), peak)
        fall = torch.linspace(peak, 0, n_fall + 1)[1:]

        return torch.cat((rise, flat, fall))

    seg1 = triangular_segment(n_rise=84, n_flat=40, n_fall=84, peak=25.0)
    seg2 = triangular_segment(n_rise=84, n_flat=0, n_fall=84, peak=75.0)
    seg3 = torch.full((500 - (len(seg1) + len(seg2)),), 8.3333)

    flip_angles = torch.deg2rad(torch.cat((seg1, seg2, seg3)))
    return flip_angles


DELICS_FLIP_ANGLES = _delics_flipangles()


class NeuroMRF(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]):
    """A fingerprinting signal model with inversion preparation.

    An extended phase graph (`~mr2.operators.models.EPG`) simulation of a single inversion preperared
    Fisp block with varying flip angles.
    Matches the signal model used in [DeliCS]_ (raw k-space data available at [zenodo]_).

    Note
    ----
    This model is on purpose not flexible in all design choices. Instead, consider writing a custom
    `~mrpro.operators.SignalModel` based on this implementation if you need to simulate a different sequence.

    References
    ----------
    .. [DeliCS] Schauman, S.S., Iyer, S.S., Sandino, C.M. et al. Deep learning initialized compressed sensing (Deli-CS)
            in volumetric spatio-temporal subspace reconstruction. Magn Reson Mater Phy 38 (2025).
            https://doi.org/10.1007/s10334-024-01222-w
    .. [zenodo] https://zenodo.org/records/7697373
    """

    def __init__(
        self,
        flip_angles: torch.Tensor = DELICS_FLIP_ANGLES,
        echo_time: float = 0.0007,
        repetition_time: float = 0.012,
        inversion_time: float = 0.020,
    ) -> None:
        """Initialize the NeuroMRF signal model.

        Parameters
        ----------
        flip_angles
            Flip angles of the RF pulses in rad.
            Length determines the number of acquisitions
        echo_time
            Echo time in seconds.
        repetition_time
            Repetition time in seconds.
        inversion_time
            Inversion time in seconds.
        """
        super().__init__()

        self.sequence = EPGSequence()
        self.sequence.append(InversionBlock(inversion_time))

        self.sequence.append(
            FispBlock(
                flip_angles=flip_angles,
                rf_phases=torch.tensor(0.0, device=flip_angles.device),
                te=echo_time,
                tr=repetition_time,
            )
        )

    def __call__(
        self, m0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, relative_b1: torch.Tensor | None = None
    ) -> tuple[torch.Tensor]:
        """Simulate the NeuroMRF signal.

        Parameters
        ----------
        m0
            Equilibrium signal / proton density. (complex).
            Shape `(...)`, for example `(*other, coils, z, y, x)` or `(samples)`.
        t1
            Longitudinal (T1) relaxation time in seconds.
            Shape `(...)`, for example `(*other, coils, z, y, x)` or `(samples)`.
        t2
            Transversal (T2) relaxation time in seconds.
            Shape `(...)`, for example `(*other, coils, z, y, x)` or `(samples)`.
        relative_b1
            Relative B1 amplitude scaling factor. If None, no B1 inhomogeneity is applied.
            Shape `(...)`, for example `(*other, coils, z, y, x)` or `(samples)`.

        Returns
        -------
            Simulated MR Fingerprinting signal.
            Shape `(acquisitions ...)`, for example `(acquisitions, *other, coils, z, y, x)` or
            `(acquisitions, samples)` where `acquisitions` corresponds to the different acquisitions
            in the sequence.
        """
        return super().__call__(m0, t1, t2, relative_b1)

    def forward(
        self, m0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, relative_b1: torch.Tensor | None = None
    ) -> tuple[torch.Tensor]:
        """Simulate the NeuroMRF signal.

        .. note::
            Prefer calling the instance of the NeuroMRF as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        params = Parameters(m0, t1, t2, relative_b1)
        _, signals = self.sequence(params, states=100)
        shape = torch.broadcast_shapes(m0.shape, t1.shape, t2.shape)
        if relative_b1 is not None:
            shape = torch.broadcast_shapes(shape, relative_b1.shape)
        signal = torch.stack(signals, dim=0).broadcast_to(-1, *shape)
        return (signal,)
