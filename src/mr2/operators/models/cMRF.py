"""Cardiac MR Fingerprinting signal model (EPG)."""

from collections.abc import Sequence

import torch

from mr2.operators.models.EPG import DelayBlock, EPGSequence, FispBlock, InversionBlock, Parameters, T2PrepBlock
from mr2.operators.SignalModel import SignalModel


class CardiacFingerprinting(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor]):
    """Cardiac MR Fingerprinting signal model.

    This model simulates a cardiac MR fingerprinting sequence as described in [HAMI2017]_ and [HAMI2020]_ using the
    extended phase graph (`~mr2.operators.models.EPG`) formalism.

    It is a three-fold repetition of

    .. code-block:: text

        Block 0          Block 1          Block 2          Block 3          Block 4
        R-peak           R-peak           R-peak           R-peak           R-peak
        |----------------|----------------|----------------|----------------|----------------
         [INV 30ms][ACQ]           [ACQ]   [T2-prep][ACQ]   [T2-prep][ACQ]  [T2-prep][ACQ]

    Note
    ----
    This model is on purpose not flexible in all design choices. Instead, consider writing a custom
    `~mr2.operators.SignalModel` based on this implementation if you need to simulate a different sequence.

    References
    ----------
    .. [HAMI2017] Hamilton, J. I. et al. (2017) MR fingerprinting for rapid quantification of myocardial T1, T2, and
           proton spin density. Magn. Reson. Med. 77 http://doi.wiley.com/10.1002/mrm.26216
    .. [HAMI2020] Hamilton, J.I. et al. (2020) Simultaneous Mapping of T1 and T2 Using Cardiac Magnetic Resonance
           Fingerprinting in a Cohort of Healthy Subjects at 1.5T. J Magn Reson Imaging, 52: 1044-1052. https://doi.org/10.1002/jmri.27155
    """

    def __init__(
        self,
        acquisition_times: torch.Tensor | Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
        echo_time: float = 0.0015,
        repetition_time: float = 0.005,
        t2_prep_echo_times: tuple[float, float, float] = (0.03, 0.05, 0.08),
    ) -> None:
        """Initialize the Cardiac MR Fingerprinting signal model.

        Parameters
        ----------
        acquisition_times
            Times of the acquisistions in s.
            Either 15 values (the first acquisition of each block) or 705 values for all acquisitions.
            In both cases only the time of the first acquisition of each block will be used to calculate the delays.

        echo_time
            TE in s
        repetition_time
            TR in s
        t2_prep_echo_times
            Echo times of the three T2 preparation blocks in s
        """
        super().__init__()
        sequence = EPGSequence()
        max_flip_angles_deg = [12.5, 18.75, 25, 25, 25, 12.5, 18.75, 25, 25.0, 25, 12.5, 18.75, 25, 25, 25]
        if len(acquisition_times) == 15:
            block_time = torch.as_tensor(acquisition_times)
        elif len(acquisition_times) == 705:
            block_time = torch.as_tensor(acquisition_times[::47])
        else:
            raise ValueError(f'Invalid number of acquisition times: {len(acquisition_times)}. Must be 15 or 705.')
        for i in range(15):
            block = EPGSequence()
            match i % 5:
                case 0:
                    block.append(InversionBlock(0.020))
                case 1:
                    pass
                case 2:
                    block.append(T2PrepBlock(t2_prep_echo_times[0]))
                case 3:
                    block.append(T2PrepBlock(t2_prep_echo_times[1]))
                case 4:
                    block.append(T2PrepBlock(t2_prep_echo_times[2]))
            flip_angles = torch.deg2rad(
                torch.cat((torch.linspace(4, max_flip_angles_deg[i], 16), torch.full((31,), max_flip_angles_deg[i])))
            )
            block.append(FispBlock(flip_angles, 0.0, tr=repetition_time, te=echo_time))

            if i > 0:
                delay = (block_time[i] - block_time[i - 1]) - block.duration
                if (delay < 0).any():
                    raise ValueError(f'Block {i} would start before the previous block finished. ')
                sequence.append(DelayBlock(delay))
            sequence.append(block)
        self.sequence = sequence.to(device=block_time.device)

    def __call__(self, m0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor) -> tuple[torch.Tensor]:
        """Simulate the Cardiac MR Fingerprinting (cMRF) signal.

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

        Returns
        -------
            Simulated Cardiac MR Fingerprinting signal.
            Shape `(acquisitions ...)`, for example `(acquisitions, *other, coils, z, y, x)` or
            `(acquisitions, samples)` where `acquisitions` corresponds to the different acquisitions
            in the sequence.
        """
        return super().__call__(m0, t1, t2)

    def forward(self, m0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply forward of CardiacFingerprinting.

        .. note::
            Prefer calling the instance of the CardiacFingerprinting as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        parameters = Parameters(m0, t1, t2)
        _, signals = self.sequence(parameters, states=20)
        signal = torch.stack(signals, dim=0)
        return (signal,)
