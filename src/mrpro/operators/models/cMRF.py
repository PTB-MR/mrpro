"""Cardiac MR Fingerprinting signal model (EPG)."""

import torch

from mrpro.operators.models.EPG import DelayBlock, EPGSequence, EPGSignalModel, FispBlock, InversionBlock, T2PrepBlock
from mrpro.operators.SignalModel import SignalModel


class CardiacFingerprinting(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor]):
    """Cardiac MR Fingerprinting signal model.

    This model simulates a cardiac MR fingerprinting sequence as described in Hamilton  et al. 'MR fingerprinting
    for rapid quantification of myocardial T1, T2, and proton spin density'. Magn. Reson. Med. 77, 1446-1458 (2017)
    [http://doi.wiley.com/10.1002/mrm.26668] using the extended phase graph (`~mrpro.operators.models.EPG`) formalism.

    It is a four-fold repetition of

                Block 0                   Block 1                   Block 2                     Block 3
       R-peak                   R-peak                    R-peak                    R-peak                    R-peak
    ---|-------------------------|-------------------------|-------------------------|-------------------------|-----

            [INV TI=20ms][ACQ]                     [ACQ]     [T2-prep TE=40ms][ACQ]    [T2-prep TE=80ms][ACQ]


    ```{note}
    This model is on purpose not flexible in all design choices. Instead, consider writing a custom
    `~mrpro.operators.SignalModel` based on this implementation if you need to simulate a different sequence.
    ```
    """

    def __init__(self, acquisition_times: torch.Tensor, echo_time: float) -> None:
        """Initialize the Cardiac MR Fingerprinting signal model.

        Parameters
        ----------
        acquisition_times
            Acquisition times in s
            Times of all acquisitions. Only the first acquisition time of each block is used to determine the
            heart rate dependent delays.
        echo_time
            TE in s

        Returns
        -------
            Cardiac MR Fingerprinting signal with the different acquisitions in the first dimension.
        """
        super().__init__()
        sequence = EPGSequence()
        max_flip_angles_deg = [
            12.5,
            18.75,
            25.0,
            25.0,
            25.0,
            12.5,
            18.75,
            25.0,
            25.0,
            25.0,
            12.5,
            18.75,
            25.0,
            25.0,
            25.0,
        ]
        if len(acquisition_times) != 705:
            raise ValueError(f'Expected 705 acquisition times. Got {acquisition_times.size(-1)}')
        block_time = acquisition_times[::47]  # Start time of each acquisition block. Varies due to heart rate.
        for i in range(block_time.size(-1)):
            block = EPGSequence()
            match i % 5:
                case 0:
                    block.append(InversionBlock(0.020))
                case 1:
                    pass
                case 2:
                    block.append(T2PrepBlock(0.03))
                case 3:
                    block.append(T2PrepBlock(0.05))
                case 4:
                    block.append(T2PrepBlock(0.1))
            flip_angles = torch.deg2rad(
                torch.cat((torch.linspace(4, max_flip_angles_deg[i], 16), torch.full((31,), max_flip_angles_deg[i])))
            )
            block.append(FispBlock(flip_angles, 0.0, tr=0.01, te=echo_time))
            if i > 0:
                delay = (block_time[i] - block_time[i - 1]) - block.duration
                sequence.append(DelayBlock(delay))
            sequence.append(block)
        self.model = EPGSignalModel(sequence, n_states=20)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor, m0: torch.Tensor) -> tuple[torch.Tensor]:
        """Simulate the Cardiac MR Fingerprinting signal.

        Parameters
        ----------
        t1
            T1 relaxation time [s]
        t2
            T2 relaxation time [s]
        m0
            Steady state magnetization (complex)

        Returns
        -------
            Simulated Cardiac MR Fingerprinting signal with the different acquisitions in the first dimension.
        """
        return self.model(t1, t2, m0, None)
