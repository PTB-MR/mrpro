"""K-space trajectory from .seq file class."""

from pathlib import Path

import torch
from einops import repeat

from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.data.traj_calculators.KTrajectoryCalculator import KTrajectoryCalculator


class KTrajectoryPulseq(KTrajectoryCalculator):
    """Trajectory from .seq file."""

    def __init__(self, seq_path: str | Path, repeat_detection_tolerance: None | float = 1e-3) -> None:
        """Initialize KTrajectoryPulseq.

        Parameters
        ----------
        seq_path
            absolute path to .seq file
        repeat_detection_tolerance
            tolerance for repeat detection when creating KTrajectory
        """
        super().__init__()
        self.seq_path = seq_path
        self.repeat_detection_tolerance = repeat_detection_tolerance

    def __call__(
        self,
        *,
        n_k0: int,
        encoding_matrix: SpatialDimension,
        **_,
    ) -> KTrajectory:
        """Calculate trajectory from given .seq file and header information.

        Parameters
        ----------
        n_k0
            number of samples in k0
        encoding_matrix
            encoding matrix

        Returns
        -------
            trajectory of type KTrajectory
        """
        from pypulseq import Sequence

        # calculate k-space trajectory using PyPulseq
        seq = Sequence()
        seq.read(file_path=str(self.seq_path))
        k_traj_adc_numpy = seq.calculate_kspace()[0]
        k_traj_adc = torch.tensor(k_traj_adc_numpy, dtype=torch.float32)
        k_traj_reshaped = repeat(k_traj_adc, 'xyz (other k0) -> xyz other coils k2 k1 k0', coils=1, k2=1, k1=1, k0=n_k0)

        return KTrajectory.from_tensor(
            k_traj_reshaped,
            axes_order='xyz',
            scaling_matrix=encoding_matrix,
            repeat_detection_tolerance=self.repeat_detection_tolerance,
        )
