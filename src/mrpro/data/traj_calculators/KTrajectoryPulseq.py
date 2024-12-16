"""K-space trajectory from .seq file class."""

from pathlib import Path

import pypulseq as pp
import torch
from einops import rearrange

from mrpro.data.KTrajectoryRawShape import KTrajectoryRawShape
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
    ) -> KTrajectoryRawShape:
        """Calculate trajectory from given .seq file and header information.

        Parameters
        ----------
        n_k0
            number of samples in k0
        encoding_matrix
            encoding matrix

        Returns
        -------
            trajectory of type KTrajectoryRawShape
        """
        # create PyPulseq Sequence object and read .seq file
        seq = pp.Sequence()
        seq.read(file_path=str(self.seq_path))

        # calculate k-space trajectory using PyPulseq
        k_traj_adc_numpy, _, _, _, _ = seq.calculate_kspace()
        k_traj_adc = torch.tensor(k_traj_adc_numpy, dtype=torch.float32)

        def reshape(k_traj: torch.Tensor, encoding_size: int) -> torch.Tensor:
            max_value_range = 2 * torch.max(torch.abs(k_traj))
            if max_value_range > 1e-9 and encoding_size > 1:
                k_traj = k_traj * encoding_size / max_value_range
            else:
                # If encoding matrix is 1, we force k_traj to be 0. We assume here that the values are
                # numerical noise returned by pulseq, not real trajectory values
                # even if pulseq returned some numerical noise. Also we avoid division by zero.
                k_traj = torch.zeros_like(k_traj)
            return rearrange(k_traj, '(other k0) -> other k0', k0=n_k0)

        # rearrange k-space trajectory to match MRpro convention
        kx = reshape(k_traj_adc[0], encoding_matrix.x)
        ky = reshape(k_traj_adc[1], encoding_matrix.y)
        kz = reshape(k_traj_adc[2], encoding_matrix.z)

        return KTrajectoryRawShape(kz, ky, kx, self.repeat_detection_tolerance)
