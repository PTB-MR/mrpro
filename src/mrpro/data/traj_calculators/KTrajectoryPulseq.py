"""K-space trajectory from .seq file class."""

from pathlib import Path

import pypulseq as pp
import torch
from einops import rearrange

from mrpro.data.KHeader import KHeader
from mrpro.data.KTrajectoryRawShape import KTrajectoryRawShape
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

    def __call__(self, kheader: KHeader) -> KTrajectoryRawShape:
        """Calculate trajectory from given .seq file and header information.

        Parameters
        ----------
        kheader
           MR raw data header (KHeader) containing required meta data

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

        n_samples = kheader.acq_info.number_of_samples
        n_samples = torch.unique(n_samples)
        if len(n_samples) > 1:
            raise ValueError('We currently only support constant number of samples')
        n_k0 = int(n_samples.item())

        def rescale_and_reshape_traj(k_traj: torch.Tensor, encoding_size: int):
            if encoding_size > 1 and torch.max(torch.abs(k_traj)) > 0:
                k_traj = k_traj * encoding_size / (2 * torch.max(torch.abs(k_traj)))
            else:
                # We force k_traj to be 0 if encoding_size = 1. This is typically the case for kz in 2D sequences.
                # However, it happens that seq.calculate_kspace() returns values != 0 (numerical noise) in such cases.
                k_traj = torch.zeros_like(k_traj)
            return rearrange(k_traj, '(other k0) -> other k0', k0=n_k0)

        # rearrange k-space trajectory to match MRpro convention
        kx = rescale_and_reshape_traj(k_traj_adc[0], kheader.encoding_matrix.x)
        ky = rescale_and_reshape_traj(k_traj_adc[1], kheader.encoding_matrix.y)
        kz = rescale_and_reshape_traj(k_traj_adc[2], kheader.encoding_matrix.z)

        return KTrajectoryRawShape(kz, ky, kx, self.repeat_detection_tolerance)
