"""Returns the trajectory saved in an ISMRMRD raw data file."""

import warnings
from collections.abc import Sequence

import ismrmrd
import torch

from mrpro.data.KTrajectory import KTrajectory
from mrpro.utils.reshape import unsqueeze_tensors_at
from mrpro.utils.typing import FileOrPath


class KTrajectoryIsmrmrd:
    """Get trajectory in ISMRMRD raw data file.

    The trajectory in the ISMRMRD raw data file is read out [TRA]_.

    The value range of the trajectory in the ISMRMRD file is not well defined, thus we normalize
    based on the highest value and ensure it is within [-pi, pi].

    References
    ----------
    .. [TRA] ISMRMRD trajectory https://ismrmrd.readthedocs.io/en/latest/mrd_raw_data.html#k-space-trajectory
    """

    def __init__(self, filename: None | FileOrPath = None):
        """Initialize KTrajectoryIsmrmrd.

        Parameters
        ----------
        filename
            Optional file to read the trajectory from. If set to None,
            the trajectory saved inside the acquisitionons of the KData file will be used.
        """
        self.filename = filename

    def __call__(self, acquisitions: Sequence[ismrmrd.Acquisition]) -> KTrajectory:
        """Read out the trajectory from the ISMRMRD data file.

        Parameters
        ----------
        acquisitions:
            list of ismrmrd acquisistions to read from. Needs at least one acquisition.

        Returns
        -------
            trajectory in the shape of the original raw data.
        """
        # Read out the trajectory

        if self.filename is None:
            ktraj_mrd = torch.stack([torch.as_tensor(acq.traj, dtype=torch.float32) for acq in acquisitions])
        else:
            with ismrmrd.File(self.filename, 'r') as file:
                datasets = list(file.keys())
                if len(datasets) == 0:
                    raise ValueError('No datasets found in the ISMRMRD file.')
                elif len(datasets) > 1:
                    warnings.warn('More than one dataset found in the ISMRMRD file. Using the last one.', stacklevel=1)
                ktraj_mrd = torch.stack(
                    [torch.as_tensor(acq.traj, dtype=torch.float32) for acq in file[datasets[-1]].acquisitions]
                )

        if ktraj_mrd.numel() == 0:
            raise ValueError('No trajectory information available in the ISMRMD file.')

        kz = torch.zeros_like(ktraj_mrd[..., 1]) if ktraj_mrd.shape[-1] == 2 else ktraj_mrd[..., 2]
        ky = ktraj_mrd[..., 1]
        kx = ktraj_mrd[..., 0]

        kz, ky, kx = unsqueeze_tensors_at(kz, ky, kx, dim=-2, ndim=5)
        return KTrajectory(kz, ky, kx)
