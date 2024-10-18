"""Returns the trajectory saved in an ISMRMRD raw data file."""

from collections.abc import Sequence

import ismrmrd
import torch

from mrpro.data.KTrajectoryRawShape import KTrajectoryRawShape


class KTrajectoryIsmrmrd:
    """Get trajectory in ISMRMRD raw data file.

    The trajectory in the ISMRMRD raw data file is read out [TRA]_.

    The value range of the trajectory in the ISMRMRD file is not well defined. Here we simple normalize everything
    based on the highest value and ensure it is within [-pi, pi]. The trajectory is in the shape of the unsorted
    raw data.

    References
    ----------
    .. [TRA] ISMRMRD trajectory https://ismrmrd.readthedocs.io/en/latest/mrd_raw_data.html#k-space-trajectory
    """

    def __call__(self, acquisitions: Sequence[ismrmrd.Acquisition]) -> KTrajectoryRawShape:
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
        ktraj_mrd = torch.stack([torch.as_tensor(acq.traj, dtype=torch.float32) for acq in acquisitions])

        if ktraj_mrd.numel() == 0:
            raise ValueError('No trajectory information available in the acquisitions.')

        if ktraj_mrd.shape[2] == 2:
            ktraj = KTrajectoryRawShape(
                kz=torch.zeros_like(ktraj_mrd[..., 1]),
                ky=ktraj_mrd[..., 1],
                kx=ktraj_mrd[..., 0],
            )
        else:
            ktraj = KTrajectoryRawShape(kz=ktraj_mrd[..., 2], ky=ktraj_mrd[..., 1], kx=ktraj_mrd[..., 0])

        return ktraj
