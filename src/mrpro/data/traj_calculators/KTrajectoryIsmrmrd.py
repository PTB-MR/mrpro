"""Returns the trajectory saved in an ISMRMRD raw data file."""

from collections.abc import Sequence

import ismrmrd
import torch

from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.utils.reshape import unsqueeze_at


class KTrajectoryIsmrmrd:
    """Get trajectory in ISMRMRD raw data file.

    Use an instance of this class to tell `mrpro.data.KData.from_file` to read in the trajectory
    from the ISMRMRD file [TRA]_.

    The trajectory will be normalized to fit in the encoding matrix.

    References
    ----------
    .. [TRA] ISMRMRD trajectory https://ismrmrd.readthedocs.io/en/latest/mrd_raw_data.html#k-space-trajectory
    """

    def __call__(self, acquisitions: Sequence[ismrmrd.Acquisition], encoding_matrix: SpatialDimension) -> KTrajectory:
        """Read out the trajectory from the ISMRMRD data file.

        Parameters
        ----------
        acquisitions
            list of ismrmrd acquisistions to read from. Needs at least one acquisition.
        encoding_matrix
            encoding matrix, used to normalize the trajectory.

        Returns
        -------
            trajectory in the shape of the original raw data.
        """
        traj = torch.stack([torch.as_tensor(acq.traj, dtype=torch.float32) for acq in acquisitions])

        if not traj.numel():
            raise ValueError('No trajectory information available in the ISMRMD file.')

        if traj.shape[-1] != 3:
            zero = torch.zeros_like(traj[..., :1])
            traj = torch.cat([traj, *([zero] * (3 - traj.shape[-1]))], dim=-1)

        traj = unsqueeze_at(traj, dim=-3, n=5 - traj.ndim + 1)  # +1 due to stack dim

        return KTrajectory.from_tensor(traj, stack_dim=-1, axes_order='xyz', scaling_matrix=encoding_matrix)
