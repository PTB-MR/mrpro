"""2D radial trajectory class."""

import torch

from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.traj_calculators.KTrajectoryCalculator import KTrajectoryCalculator
from mrpro.utils.reshape import unsqueeze_tensors_left


class KTrajectoryRadial2D(KTrajectoryCalculator):
    """Radial 2D trajectory."""

    def __init__(self, angle: float = torch.pi * 0.618034) -> None:
        """Initialize KTrajectoryRadial2D.

        Parameters
        ----------
        angle
            angle in rad between two radial lines
        """
        super().__init__()
        self.angle: float = angle

    def __call__(
        self,
        *,
        n_k0: int,
        k0_center: int | torch.Tensor,
        k1_idx: torch.Tensor,
        reversed_readout_mask: torch.Tensor | None = None,
        **_,
    ) -> KTrajectory:
        """Calculate radial 2D trajectory for given header information.

        Parameters
        ----------
        n_k0
            number of samples in k0 (radial readout)
        k0_center
            position of k-space center in k0
        k1_idx
            indices of k1 (angle)
        reversed_readout_mask
            boolean tensor indicating reversed readout

        Returns
        -------
            radial 2D trajectory for given KHeader
        """
        radial = self._readout(n_k0=n_k0, k0_center=k0_center, reversed_readout_mask=reversed_readout_mask)
        angle = k1_idx * self.angle
        kx = radial * torch.cos(angle)
        ky = radial * torch.sin(angle)
        kz = torch.zeros(1)
        kz, ky, kx = unsqueeze_tensors_left(kz, ky, kx, ndim=5)
        return KTrajectory(kz, ky, kx)
