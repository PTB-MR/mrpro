"""Radial phase encoding (RPE) trajectory class."""

import torch

from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.traj_calculators.KTrajectoryCalculator import KTrajectoryCalculator
from mrpro.utils.reshape import unsqueeze_tensors_left


class KTrajectoryRpe(KTrajectoryCalculator):
    """Radial phase encoding trajectory.

    Frequency encoding along kx is carried out in a standard Cartesian way. The phase encoding points along ky and kz
    are positioned along radial lines [BOU2009]_ [KOL2014]_.

    References
    ----------
    .. [BOU2009] Boubertakh R, Schaeffter T (2009) Whole-heart imaging using undersampled radial phase encoding (RPE)
        and iterative sensitivity encoding (SENSE) reconstruction. MRM 62(5) https://doi.org/10.1002/mrm.22102

    .. [KOL2014] Kolbitsch C, Schaeffter T (2014) A 3D MR-acquisition scheme for nonrigid bulk motion correction
        in simultaneous PET-MR. Medical Physics 41(8) https://doi.org/10.1118/1.4890095
    """

    def __init__(self, angle: float, shift_between_rpe_lines: tuple | torch.Tensor = (0, 0.5, 0.25, 0.75)) -> None:
        """Initialize KTrajectoryRpe.

        Parameters
        ----------
        angle
            angle in rad between two radial phase encoding lines
        shift_between_rpe_lines
            shift between radial phase encoding lines along the radial direction.
            See _apply_shifts_between_rpe_lines() for more details
        """
        super().__init__()

        self.angle: float = angle
        self.shift_between_rpe_lines: torch.Tensor = torch.as_tensor(shift_between_rpe_lines)

    def _apply_shifts_between_rpe_lines(self, radial: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Shift radial phase encoding lines relative to each other.

        Example: shift_between_rpe_lines = [0, 0.5, 0.25, 0.75] leads to a shift of the 0th line by 0,
        the 1st line by 0.5, the 2nd line by 0.25, the 3rd line by 0.75, the 4th line by 0, the 5th line
        by 0.5 and so on. Phase encoding points in k-space center are not shifted [PRI2010]_.

        Line #          k-space points before shift             k-space points after shift
        0               +    +    +    +    +    +    +         +    +    +    +    +    +    +
        1               +    +    +    +    +    +    +           +    +    +  +      +    +    +
        2               +    +    +    +    +    +    +          +    +    +   +     +    +    +
        3               +    +    +    +    +    +    +            +    +    + +       +    +    +
        4               +    +    +    +    +    +    +         +    +    +    +    +    +    +
        5               +    +    +    +    +    +    +           +    +    +  +      +    +    +

        Parameters
        ----------
        radial
            k-space positions along each phase encoding line, to be shifted
        idx
            indices used for shift calculation

        Returns
        -------
            shifted radial k-space positions

        References
        ----------
        .. [PRI2010] Prieto C, Schaeffter T (2010) 3D undersampled golden-radial phase encoding
        for DCE-MRA using inherently regularized iterative SENSE. MRM 64(2). https://doi.org/10.1002/mrm.22446
        """
        radial, idx = torch.broadcast_tensors(radial, idx)
        radial = radial.clone()
        not_center = ~torch.isclose(radial, torch.tensor(0.0))
        for ind, shift in enumerate(self.shift_between_rpe_lines):
            current_mask = (idx % len(self.shift_between_rpe_lines)) == ind
            # k-space center should not be shifted
            current_mask &= not_center
            radial[current_mask] += shift

        return radial

    def __call__(
        self,
        *,
        n_k0: int,
        k0_center: int | torch.Tensor,
        k1_idx: torch.Tensor,
        k1_center: int | torch.Tensor,
        k2_idx: torch.Tensor,
        reversed_readout_mask: torch.Tensor | None = None,
        **_,
    ) -> KTrajectory:
        """Calculate radial phase encoding trajectory for given header information.

        Parameters
        ----------
        n_k0
            number of samples in k0
        k0_center
            position of k-space center in k0
        k1_idx
            indices of k1
        k1_center
            position of k-space center in k1
        k2_idx
            indices of k2
        reversed_readout_mask
            boolean tensor indicating reversed readout

        Returns
        -------
            radial phase encoding trajectory for given header information
        """
        angles = k2_idx * self.angle

        radial = (k1_idx - k1_center).to(torch.float32)
        radial = self._apply_shifts_between_rpe_lines(radial, k2_idx)

        kz = radial * torch.sin(angles)
        ky = radial * torch.cos(angles)
        kx = self._readout(n_k0, k0_center, reversed_readout_mask=reversed_readout_mask)
        kz, ky, kx = unsqueeze_tensors_left(kz, ky, kx, ndim=5)
        return KTrajectory(kz, ky, kx)
