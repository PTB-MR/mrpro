"""Radial phase encoding (RPE) trajectory class."""

import torch
from einops import repeat

from mrpro.data.KHeader import KHeader
from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.traj_calculators.KTrajectoryCalculator import KTrajectoryCalculator


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

    def _apply_shifts_between_rpe_lines(self, krad: torch.Tensor, kang_idx: torch.Tensor) -> torch.Tensor:
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
        krad
            k-space positions along each phase encoding line
        kang_idx
            indices of angles to be used for shift calculation

        References
        ----------
        .. [PRI2010] Prieto C, Schaeffter T (2010) 3D undersampled golden-radial phase encoding
        for DCE-MRA using inherently regularized iterative SENSE. MRM 64(2). https://doi.org/10.1002/mrm.22446
        """
        for ind, shift in enumerate(self.shift_between_rpe_lines):
            curr_angle_idx = torch.nonzero(
                torch.fmod(kang_idx, len(self.shift_between_rpe_lines)) == ind,
                as_tuple=True,
            )
            curr_krad = krad[curr_angle_idx]

            # Do not shift the k-space center
            curr_krad += shift * (curr_krad != 0)

            krad[curr_angle_idx] = curr_krad
        return krad

    def _kang(self, kheader: KHeader) -> torch.Tensor:
        """Calculate the angles of the phase encoding lines.

        Parameters
        ----------
        kheader
            MR raw data header (KHeader) containing required meta data

        Returns
        -------
            angles of phase encoding lines
        """
        return repeat(kheader.acq_info.idx.k2 * self.angle, '... k2 k1 -> ... k2 k1 k0', k0=1)

    def _krad(self, kheader: KHeader) -> torch.Tensor:
        """Calculate the k-space locations along the phase encoding lines.

        Parameters
        ----------
        kheader
            MR raw data header (KHeader) containing required meta data

        Returns
        -------
            k-space locations along the phase encoding lines
        """
        krad = (kheader.acq_info.idx.k1 - kheader.encoding_limits.k1.center).to(torch.float32)
        krad = self._apply_shifts_between_rpe_lines(krad, kheader.acq_info.idx.k2)
        return repeat(krad, '... k2 k1 -> ... k2 k1 k0', k0=1)

    def __call__(self, kheader: KHeader) -> KTrajectory:
        """Calculate radial phase encoding trajectory for given KHeader.

        Parameters
        ----------
        kheader
           MR raw data header (KHeader) containing required meta data

        Returns
        -------
            radial phase encoding trajectory for given KHeader
        """
        # Trajectory along readout
        kx = self._kfreq(kheader)

        # Angles of phase encoding lines
        kang = self._kang(kheader)

        # K-space locations along phase encoding lines
        krad = self._krad(kheader)

        kz = krad * torch.sin(kang)
        ky = krad * torch.cos(kang)
        return KTrajectory(kz, ky, kx)
