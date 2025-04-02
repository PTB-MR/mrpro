"""Density compensation data (DcfData) class."""

from functools import reduce
from typing import TYPE_CHECKING

import torch
from typing_extensions import Self

from mrpro.algorithms.dcf.dcf_voronoi import dcf_1d, dcf_2d3d_voronoi
from mrpro.data.Dataclass import Dataclass
from mrpro.data.KTrajectory import KTrajectory
from mrpro.utils import smap

if TYPE_CHECKING:
    from mrpro.operators.DensityCompensationOp import DensityCompensationOp


class DcfData(Dataclass):
    """Density compensation data (DcfData) class."""

    data: torch.Tensor
    """Density compensation values. Shape `(... other, coils, k2, k1, k0)`"""

    @classmethod
    def from_traj_voronoi(cls, traj: KTrajectory) -> Self:
        """Calculate dcf using voronoi approach for 2D or 3D trajectories.

        Parameters
        ----------
        traj
            Trajectory to calculate the density compensation for. Can be broadcasted or dense.
        """
        dcfs = []

        ks = [traj.kz, traj.ky, traj.kx]
        spatial_dims = (-3, -2, -1)
        ks_needing_voronoi = set()
        for dim in spatial_dims:
            non_singleton_ks = [ax for ax in ks if ax.shape[dim] != 1]
            if len(non_singleton_ks) == 1:
                # Found a dimension with only one non-singleton axes in ks
                # --> Can handle this as a 1D trajectory
                dcfs.append(smap(dcf_1d, non_singleton_ks.pop(), (dim,)))
            elif len(non_singleton_ks) > 0:
                # More than one of the ks is non-singleton
                # --> A full dimension needing voronoi
                ks_needing_voronoi |= set(non_singleton_ks)
            else:
                # A dimension in which each of ks is singleton
                # --> Don't need to do anything
                pass

        if ks_needing_voronoi:
            # Handle full dimensions needing voronoi
            dcfs.append(smap(dcf_2d3d_voronoi, torch.stack(torch.broadcast_tensors(*ks_needing_voronoi), -4), 4))

        if dcfs:
            # Multiply all dcfs together
            dcf = reduce(torch.mul, dcfs)
        else:
            # Edgecase: Only singleton spatial dimensions
            dcf = torch.ones(*traj.shape[-3:], 1, 1, 1, device=traj.kx.device)

        return cls(data=dcf)

    def as_operator(self) -> 'DensityCompensationOp':
        """Create a density compensation operator using a copy of the DCF."""
        from mrpro.operators.DensityCompensationOp import DensityCompensationOp

        return DensityCompensationOp(self.data.clone())
