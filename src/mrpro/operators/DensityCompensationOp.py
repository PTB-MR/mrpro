"""Class for Density Compensation Operator."""

from functools import reduce

import torch
from typing_extensions import Self

from mrpro.algorithms.dcf.dcf_voronoi import dcf_1d, dcf_2d3d_voronoi
from mrpro.data import KTrajectory
from mrpro.operators.EinsumOp import EinsumOp
from mrpro.utils import smap
from mrpro.utils.reduce_repeat import reduce_repeat


class DensityCompensationOp(EinsumOp):
    """Density Compensation Operator."""

    def __init__(self, dcf: torch.Tensor) -> None:
        """Initialize a Density Compensation Operator.

        Parameters
        ----------
        dcf
           Density compensation data
        """
        super().__init__(dcf, '...,... -> ...')

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply density compensation to k-space data.

        This operator performs an element-wise multiplication of the input k-space data
        with the density compensation factors (DCF).

        Parameters
        ----------
        x
            Input k-space data.

        Returns
        -------
            Density compensated k-space data.
        """
        return super().__call__(x)

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

        return cls(dcf=reduce_repeat(dcf))
