"""Density compensation data (DcfData) class."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from __future__ import annotations

import dataclasses
from concurrent.futures import ProcessPoolExecutor
from functools import reduce
from itertools import product

import numpy as np
import torch
from numpy.typing import ArrayLike
from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi

from mrpro.data import KTrajectory
from mrpro.utils import smap

UNIQUE_ROUNDING_DECIMALS = 15


def _volume(v: ArrayLike):
    return ConvexHull(v).volume


@dataclasses.dataclass(slots=True, frozen=False)
class DcfData:
    """Density compensation data (DcfData) class."""

    data: torch.Tensor

    @staticmethod
    def _dcf_1d(traj: torch.Tensor) -> torch.Tensor:
        """Calculate sample density compensation function for 1D trajectory.

        Parameters
        ----------
        traj
            k-space positions, 1D tensor
        """
        traj_sorted, inverse, counts = torch.unique(
            torch.round(traj, decimals=UNIQUE_ROUNDING_DECIMALS),
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )

        # For a sorted trajectory: x0 x1 x2 ... xN
        # we assign the point at x1 the area (x1 - x0) / 2  + (x2 - x1) / 2,
        # this is done by central differences (-1,0,1).
        # For the edges, we append/prepend the values afterwards, such that:
        # we assign the point at x0 the area (x1 - x0) / 2 + (x1 - x0) / 2, and
        # we assign the point at xN the area (xN - xN-1) / 2 + (xN - xN-1) / 2.

        kernel = torch.tensor([-1 / 2, 0, 1 / 2], dtype=torch.float32, device=traj.device).reshape(1, 1, 3)

        if (elements := len(traj_sorted)) >= 3:
            central_diff = torch.nn.functional.conv1d(traj_sorted[None, None, :], kernel)[0, 0]
            first = traj_sorted[1] - traj_sorted[0]
            last = traj_sorted[-1] - traj_sorted[-2]
            central_diff = torch.cat((first[None], central_diff, last[None]), -1)
        elif elements == 2:
            diff = traj_sorted[1] - traj_sorted[0]
            central_diff = torch.cat((diff[None], diff[None]), -1)
        else:
            central_diff = torch.ones_like(traj_sorted)

        # Repeated points are reduced by the number of repeats
        dcf = torch.nan_to_num(central_diff / counts)[inverse]
        return dcf

    @staticmethod
    def _dcf_2d3d_voronoi(traj: torch.Tensor) -> torch.Tensor:
        """Calculate sample density compensation function using voronoi method.

        Points at the edge of k-space are detected as outliers and assigned the
        area of the 1% largest dcf values.

        Parameters
        ----------
        traj
            k-space positions (2 or 3, k2, k1, k0)

        Returns
        -------
            density compensation values (1, k2, k1, k0)
        """
        # 2D and 3D trajectories supported
        dim = traj.shape[0]
        if dim not in (2, 3):
            raise ValueError(f'Only 2D or 3D trajectories supported, not {dim}D.')

        # Calculate dcf only for unique k-space positions
        traj_dim = traj.shape
        traj_numpy = np.round(traj.cpu().numpy(), decimals=UNIQUE_ROUNDING_DECIMALS)
        traj_numpy = traj_numpy.reshape(dim, -1)
        traj_unique, inverse, counts = np.unique(traj_numpy, return_inverse=True, return_counts=True, axis=1)

        # Especially in 3D, errors in the calculation of the convex hull can occur for edge points. To avoid this,
        # the corner points of a cube bounding box are added here. The bounding box is chosen very large to ensure these
        # edge points of the trajectory can still be accurately detected in the outlier detection further down.
        furthest_corner = np.max(np.abs(traj_unique))
        corner_points = np.array(list(product([-1, 1], repeat=dim))) * furthest_corner * 10
        traj_extendend = np.concatenate((traj_unique, corner_points.transpose()), axis=1)

        # Carry out voronoi tessellation
        vdiagram = Voronoi(traj_extendend.transpose())
        regions = [vdiagram.regions[r] for r in vdiagram.point_region[: -len(corner_points)]]  # Ignore corner points
        vertices = [vdiagram.vertices[region] for region in regions]

        if dim == 2:
            # Shoelace equation for 2d
            dcf = np.array([np.abs(np.cross(v[:-1], v[1:]).sum(0) + np.cross(v[-1], v[0])) / 2 for v in vertices])

        else:
            # Calculate volume/area of voronoi cells using processes, as this is a very time-consuming operation
            # and ConvexHull is singlethreaded and does not seem to drop the GIL
            # TODO: this could maybe be made faster as the polyhedrons are known to be convex
            future = ProcessPoolExecutor(max_workers=torch.get_num_threads()).map(_volume, vertices, chunksize=100)
            dcf = np.array(list(future))

        # Get outliers (i.e. voronoi cell which are unbound) and set them to a reasonable value
        # Outliers are defined as values larger than 1.5 * inter quartile range of the values
        # Outliers are set to the average of the 1% largest values.
        dcf_sorted = np.sort(dcf)
        q1, q3 = np.percentile(dcf_sorted, [25, 75])
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        idx_outliers = np.nonzero(dcf > upper_bound)
        num_outliers = len(idx_outliers[0])
        high_values_start = int(0.99 * (len(dcf_sorted) - num_outliers))
        high_values_end = len(dcf_sorted) - num_outliers  # this works also for num_outliers==0
        fill_value = np.average(dcf_sorted[high_values_start:high_values_end])
        dcf[idx_outliers] = fill_value

        # Sort dcf values back into the original order (i.e. before calling unique)
        dcf = np.reshape((dcf / counts)[inverse], traj_dim[1:])

        return torch.tensor(dcf, dtype=torch.float32, device=traj.device)

    @classmethod
    def from_traj_voronoi(cls, traj: KTrajectory) -> DcfData:
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
                dcfs.append(smap(DcfData._dcf_1d, non_singleton_ks.pop(), (dim,)))
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
            dcfs.append(smap(DcfData._dcf_2d3d_voronoi, torch.stack(list(ks_needing_voronoi), -4), 4))

        if dcfs:
            # Multiply all dcfs together
            dcf = reduce(torch.mul, dcfs)
        else:
            # Edgecase: Only singleton spatial dimensions
            dcf = torch.ones(*traj.broadcasted_shape[-3:], 1, 1, 1, device=traj.kx.device)

        return cls(data=dcf)
