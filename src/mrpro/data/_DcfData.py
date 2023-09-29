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
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from itertools import product

import numpy as np
import torch
from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi

from mrpro.data import KTrajectory

UNIQUE_ROUNDING_DECIMALS = 15


@dataclasses.dataclass(slots=True, frozen=False)
class DcfData:
    """Density compensation data (DcfData) class."""

    data: torch.Tensor

    @staticmethod
    def _dcf_1d(traj: torch.Tensor) -> torch.Tensor:
        """Calculate sample density compensation function for 1D trajectory.

        Parameters
        ----------
        traj:
            k-space positions, 1D tensor
        """

        traj_sorted, inverse, counts = torch.unique(
            torch.round(traj, decimals=UNIQUE_ROUNDING_DECIMALS), sorted=True, return_inverse=True, return_counts=True
        )

        # For a sorted trajectory: x0 x1 x2 ... xN
        # We assign the point at x0 the area (x1-x0) / 2 * 2
        # We assign the point at x1 the area (x1-x0) / 2  + (x2-x1) / 2
        # We assign the pint at xN the area (xN-XN-1) / 2 * 2
        # This is done by central differences (-1,0,1). As the be complicated,
        # we just append/prepend the correct values afterwards.

        kernel = torch.tensor([-1 / 2, 0, 1 / 2], dtype=torch.float32, device=traj.device).reshape(1, 1, 3)
        central_diff = torch.nn.functional.conv1d(traj_sorted[None, None, :], kernel)[0, 0]
        first = traj_sorted[1] - traj_sorted[0]
        last = traj_sorted[-1] - traj_sorted[-2]
        central_diff = torch.cat((first[None], central_diff, last[None]), -1)

        # Repeated points are reduced by the number of repeats
        dcf = torch.nan_to_num(1 / (central_diff * counts))[inverse]
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
        traj = np.round(traj.numpy(), decimals=UNIQUE_ROUNDING_DECIMALS)
        traj = traj.reshape(dim, -1)
        traj_unique, inverse, counts = np.unique(traj, return_inverse=True, return_counts=True, axis=1)

        # Especially in 3D, errors in the calculation of the convex hull can occur for edge points. To avoid this,
        # the corner points of a cube bounding box are added here. The bouding box is chosen very large to ensure these
        # edge points of the trajectory can still be accurately detected in the outlier detection further down.
        furthest_corner = np.max(np.abs(traj_unique))
        corner_points = np.array(list(product([-1, 1], repeat=dim))) * furthest_corner * 10
        traj_extendend = np.concatenate((traj_unique, corner_points.transpose()), axis=1)

        # Carry out voronoi tessellation
        vdiagram = Voronoi(traj_extendend.transpose())
        regions = [vdiagram.regions[r] for r in vdiagram.point_region[: -len(corner_points)]]  # Ignore corner points
        vertices = [vdiagram.vertices[region] for region in regions]

        # Calculate volume/area of voronoi cells using threads (ConvexHull is thread safe and drops the GIL)
        future = ThreadPoolExecutor(max_workers=torch.get_num_threads()).map(lambda v: ConvexHull(v).volume, vertices)
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
        fill_value = np.average(dcf_sorted[high_values_start:-num_outliers])
        dcf[idx_outliers] = fill_value

        # Sort dcf values back into the original order (i.e. before calling unique)
        dcf = np.reshape((dcf / counts)[inverse], traj_dim[1:])

        return torch.tensor(dcf, dtype=torch.float32)

    @classmethod
    def from_traj_voronoi(cls, traj: KTrajectory) -> DcfData:
        """Calculate dcf using voronoi approach for 2D or 3D trajectories.

        Parameters
        ----------
        traj
            torch.Tensor containing k-space points (other, 2 or 3, k2, k1, k0).
        """

        ks_needing_voronoi = []
        dcfs = []
        for k in [traj.kz, traj.ky, traj.kx]:
            non_singleton_dims = tuple(dim for dim in (-1, -2, -3) if k.shape[dim] != 1)
            if len(non_singleton_dims) == 1:
                # Found a direction with two singleton dimensions and one non-singleton dimension
                # We can handle this as a 1D trajectory
                dcfs.append(smap(DcfData._dcf_1d, k, non_singleton_dims))
            elif len(non_singleton_dims) == 0:
                # Found a direction with only singleton dimensions, i.e. constant dcf
                dcfs.append(torch.ones_like(k))
            else:
                # A Full Dimension needing voronoi
                ks_needing_voronoi.append(k)
        if ks_needing_voronoi:
            # Any full dimensions needing voronoi
            dcfs.append(smap(DcfData._dcf_2d3d_voronoi, torch.stack(ks_needing_voronoi, -4), 4))
        # Multiply all dcfs together
        dcf = reduce(torch.mul, dcfs)
        return cls(data=dcf)


def smap(
    fun: Callable[[torch.Tensor], torch.Tensor], tensor: torch.Tensor, fun_dims: tuple[int, ...] | int = (-1,)
) -> torch.Tensor:
    """Apply a function to a tensor serially along multiple dimensions.

    The function is applied serially without a batch dimensions.
    Compared to torch.vmap, it works with arbitrary functions, but is slower.

    Parameters
    ----------
    fun
        Function to apply to the tensor.
        Should handle len(fun_dims) dimensions and not change the number of dimensions.
    tensor
        Tensor to apply the function to.
    fun_dims
        Dimensions NOT to be batched / dimensions that are passed to the function
        tuple of dimension indices (negative indices are supported) or an integer
        an integer n means the last n dimensions are passed to the function
    """
    # TODO: Move to utilities
    if isinstance(fun_dims, int):
        # use the last fun_dims dimensions for the function
        moved = tensor
        first_fun_dim = -fun_dims
    else:
        # Move fun_dims to the end
        fun_dims_dst = tuple(range(-len(fun_dims), 0))
        moved = tensor.moveaxis(fun_dims, fun_dims_dst)
        first_fun_dim = fun_dims_dst[0]

    reshaped = moved.flatten(end_dim=first_fun_dim - 1)  # shape: (prod(batch_dims), fun_dim_1, ..., fun_dim_n)
    result_reshaped = torch.stack([fun(x) for x in reshaped])
    result = result_reshaped.reshape(moved.shape[:first_fun_dim] + result_reshaped.shape[1:])

    if not isinstance(fun_dims, int):
        # Move fun_dims back to their original position if we moved them
        result = result.moveaxis(fun_dims_dst, fun_dims)
    return result
