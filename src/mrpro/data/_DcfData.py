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
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import numpy as np
import torch
from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi

from mrpro.data import KTrajectory


@dataclasses.dataclass(slots=True, frozen=False)
class DcfData:
    """Density compensation data (DcfData) class."""

    data: torch.Tensor

    @staticmethod
    def _dcf_using_voronoi(traj: torch.Tensor) -> torch.Tensor:
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
        UNIQUE_ROUNDING_DECIMALS = 15

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

        # Calculate volume/area of voronoi cells
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
            torch.Tensor containing k-space points (d4, 2 or 3, k2, k1, k0).
        """

        ks = [traj.kx, traj.ky, traj.kz]
        for i, k in enumerate(ks):
            if any(all(k.shape[ax] == 1 for ax in two_axes) for two_axes in [(-1, -2), (-1, -3), (-2, -3)]):
                # Found a direction with at least two singleton dimensions, i.e. we have a 2D trajectory in 3D space
                # Remove this direction from the list of k-space points and calculate dcf for the remaining 2D trajectory
                ks.pop(i)
                new_traj = torch.stack(torch.broadcast_tensors(*ks), 1)
                dcf = torch.stack([DcfData._dcf_using_voronoi(t) for t in new_traj])
                dcf = dcf.expand(traj.broadcasted_shape)
                break
        else:
            # Full 3D trajectory
            dcf = torch.stack([DcfData._dcf_using_voronoi(t) for t in traj.as_tensor(1)])

        return cls(data=dcf)
