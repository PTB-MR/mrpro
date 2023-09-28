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
import itertools
from functools import partial
from multiprocessing import Pool

import numpy as np
import psutil
import torch
from einops import repeat
from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi


# Calculate volume/area of voronoi cells
def calc_voronoi_volume_area(points_to_regions, verts, v_cell, idx):
    """Calculate volume/area of voronoi cells."""
    dcf_vol = np.zeros_like(idx, dtype=np.float64)
    for n, id in enumerate(idx):
        cell = v_cell[points_to_regions[id]]
        vertices = [verts[j] for j in cell]
        dcf_vol[n] = ConvexHull(vertices).volume
    return dcf_vol


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
        # 2D and 3D trajectories supported
        if traj.shape[0] not in (2, 3):
            raise ValueError('Only 2D or 3D trajectories supported.')

        # Calculate dcf only for unique k-space positions
        traj_dim = traj.shape
        traj = np.round(traj.numpy(), decimals=15)
        traj = traj.reshape(traj_dim[0], -1)
        traj_unique, inverse, counts = np.unique(traj, return_inverse=True, return_counts=True, axis=1)

        # Especially in 3D, errors in the calculation of the convex hull can occur for edge points. To avoid this,
        # the corner points of a cube bounding box are added here. The bouding box is chosen very large to ensure these
        # edge points of the trajectory can still be accurately detected in the outlier detection further down.
        furthest_corner = np.max(np.abs(traj_unique))
        corner_points = np.array(list(itertools.product([-1, 1], repeat=traj.shape[0]))) * furthest_corner * 10
        traj_unique = np.concatenate((traj_unique, corner_points.transpose()), axis=1)

        # Carry out voronoi tessellation
        vdiagram = Voronoi(traj_unique.transpose())

        # List of regions is now an array so we have fixed indices
        v_cell = np.array(vdiagram.regions, dtype=object)

        # Calculate area (2D) or volume (3D) of each voronoi cell but not the corner points added above
        dcf = np.zeros(len(vdiagram.points) - len(corner_points), dtype=np.float64) - 10

        # Split vertices for different workers (without the added corner points)
        num_cpu = psutil.cpu_count(logical=False)
        idx_cpu = []
        for ind in range(num_cpu):
            idx_cpu.append(np.asarray(range(ind, len(vdiagram.points) - len(corner_points), num_cpu)))

        # Open Pool
        pool = Pool(processes=num_cpu)
        results = pool.starmap(
            partial(calc_voronoi_volume_area, vdiagram.point_region, vdiagram.vertices, v_cell), zip(iter(idx_cpu))
        )
        pool.close()  # shut down the pool

        # Sort in results from different cpus
        for ind in range(num_cpu):
            cidx = idx_cpu[ind]
            dcf[cidx] = results[ind]

        # Get outliers (i.e. voronoi cell which are unbound) and set them to a reasonable value
        # Outliers are defined as values larger than 1.5 * inter quartile range of the values
        # Outliers are set to the average of the 1% largest values.
        dcf_sorted = np.sort(dcf)
        q1, q3 = np.percentile(dcf_sorted, [25, 75])
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        idx_outlier = np.where(dcf > upper_bound)
        dcf_sorted_without_outliers = np.sort(np.delete(dcf, idx_outlier))
        max_idx = int(len(dcf_sorted_without_outliers) * 0.99)
        outlier_val = np.average(dcf_sorted_without_outliers[max_idx:])
        dcf[idx_outlier] = outlier_val

        # Sort dcf values back into the original order (i.e. before calling unique)
        dcf = np.reshape((dcf / counts)[inverse], (1,) + traj_dim[1:])

        return torch.tensor(dcf, dtype=torch.float32)

    @staticmethod
    def _voronoi_dcf_for_each_d4(traj: torch.Tensor) -> torch.Tensor:
        """Loop over all entries in d4 and calculate voronoi dcf.

        Parameters
        ----------
        traj
            torch.Tensor containing k-space points (d4, 2 or 3, k2, k1, k0).
        """
        # Calculate dcf for each dynamic, phase, average...
        dcf = torch.zeros((traj.shape[0], 1) + traj.shape[2:], dtype=torch.float32)
        for ind in range(traj.shape[0]):
            dcf[ind, ...] = DcfData._dcf_using_voronoi(traj[ind, ...])
        return dcf

    @classmethod
    def from_traj_voronoi(cls, traj: torch.Tensor) -> DcfData:
        """Calculate dcf using voronoi approach for 2D or 3D trajectories.

        Parameters
        ----------
        traj
            torch.Tensor containing k-space points (d4, 2 or 3, k2, k1, k0).
        """
        if traj.shape[1] != 2 and traj.shape[1] != 3:
            raise ValueError('Trajectoy has to be 2D or 3D')
        return cls(data=DcfData._voronoi_dcf_for_each_d4(traj))

    @classmethod
    def from_rpe_traj_voronoi(cls, traj: torch.Tensor) -> DcfData:
        """Calculate dcf using voronoi approach for RPE trajectories.

        For RPE the trajectoy is the same for each readout/frequency encoding position (k0), therefore it is
        calculated in 2D for one k0 position and replicated. This is faster than calculating the trajectory in 3D.

        Parameters
        ----------
        traj
            torch.Tensor containing k-space points (d4, 3, k2, k1, k0).
        """
        if traj.shape[1] != 3:
            raise ValueError('Trajectoy has to be 3D')

        # Calculate trajectory in k1-k2 phase encoding plane for one frequency encoding "slice" and then copy to all
        # other frequency encoding positions.
        dcf = DcfData._voronoi_dcf_for_each_d4(traj[:, 1:, :, :, None, 0])
        dcf = repeat(dcf, 'd4 dim k2 k1 k0->d4 dim k2 k1 (k0 nk0)', nk0=traj.shape[4])
        return cls(data=dcf)
