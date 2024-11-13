"""A pytorch implementation of scipy.spatial.transform.Rotation.

A container for proper and improper Rotations, that can be created from quaternions, euler angles, rotation vectors,
rotation matrices, etc, can be applied to torch.Tensors and SpatialDimensions, multiplied, and can be converted
to quaternions, euler angles, etc.

see also https://github.com/scipy/scipy/blob/main/scipy/spatial/transform/_rotation.pyx
"""

# based on Scipy implementation, which has the following copyright:
# Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Callable, Sequence
from typing import Literal, cast

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from scipy._lib._util import check_random_state
from typing_extensions import Self, Unpack, overload

from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.utils.typing import NestedSequence, TorchIndexerType
from mrpro.utils.vmf import sample_vmf

AXIS_ORDER = 'zyx'  # This can be modified
QUAT_AXIS_ORDER = AXIS_ORDER + 'w'  # Do not modify
assert QUAT_AXIS_ORDER[:3] == AXIS_ORDER, 'Quaternion axis order has to match axis order'  # noqa: S101


def _compose_quaternions_single(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Calculate p * q."""
    cross = torch.linalg.cross(p[:3], q[:3])
    product = torch.stack(
        (
            p[3] * q[0] + q[3] * p[0] + cross[0],
            p[3] * q[1] + q[3] * p[1] + cross[1],
            p[3] * q[2] + q[3] * p[2] + cross[2],
            p[3] * q[3] - p[0] * q[0] - p[1] * q[1] - p[2] * q[2],
        ),
        0,
    )
    return product


def _compose_quaternions(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Calculate p * q, with p and q batched quaternions."""
    p, q = torch.broadcast_tensors(p, q)
    product = torch.vmap(_compose_quaternions_single)(p.reshape(-1, 4), q.reshape(-1, 4)).reshape(p.shape)
    return product


def _canonical_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert to canonical form, i.e. positive w."""
    x, y, z, w = (quaternion[..., QUAT_AXIS_ORDER.index(axis)] for axis in 'xyzw')
    needs_inversion = (w < 0) | ((w == 0) & ((x < 0) | ((x == 0) & ((y < 0) | ((y == 0) & (z < 0))))))
    canonical_quaternion = torch.where(needs_inversion.unsqueeze(-1), -quaternion, quaternion)
    return canonical_quaternion


def _matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert matrix to quaternion."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f'Invalid rotation matrix shape {matrix.shape}.')

    batch_shape = matrix.shape[:-2]
    # matrix elements
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.flatten(start_dim=-2), -1)
    # q,r,s are some permutation of x,y,z
    qrsw = torch.nn.functional.relu(
        torch.stack(
            [
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
                1.0 + m00 + m11 + m22,
            ],
            dim=-1,
        )
    )
    q, r, s, w = qrsw.unbind(-1)
    # all these are the same except in edge cases.
    # we will choose the one that is most numerically stable.
    # we calculate all choices as this is faster
    candidates = torch.stack(
        (
            *(q, m10 + m01, m02 + m20, m21 - m12),
            *(m10 + m01, r, m12 + m21, m02 - m20),
            *(m20 + m02, m21 + m12, s, m10 - m01),
            *(m21 - m12, m02 - m20, m10 - m01, w),
        ),
        dim=-1,
    ).reshape(*batch_shape, 4, 4)
    # now we make the choice.
    # the choice will not influence the gradients.
    choice = qrsw.argmax(dim=-1)
    quaternion = candidates.take_along_dim(choice[..., None, None], -2).squeeze(-2) / (
        qrsw.take_along_dim(choice[..., None], -1).sqrt() * 2
    )
    return quaternion


def _make_elementary_quat(axis: str, angle: torch.Tensor):
    """Make a quaternion for the rotation around one of the axes."""
    quat = torch.zeros(*angle.shape, 4, device=angle.device, dtype=angle.dtype)
    axis_index = QUAT_AXIS_ORDER.index(axis)
    w_index = QUAT_AXIS_ORDER.index('w')
    quat[..., w_index] = torch.cos(angle / 2)
    quat[..., axis_index] = torch.sin(angle / 2)
    return quat


def _quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to rotation matrix."""
    # use same order for quaternions as for matrix. this saves two index lookups.
    # we use q, r, s for a permutation of x, y, z
    # as this function will be used for the application of the rotatoin matrix, it should be fast.
    q, r, s, w = quaternion.unbind(-1)
    qq = q.square()
    rr = r.square()
    ss = s.square()
    ww = w.square()
    qr = q * r
    sw = s * w
    qs = q * s
    rw = r * w
    rs = r * s
    qw = q * w

    matrix = torch.stack(
        (
            *(qq - rr - ss + ww, 2 * (qr - sw), 2 * (qs + rw)),
            *(2 * (qr + sw), -qq + rr - ss + ww, 2 * (rs - qw)),
            *(2 * (qs - rw), 2 * (rs + qw), -qq - rr + ss + ww),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)
    return matrix


def _quaternion_to_axis_angle(quaternion: torch.Tensor, degrees: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert quaternion to rotation axis and angle.

    Parameters
    ----------
    quaternion
        The batched quaternions, shape (..., 4)
    degrees
        If True, the angle is returned in degrees, otherwise in radians.

    Returns
    -------
    axis
        The rotation axis, shape (..., 3)
    angle
        The rotation angle, shape (...)
    """
    quaternion = _canonical_quaternion(quaternion)
    angle = 2 * torch.atan2(torch.linalg.vector_norm(quaternion[..., :3], dim=-1), quaternion[..., 3])
    axis = quaternion[..., :3] / torch.linalg.vector_norm(quaternion[..., :3], dim=-1, keepdim=True)
    if degrees:
        angle = torch.rad2deg(angle)
    return axis, angle


def _quaternion_to_euler(quaternion: torch.Tensor, seq: str, extrinsic: bool):
    """Convert quaternion to euler angles.

    Parameters
    ----------
    quaternion
        The batched quaternions
    seq
        The axes sequence, lower case. For example 'xyz'
    extrinsic
        If the rotations are extrinsic (True) or intrinsic (False)
    """
    # The algorithm assumes extrinsic frame transformations. The algorithm
    # in the paper is formulated for rotation quaternions, which are stored
    # directly by Rotation.
    # Adapt the algorithm for our case by reversing both axis sequence and
    # angles for intrinsic rotations when needed

    if not extrinsic:
        seq = seq[::-1]
    q, r, s = (QUAT_AXIS_ORDER.index(axis) for axis in seq)  # one of x,y,z
    w = QUAT_AXIS_ORDER.index('w')

    # proper angles, with first and last axis the same
    if symmetric := q == s:
        s = 3 - q - r  # get third axis

    # Check if permutation is even (+1) or odd (-1)
    sign = (q - r) * (r - s) * (s - q) // 2

    if symmetric:
        a = quaternion[..., w]
        b = quaternion[..., q]
        c = quaternion[..., r]
        d = quaternion[..., s] * sign
    else:
        a = quaternion[..., w] - quaternion[..., r]
        b = quaternion[..., q] + quaternion[..., s] * sign
        c = quaternion[..., r] + quaternion[..., w]
        d = quaternion[..., s] * sign - quaternion[..., q]

    # Compute angles
    angles_1 = 2 * torch.atan2(torch.hypot(c, d), torch.hypot(a, b))
    half_sum = torch.atan2(b, a)
    half_diff = torch.atan2(d, c)

    angles_0 = half_sum - half_diff
    angles_2 = half_sum + half_diff

    if not symmetric:
        angles_2 *= sign
        angles_1 -= torch.pi / 2
    if not extrinsic:
        # flip first and last rotation
        angles_2, angles_0 = angles_0, angles_2

    # Check if angles_1 is equal to is 0 (case=1) or pi (case=2), causing a singularity,
    # i.e. a gimble lock. case=0 is the normal.
    case = 1 * (torch.abs(angles_1) <= 1e-7) + 2 * (torch.abs(angles_1 - torch.pi) <= 1e-7)
    # if Gimbal lock, sett last angle to 0 and use 2 * half_sum / 2 * half_diff for first angle.
    angles_2 = (case == 0) * angles_2
    angles_0 = (
        (case == 0) * angles_0 + (case == 1) * 2 * half_sum + (case == 2) * 2 * half_diff * (-1 if extrinsic else 1)
    )

    angles = torch.stack((angles_0, angles_1, angles_2), -1)
    angles += (angles < -torch.pi) * 2 * torch.pi
    angles -= (angles > torch.pi) * 2 * torch.pi
    return angles


def _align_vectors(
    a: torch.Tensor,
    b: torch.Tensor,
    weights: torch.Tensor,
    return_sensitivity: bool = False,
    allow_improper: bool = False,
):
    """Estimate a rotation to optimally align two sets of vectors."""
    n_vecs = a.shape[0]
    if a.shape != b.shape:
        raise ValueError(f'Expected inputs to have same shapes, got {a.shape} and {b.shape}')
    if a.shape[-1] != 3:
        raise ValueError(f'Expected inputs to have shape (..., 3), got {a.shape} and {b.shape}')
    if weights.shape != (n_vecs,) or (weights < 0).any():
        raise ValueError(f'Invalid weights: expected shape ({n_vecs},) with non-negative values')
    if (a.norm(dim=-1) < 1e-6).any() or (b.norm(dim=-1) < 1e-6).any():
        raise ValueError('Cannot align zero length primary vectors')
    dtype = torch.result_type(a, b)
    # we require double precision for the calculations to match scipy results
    weights = weights.double()
    a = a.double()
    b = b.double()

    inf_mask = torch.isinf(weights)
    if inf_mask.sum() > 1:
        raise ValueError('Only one infinite weight is allowed')

    if inf_mask.any() or n_vecs == 1:
        # special case for one vector pair or one infinite weight

        if return_sensitivity:
            raise ValueError('Cannot return sensitivity matrix with an infinite weight or one vector pair')

        a_primary, b_primary = (a[0], b[0]) if n_vecs == 1 else (a[inf_mask][0], b[inf_mask][0])
        a_primary, b_primary = F.normalize(a_primary, dim=0), F.normalize(b_primary, dim=0)
        cross = torch.linalg.cross(b_primary, a_primary, dim=0)
        angle = torch.atan2(torch.norm(cross), torch.dot(a_primary, b_primary))
        rot_primary = _axisangle_to_matrix(cross, angle)

        if n_vecs == 1:
            return rot_primary.to(dtype), torch.tensor(0.0, device=a.device, dtype=dtype)

        a_secondary, b_secondary = a[~inf_mask], b[~inf_mask]
        sec_w = weights[~inf_mask]
        rot_sec_b = (rot_primary @ b_secondary.T).T
        sin_term = torch.einsum('ij,j->i', torch.linalg.cross(rot_sec_b, a_secondary, dim=1), a_primary)
        cos_term = torch.einsum('ij,ij->i', rot_sec_b, a_secondary) - torch.einsum(
            'ij,j->i', rot_sec_b, a_primary
        ) * torch.einsum('ij,j->i', a_secondary, a_primary)

        phi = torch.atan2((sec_w * sin_term).sum(), (sec_w * cos_term).sum())
        rot_secondary = _axisangle_to_matrix(a_primary, phi)
        rot_optimal = rot_secondary @ rot_primary
        rssd_w = weights.clone()
        rssd_w[inf_mask] = 0
        est_a = (rot_optimal @ b.T).T
        rssd = torch.sqrt(torch.sum(rssd_w * torch.sum((a - est_a) ** 2, dim=1)))
        return rot_optimal.to(dtype), rssd.to(dtype)

    corr_mat = torch.einsum('i j, i k, i -> j k', a, b, weights)
    u, s, vt = cast(tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.linalg.svd(corr_mat))
    if s[1] + s[2] < 1e-16 * s[0]:
        warnings.warn('Optimal rotation is not uniquely or poorly defined for the given sets of vectors.', stacklevel=2)

    if (u @ vt).det() < 0 and not allow_improper:
        u[:, -1] *= -1

    rot_optimal = (u @ vt).to(dtype)
    rssd = ((weights * (b**2 + a**2).sum(dim=1)).sum() - 2 * s.sum()).clamp_min(0.0).sqrt().to(dtype)

    if return_sensitivity:
        zeta = (s[0] + s[1]) * (s[1] + s[2]) * (s[2] + s[0])
        kappa = s[0] * s[1] + s[1] * s[2] + s[2] * s[0]
        sensitivity = (
            weights.mean() / zeta * (kappa * torch.eye(3, device=a.device, dtype=torch.float64) + corr_mat @ corr_mat.T)
        ).to(dtype)
        return rot_optimal, rssd, sensitivity

    return rot_optimal, rssd


def _axisangle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Compute a rotation matrix using Rodrigues' rotation formula."""
    axis = F.normalize(axis, dim=-1, eps=1e-6)
    cos, sin = torch.cos(angle), torch.sin(angle)
    t = 1 - cos
    q, r, s = axis.unbind(-1)
    matrix = rearrange(
        torch.stack(
            [
                t * q * q + cos,
                t * q * r - s * sin,
                t * q * s + r * sin,
                t * q * r + s * sin,
                t * r * r + cos,
                t * r * s - q * sin,
                t * q * s - r * sin,
                t * r * s + q * sin,
                t * s * s + cos,
            ],
            dim=-1,
        ),
        '... (row col) -> ... row col',
        row=3,
    )
    return matrix


class Rotation(torch.nn.Module):
    """A container for Rotations.

    A pytorch implementation of scipy.spatial.transform.Rotation.
    For more information see the scipy documentation:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html

    Differences compared to scipy.spatial.transform.Rotation:

    - torch.nn.Module based, the quaternions are a Parameter
    - not all features are implemented. Notably, mrp, davenport, and reduce are missing.
    - arbitrary number of batching dimensions
    - support for improper rotations (rotoinversion), i.e., rotations with an coordinate inversion
        or a reflection about a plane perpendicular to the rotation axis.
    """

    def __init__(
        self,
        quaternions: torch.Tensor | NestedSequence[float],
        normalize: bool = True,
        copy: bool = True,
        inversion: torch.Tensor | NestedSequence[bool] | bool = False,
        reflection: torch.Tensor | NestedSequence[bool] | bool = False,
    ) -> None:
        """Initialize a new Rotation.

        Instead of calling this method, also consider the different ``from_*`` class methods to construct a Rotation.

        Parameters
        ----------
        quaternions
            Rotatation quaternions. If these requires_grad, the resulting Rotation will require gradients
        normalize
            If the quaternions should be normalized. Only disable if you are sure the quaternions are already
            normalized.
            Will keep a possible negative w to represent improper rotations.
        copy
            Always ensure that a copy of the quaternions is created. If both normalize and copy are False,
            the quaternions Parameter of this instance will be a view if the quaternions passed in.
        inversion
            If the rotation should contain an inversion of the coordinate system, i.e. a reflection of all three axes,
            resulting in a rotoinversion (improper rotation).
            If a boolean tensor is given, it should broadcast with the quaternions.
        reflection
            If the rotation should contain a reflection about a plane perpendicular to the rotation axis.
            This will result in a rotoflexion (improper rotation).
            If a boolean tensor is given, it should broadcast with the quaternions.
        """
        super().__init__()

        quaternions_ = torch.as_tensor(quaternions)
        if torch.is_complex(quaternions_):
            raise ValueError('quaternions should be real numbers')
        if not torch.is_floating_point(quaternions_):
            # integer or boolean dtypes
            quaternions_ = quaternions_.float()
        if quaternions_.shape[-1] != 4:
            raise ValueError('Expected `quaternions` to have shape (..., 4), ' f'got {quaternions_.shape}.')

        reflection_ = torch.as_tensor(reflection)
        inversion_ = torch.as_tensor(inversion)
        if reflection_.any():
            axis, angle = _quaternion_to_axis_angle(quaternions_)
            angle = (angle + torch.pi * reflection_.float()).unsqueeze(-1)
            is_improper = inversion_ ^ reflection_
            quaternions_ = torch.cat((torch.sin(angle / 2) * axis, torch.cos(angle / 2)), -1)
        elif inversion_.any():
            is_improper = inversion_
        else:
            is_improper = torch.zeros_like(quaternions_[..., 0], dtype=torch.bool)

        batchsize = torch.broadcast_shapes(quaternions_.shape[:-1], is_improper.shape)
        is_improper = is_improper.expand(batchsize)

        # If a single quaternion is given, convert it to a 2D 1 x 4 matrix but
        # set self._single to True so that we can return appropriate objects
        # in the `to_...` methods
        if quaternions_.shape == (4,):
            quaternions_ = quaternions_[None, :]
            is_improper = is_improper[None]
            self._single = True
        else:
            self._single = False

        if normalize:
            norms = torch.linalg.vector_norm(quaternions_, dim=-1, keepdim=True)
            if torch.any(torch.isclose(norms.float(), torch.tensor(0.0))):
                raise ValueError('Found zero norm quaternion in `quaternions`.')
            quaternions_ = quaternions_ / norms
        elif copy:
            # no need to clone if we are normalizing
            quaternions_ = quaternions_.clone()
        if copy:
            is_improper = is_improper.clone()

        if is_improper.requires_grad:
            warnings.warn('Rotation is not differentiable in the improper parameter.', stacklevel=2)

        self._quaternions = torch.nn.Parameter(quaternions_, quaternions_.requires_grad)
        self._is_improper = torch.nn.Parameter(is_improper, False)

    @property
    def single(self) -> bool:
        """Returns true if this a single rotation."""
        return self._single

    @property
    def is_improper(self) -> torch.Tensor:
        """Returns a true boolean tensor if the rotation is improper."""
        return self._is_improper

    @is_improper.setter
    def is_improper(self, improper: torch.Tensor | NestedSequence[bool] | bool) -> None:
        """Set the improper parameter."""
        self._is_improper[:] = torch.as_tensor(improper, dtype=torch.bool, device=self._is_improper.device)

    @property
    def det(self) -> torch.Tensor:
        """Returns the determinant of the rotation matrix.

        Will be 1. for proper rotations and -1. for improper rotations.
        """
        return self._is_improper.float() * -2 + 1

    @classmethod
    def from_quat(
        cls,
        quaternions: torch.Tensor | NestedSequence[float],
        inversion: torch.Tensor | NestedSequence[bool] | bool = False,
        reflection: torch.Tensor | NestedSequence[bool] | bool = False,
    ) -> Self:
        """Initialize from quaternions.

        3D rotations can be represented using unit-norm quaternions [QUAa]_.
        As an extension to the standard, this class also supports improper rotations,
        i.e. rotations with reflection with respect to the plane perpendicular to the rotation axis
        or inversion of the coordinate system.

        Note: If inversion != reflection, the rotation will be improper and save as a rotation followed by an inversion.
        containing an inversion of the coordinate system.

        Parameters
        ----------
        quaternions
            shape (..., 4)
            Each row is a (possibly non-unit norm) quaternion representing an
            active rotation, in scalar-last (x, y, z, w) format. Each
            quaternion will be normalized to unit norm.
        inversion
            if the rotation should contain an inversion of the coordinate system, i.e. a reflection
            of all three axes. If a boolean tensor is given, it should broadcast with the quaternions.
        reflection
            if the rotation should contain a reflection about a plane perpendicular to the rotation axis.


        Returns
        -------
        rotation
            Object containing the rotations represented by input quaternions.

        References
        ----------
        .. [QUAa] Quaternions and spatial rotation https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        """
        return cls(quaternions, normalize=True, copy=True, inversion=inversion, reflection=reflection)

    @classmethod
    def from_matrix(cls, matrix: torch.Tensor | NestedSequence[float], allow_improper: bool = True) -> Self:
        """Initialize from rotation matrix.

        Rotations in 3 dimensions can be represented with 3 x 3 proper
        orthogonal matrices [ROTa]_. If the input is not proper orthogonal,
        an approximation is created using the method described in [MAR2008]_.
        If the input matrix has a negative determinant, the rotation is considered
        as improper, i.e. containing a reflection. The resulting rotation
        will include this reflection [ROTb]_.

        Parameters
        ----------
        matrix
            A single matrix or a stack of matrices, shape (..., 3, 3)
        allow_improper
            If true, the rotation is considered as improper if the determinant of the matrix is negative.
            If false, an ValueError is raised if the determinant is negative.

        Returns
        -------
        rotation
            Object containing the rotations represented by the rotation
            matrices.

        References
        ----------
        .. [ROTa] Rotation matrix https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        .. [ROTb] Rotation matrix https://en.wikipedia.org/wiki/Improper_rotation
        .. [MAR2008] Landis Markley F (2008) Unit Quaternion from Rotation Matrix, Journal of guidance, control, and
           dynamics 31(2),440-442.
        """
        matrix_ = torch.as_tensor(matrix)
        if matrix_.shape[-2:] != (3, 3):
            raise ValueError(f'Expected `matrix` to have shape (..., 3, 3), got {matrix_.shape}')
        if torch.is_complex(matrix_):
            raise ValueError('matrix should be real, not complex.')
        if not torch.is_floating_point(matrix_):
            # integer or boolean dtypes
            matrix_ = matrix_.float()

        det = torch.linalg.det(matrix_)
        improper = det < 0
        if improper.any():
            if not allow_improper:
                raise ValueError(
                    'Found negative determinant in `matrix`. '
                    'This would result in an improper rotation, but allow_improper is False.'
                )
            matrix_ = matrix_ * det.unsqueeze(-1).unsqueeze(-1).sign()

        quaternions = _matrix_to_quaternion(matrix_)

        return cls(quaternions, normalize=True, copy=False, inversion=improper, reflection=False)

    @classmethod
    def from_directions(
        cls, *basis: Unpack[tuple[SpatialDimension, SpatialDimension, SpatialDimension]], allow_improper: bool = True
    ):
        """Initialize from basis vectors as SpatialDimensions.

        Parameters
        ----------
        *basis
            3 Basis vectors of the new coordinate system, i.e. the columns of the rotation matrix
        allow_improper
            If true, the rotation is considered as improper if the determinant of the matrix is negative
            and the sign will be preserved. If false, a ValueError is raised if the determinant is negative.


        Returns
        -------
        rotation
            Object containing the rotations represented by the basis vectors.
        """
        b1, b2, b3 = (torch.stack([torch.as_tensor(getattr(v_, axis)) for axis in AXIS_ORDER], -1) for v_ in basis)
        matrix = torch.stack((b1, b2, b3), -1)
        det = torch.linalg.det(matrix)
        if not allow_improper and (det < 0).any():
            raise ValueError('The given basis vectors do not form a proper rotation matrix.')
        if ((1 - det.abs()) > 0.1).any():
            raise ValueError('The given basis vectors do not form a rotation matrix.')

        return cls.from_matrix(matrix, allow_improper=allow_improper)

    def as_directions(
        self,
    ) -> tuple[SpatialDimension[torch.Tensor], SpatialDimension[torch.Tensor], SpatialDimension[torch.Tensor]]:
        """Represent as the basis vectors of the new coordinate system as SpatialDimensions.

        Returns the three basis vectors of the new coordinate system after rotation,
        i.e. the columns of the rotation matrix, as SpatialDimensions.

        Returns
        -------
        basis
            The basis vectors of the new coordinate system.
        """
        matrix = self.as_matrix()
        ret = (
            SpatialDimension(**dict(zip(AXIS_ORDER, matrix[..., 0].unbind(-1), strict=True))),
            SpatialDimension(**dict(zip(AXIS_ORDER, matrix[..., 1].unbind(-1), strict=True))),
            SpatialDimension(**dict(zip(AXIS_ORDER, matrix[..., 2].unbind(-1), strict=True))),
        )
        return ret

    @classmethod
    def from_rotvec(
        cls,
        rotvec: torch.Tensor | NestedSequence[float],
        degrees: bool = False,
        reflection: torch.Tensor | NestedSequence[bool] | bool = False,
        inversion: torch.Tensor | NestedSequence[bool] | bool = False,
    ) -> Self:
        """Initialize from rotation vector.

        A rotation vector is a 3 dimensional vector which is co-directional to the
        axis of rotation and whose norm gives the angle of rotation.

        Parameters
        ----------
        rotvec
            shape (..., 3), the rotation vectors.
        degrees
            If True, then the given angles are assumed to be in degrees,
            otherwise radians.
        reflection
            If True, the resulting transformation will contain a reflection
            about a plane perpendicular to the rotation axis, resulting in a rotoflection
            (improper rotation).
        inversion
            If True, the resulting transformation will contain an inversion of the coordinate system,
            resulting in a rotoinversion (improper rotation).

        Returns
        -------
        rotation
            Object containing the rotations represented by the rotation vectors.

        """
        rotvec_ = torch.as_tensor(rotvec)
        reflection_ = torch.as_tensor(reflection)
        inversion_ = torch.as_tensor(inversion)
        if rotvec_.is_complex():
            raise ValueError('rotvec should be real numbers')
        if not rotvec_.is_floating_point():
            # integer or boolean dtypes
            rotvec_ = rotvec_.float()
        if degrees:
            rotvec_ = torch.deg2rad(rotvec_)

        if rotvec_.shape[-1] != 3:
            raise ValueError(f'Expected `rot_vec` to have shape (..., 3), got {rotvec_.shape}')

        angles = torch.linalg.vector_norm(rotvec_, dim=-1, keepdim=True)
        scales = torch.special.sinc(angles / (2 * torch.pi)) / 2
        quaternions = torch.cat((scales * rotvec_, torch.cos(angles / 2)), -1)
        if reflection_.any():
            # we can do it here and avoid the extra of converting to quaternions,
            # back to axis-angle and then to quaternions.
            inversion_ = reflection_ ^ inversion_
            scales = torch.cos(0.5 * angles) / angles
            reflected_quaternions = torch.cat((scales * rotvec_, -torch.sin(angles / 2)), -1)
            quaternions = torch.where(reflection_, reflected_quaternions, quaternions)

        return cls(quaternions, normalize=False, copy=False, inversion=inversion_, reflection=False)

    @classmethod
    def from_euler(
        cls,
        seq: str,
        angles: torch.Tensor | NestedSequence[float] | float,
        degrees: bool = False,
        inversion: torch.Tensor | NestedSequence[bool] | bool = False,
        reflection: torch.Tensor | NestedSequence[bool] | bool = False,
    ) -> Self:
        """Initialize from Euler angles.

        Rotations in 3-D can be represented by a sequence of 3
        rotations around a sequence of axes. In theory, any three axes spanning
        the 3-D Euclidean space are enough. In practice, the axes of rotation are
        chosen to be the basis vectors.

        The three rotations can either be in a global frame of reference
        (extrinsic) or in a body centered frame of reference (intrinsic), which
        is attached to, and moves with, the object under rotation [EULa]_.

        Parameters
        ----------
        seq
            Specifies sequence of axes for rotations. Up to 3 characters
            belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations, or
            {'x', 'y', 'z'} for extrinsic rotations. Extrinsic and intrinsic
            rotations cannot be mixed in one function call.
        angles
            (..., [1 or 2 or 3]), matching the number of axes in seq.
            Euler angles specified in radians (`degrees` is False) or degrees
            (`degrees` is True).
        degrees
            If True, then the given angles are assumed to be in degrees.
            Otherwise they are assumed to be in radians
        inversion
            If True, the resulting transformation will contain an inversion of the coordinate system,
            resulting in a rotoinversion (improper rotation).
        reflection
            If True, the resulting transformation will contain a reflection
            about a plane perpendicular to the rotation axis, resulting in an
            improper rotation.

        Returns
        -------
        rotation
            Object containing the rotation represented by the sequence of
            rotations around given axes with given angles.

        References
        ----------
        .. [EULa] Euler angles https://en.wikipedia.org/wiki/Euler_angles#Definition_by_intrinsic_rotations
        """
        n_axes = len(seq)
        if n_axes < 1 or n_axes > 3:
            raise ValueError('Expected axis specification to be a non-empty ' f'string of upto 3 characters, got {seq}')

        intrinsic = re.match(r'^[XYZ]{1,3}$', seq) is not None
        extrinsic = re.match(r'^[xyz]{1,3}$', seq) is not None
        if not (intrinsic or extrinsic):
            raise ValueError("Expected axes from `seq` to be from ['x', 'y', " f"'z'] or ['X', 'Y', 'Z'], got {seq}")

        if any(seq[i] == seq[i + 1] for i in range(n_axes - 1)):
            raise ValueError('Expected consecutive axes to be different, ' f'got {seq}')
        seq = seq.lower()

        angles = torch.as_tensor(angles)
        if degrees:
            angles = torch.deg2rad(angles)
        if n_axes == 1 and angles.ndim == 0:
            angles = angles.reshape((1, 1))
            is_single = True
        elif angles.ndim == 1:
            angles = angles[None, :]
            is_single = True
        else:
            is_single = False
        if angles.ndim < 2 or angles.shape[-1] != n_axes:
            raise ValueError('Expected angles to have shape (..., ' f'n_axes), got {angles.shape}.')

        quaternions = _make_elementary_quat(seq[0], angles[..., 0])
        for axis, angle in zip(seq[1:], angles[..., 1:].unbind(-1), strict=False):
            if intrinsic:
                quaternions = _compose_quaternions(quaternions, _make_elementary_quat(axis, angle))
            else:
                quaternions = _compose_quaternions(_make_elementary_quat(axis, angle), quaternions)

        if is_single:
            return cls(quaternions[0], normalize=False, copy=False, inversion=inversion, reflection=reflection)
        else:
            return cls(quaternions, normalize=False, copy=False, inversion=inversion, reflection=reflection)

    @classmethod
    def from_davenport(cls, axes: torch.Tensor, order: str, angles: torch.Tensor, degrees: bool = False):
        """Not implemented."""
        raise NotImplementedError

    @classmethod
    def from_mrp(cls, mrp: torch.Tensor) -> Self:
        """Not implemented."""
        raise NotImplementedError

    @overload
    def as_quat(
        self, canonical: bool = ..., *, improper: Literal['warn'] | Literal['ignore'] = 'warn'
    ) -> torch.Tensor: ...
    @overload
    def as_quat(
        self, canonical: bool = ..., *, improper: Literal['reflection'] | Literal['inversion']
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def as_quat(
        self,
        canonical: bool = False,
        *,
        improper: Literal['reflection'] | Literal['inversion'] | Literal['ignore'] | Literal['warn'] = 'warn',
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Represent as quaternions.

        Active rotations in 3 dimensions can be represented using unit norm
        quaternions [QUAb]_. The mapping from quaternions to rotations is
        two-to-one, i.e. quaternions ``q`` and ``-q``, where ``-q`` simply
        reverses the sign of each component, represent the same spatial
        rotation. The returned value is in scalar-last (x, y, z, w) format.

        Parameters
        ----------
        canonical
            Whether to map the redundant double cover of rotation space to a
            unique "canonical" single cover. If True, then the quaternion is
            chosen from {q, -q} such that the w term is positive. If the w term
            is 0, then the quaternion is chosen such that the first nonzero
            term of the x, y, and z terms is positive.
        improper
            How to handle improper rotations. If 'warn', a warning is raised if
            the rotation is improper. If 'ignore', the reflection information is
            discarded. If 'reflection' or 'inversion', additional information is
            returned in the form of a boolean tensor indicating if the rotation
            is improper.
            If 'reflection', the boolean tensor indicates if the rotation contains
            a reflection about a plane perpendicular to the rotation axis.
            Note that this required additional computation.
            If 'inversion', the boolean tensor indicates if the rotation contains
            an inversion of the coordinate system.
            The quaternion is adjusted to represent the rotation to be performed
            before the reflection or inversion.

        Returns
        -------
        quaternions
            shape (..., 4,), depends on shape of inputs used for initialization.
        (optional) reflection (if improper is 'reflection') or inversion (if improper is 'inversion')
            boolean tensor of shape (...,), indicating if the rotation is improper
            and if a reflection or inversion should be performed after the rotation.

        References
        ----------
        .. [QUAb] Quaternions https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        """
        quaternions: torch.Tensor = self._quaternions
        is_improper: torch.Tensor = self._is_improper

        if improper == 'warn':
            if is_improper.any():
                warnings.warn(
                    'Rotation contains improper rotations. Set `improper="reflection"` or `improper="inversion"` '
                    'to get reflection or inversion information.',
                    stacklevel=2,
                )
        elif improper == 'ignore' or improper == 'inversion':
            ...
        elif improper == 'reflection':
            axis, angle = _quaternion_to_axis_angle(quaternions)
            angle = (angle + torch.pi * is_improper.float()).unsqueeze(-1)
            quaternions = torch.cat((torch.sin(angle / 2) * axis, torch.cos(angle / 2)), -1)
        else:
            raise ValueError(f'Invalid improper value: {improper}')

        if self.single:
            quaternions = quaternions[0]
            is_improper = is_improper[0]

        if canonical:
            quaternions = _canonical_quaternion(quaternions)
        else:
            quaternions = quaternions.clone()

        if improper == 'reflection' or improper == 'inversion':
            return quaternions, is_improper
        else:
            return quaternions

    def as_matrix(self) -> torch.Tensor:
        """Represent as rotation matrix.

        3D rotations can be represented using rotation matrices, which
        are 3 x 3 real orthogonal matrices with determinant equal to +1 [ROTb]_
        for proper rotations and -1 for improper rotations.

        Returns
        -------
        matrix
            shape (..., 3, 3), depends on shape of inputs used for initialization.

        References
        ----------
        .. [ROTb] Rotation matrix https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        """
        quaternions = self._quaternions
        matrix = _quaternion_to_matrix(quaternions)
        if self._is_improper.any():
            matrix = matrix * self.det.unsqueeze(-1).unsqueeze(-1)

        if self._single:
            return matrix[0]
        else:
            return matrix

    @overload
    def as_rotvec(
        self, degrees: bool = ..., *, improper: Literal['ignore'] | Literal['warn'] = 'warn'
    ) -> torch.Tensor: ...
    @overload
    def as_rotvec(
        self, degrees: bool = ..., *, improper: Literal['reflection'] | Literal['inversion']
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def as_rotvec(
        self,
        degrees: bool = False,
        improper: Literal['reflection'] | Literal['inversion'] | Literal['ignore'] | Literal['warn'] = 'warn',
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Represent as rotation vectors.

        A rotation vector is a 3 dimensional vector which is co-directional to
        the axis of rotation and whose norm gives the angle of rotation [ROTc]_.

        Parameters
        ----------
        degrees
            Returned magnitudes are in degrees if this flag is True, else they are in radians
        improper
            How to handle improper rotations. If 'warn', a warning is raised if
            the rotation is improper. If 'ignore', the reflection information is
            discarded. If 'reflection' or 'inversion', additional information is
            returned in the form of a boolean tensor indicating if the rotation
            is improper.
            If 'reflection', the boolean tensor indicates if the rotation contains
            a reflection about a plane perpendicular to the rotation axis.
            If 'inversion', the boolean tensor indicates if the rotation contains
            an inversion of the coordinate system.
            The quaternion is adjusted to represent the rotation to be performed
            before the reflection or inversion.

        Returns
        -------
        rotvec
            Shape (..., 3), depends on shape of inputs used for initialization.
        (optional) reflection (if improper is 'reflection') or inversion (if improper is 'inversion')
            boolean tensor of shape (...,), indicating if the rotation is improper
            and if a reflection or inversion should be performed after the rotation.


        References
        ----------
        .. [ROTc] Rotation vector https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Rotation_vector
        """
        if improper == 'reflection' or improper == 'inversion':
            quaternions, is_improper = self.as_quat(canonical=True, improper=improper)
        else:
            quaternions, is_improper = self.as_quat(canonical=True, improper=improper), None
        angles = 2 * torch.atan2(torch.linalg.vector_norm(quaternions[..., :3], dim=-1), quaternions[..., 3])
        scales = 2 / (torch.special.sinc(angles / (2 * torch.pi)))
        rotvec = scales[..., None] * quaternions[..., :3]
        if degrees:
            rotvec = torch.rad2deg(rotvec)
        if is_improper is not None:
            return rotvec, is_improper
        else:
            return rotvec

    @overload
    def as_euler(
        self,
        seq: str,
        degrees: bool = ...,
        *,
        improper: Literal['ignore'] | Literal['warn'] = 'warn',
    ) -> torch.Tensor: ...
    @overload
    def as_euler(
        self,
        seq: str,
        degrees: bool = ...,
        *,
        improper: Literal['reflection'] | Literal['inversion'],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def as_euler(
        self,
        seq: str,
        degrees: bool = False,
        *,
        improper: Literal['reflection'] | Literal['inversion'] | Literal['ignore'] | Literal['warn'] = 'warn',
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Represent as Euler angles.

        Any orientation can be expressed as a composition of 3 elementary
        rotations. Once the axis sequence has been chosen, Euler angles define
        the angle of rotation around each respective axis [EULb]_.

        The algorithm from [BER2022]_ has been used to calculate Euler angles for the
        rotation about a given sequence of axes.

        Euler angles suffer from the problem of gimbal lock [GIM]_, where the
        representation loses a degree of freedom and it is not possible to
        determine the first and third angles uniquely. In this case,
        a warning is raised, and the third angle is set to zero. Note however
        that the returned angles still represent the correct rotation.

        Parameters
        ----------
        seq
            3 characters belonging to the set {'X', 'Y', 'Z'} for intrinsic
            rotations, or {'x', 'y', 'z'} for extrinsic rotations [EULb]_.
            Adjacent axes cannot be the same.
            Extrinsic and intrinsic rotations cannot be mixed in one function
            call.
        degrees
            Returned angles are in degrees if this flag is True, else they are
            in radians
        improper
            How to handle improper rotations. If 'warn', a warning is raised if
            the rotation is improper. If 'ignore', the reflection information is
            discarded. If 'reflection' or 'inversion', additional information is
            returned in the form of a boolean tensor indicating if the rotation
            is improper.
            If 'reflection', the boolean tensor indicates if the rotation contains
            a reflection about a plane perpendicular to the rotation axis.
            If 'inversion', the boolean tensor indicates if the rotation contains
            an inversion of the coordinate system.
            The quaternion is adjusted to represent the rotation to be performed
            before the reflection or inversion.

        Returns
        -------
        angles
            shape (3,) or (..., 3), depending on shape of inputs used to initialize object.
            The returned angles are in the range:

            - First angle belongs to [-180, 180] degrees (both inclusive)
            - Third angle belongs to [-180, 180] degrees (both inclusive)
            - Second angle belongs to:

             + [-90, 90] degrees if all axes are different (like xyz)
             + [0, 180] degrees if first and third axes are the same (like zxz)

        References
        ----------
        .. [EULb] Euler Angles https://en.wikipedia.org/wiki/Euler_angles#Definition_by_intrinsic_rotations
        .. [BER2022] Bernardes E, Viollet S (2022) Quaternion to Euler angles conversion: A direct, general and
           computationally efficient method. PLoS ONE 17(11) https://doi.org/10.1371/journal.pone.0276302
        .. [GIM] Gimbal lock https://en.wikipedia.org/wiki/Gimbal_lock#In_applied_mathematics
        """
        if len(seq) != 3:
            raise ValueError(f'Expected 3 axes, got {seq}.')

        intrinsic = re.match(r'^[XYZ]{1,3}$', seq) is not None
        extrinsic = re.match(r'^[xyz]{1,3}$', seq) is not None
        if not (intrinsic or extrinsic):
            raise ValueError('Expected axes from `seq` to be from ' "['x', 'y', 'z'] or ['X', 'Y', 'Z'], " f'got {seq}')

        if any(seq[i] == seq[i + 1] for i in range(2)):
            raise ValueError('Expected consecutive axes to be different, ' f'got {seq}')

        seq = seq.lower()
        if improper == 'reflection' or improper == 'inversion':
            quat, is_improper = self.as_quat(improper=improper)
        else:
            quat, is_improper = self.as_quat(improper=improper), None

        if quat.ndim == 1:
            quat = quat[None, :]

        angles = _quaternion_to_euler(quat, seq, extrinsic)
        if degrees:
            angles = torch.rad2deg(angles)

        angles_ = angles[0] if self._single else angles

        if is_improper is not None:
            return angles_, is_improper
        else:
            return angles_

    def as_davenport(self, axes: torch.Tensor, order: str, degrees: bool = False) -> torch.Tensor:
        """Not implemented."""
        raise NotImplementedError

    def as_mrp(self) -> torch.Tensor:
        """Not implemented."""
        raise NotImplementedError

    @classmethod
    def concatenate(cls, rotations: Sequence[Rotation]) -> Self:
        """Concatenate a sequence of `Rotation` objects into a single object.

        Parameters
        ----------
        rotations
            The rotations to concatenate.

        Returns
        -------
        concatenated
            The concatenated rotations.
        """
        if not all(isinstance(x, Rotation) for x in rotations):
            raise TypeError('input must contain Rotation objects only')

        quats = torch.cat([torch.atleast_2d(x.as_quat()) for x in rotations])
        inversions = torch.cat([torch.atleast_1d(x._is_improper) for x in rotations])
        return cls(quats, normalize=False, copy=False, inversion=inversions, reflection=False)

    @overload
    def apply(self, fn: NestedSequence[float] | torch.Tensor, inverse: bool) -> torch.Tensor: ...

    @overload
    def apply(
        self, fn: SpatialDimension[torch.Tensor] | SpatialDimension[float], inverse: bool
    ) -> SpatialDimension[torch.Tensor]: ...

    @overload
    def apply(self, fn: Callable[[torch.nn.Module], None]) -> Self: ...

    def apply(
        self,
        fn: NestedSequence[float]
        | torch.Tensor
        | SpatialDimension[torch.Tensor]
        | SpatialDimension[float]
        | Callable[[torch.nn.Module], None],
        inverse: bool = False,
    ) -> torch.Tensor | SpatialDimension[torch.Tensor] | Self:
        """Either apply a function to the Rotation module or apply the rotation to a vector.

        This is a hybrid method that matches the signature of both `torch.nn.Module.apply` and
        `scipy.spatial.transform.Rotation.apply`.
        If a callable is passed, it is assumed to be a function that will be applied to the Rotation module.
        For applying the rotation to a vector, consider using `Rotation(vector)` instead of `Rotation.apply(vector)`.
        """
        if callable(fn):
            # torch.nn.Module.apply
            return super().apply(fn)
        else:
            # scipy.spatial.transform.Rotation.apply
            warnings.warn('Consider using Rotation(vector) instead of Rotation.apply(vector).', stacklevel=2)
            return self(fn, inverse)

    @overload
    def __call__(self, vectors: NestedSequence[float] | torch.Tensor, inverse: bool = False) -> torch.Tensor: ...

    @overload
    def __call__(
        self, vectors: SpatialDimension[torch.Tensor] | SpatialDimension[float], inverse: bool = False
    ) -> SpatialDimension[torch.Tensor]: ...

    def __call__(
        self,
        vectors: NestedSequence[float] | torch.Tensor | SpatialDimension[torch.Tensor] | SpatialDimension[float],
        inverse: bool = False,
    ) -> torch.Tensor | SpatialDimension[torch.Tensor]:
        """Apply this rotation to a set of vectors."""
        # Only for type hinting
        return super().__call__(vectors, inverse)

    def forward(
        self,
        vectors: NestedSequence[float] | torch.Tensor | SpatialDimension[torch.Tensor] | SpatialDimension[float],
        inverse: bool = False,
    ) -> torch.Tensor | SpatialDimension[torch.Tensor]:
        """Apply this rotation to a set of vectors.

        If the original frame rotates to the final frame by this rotation, then
        its application to a vector can be seen in two ways:

        - As a projection of vector components expressed in the final frame to the original frame.
        - As the physical rotation of a vector being glued to the original frame as it rotates. In this case the vector
          components are expressed in the original frame before and after the rotation.

        In terms of rotation matrices, this application is the same as
        ``self.as_matrix() @ vectors``.

        Parameters
        ----------
        vectors
            Shape(..., 3). Each `vectors[i]` represents a vector in 3D space.
            A single vector can either be specified with shape `(3, )` or `(1, 3)`.
            The number of rotations and number of vectors given must follow standard
            pytorch broadcasting rules.
        inverse
            If True then the inverse of the rotation(s) is applied to the input
            vectors.

        Returns
        -------
        rotated_vectors
            Result of applying rotation on input vectors.
            Shape depends on the following cases:

                - If object contains a single rotation (as opposed to a stack
                  with a single rotation) and a single vector is specified with
                  shape ``(3,)``, then `rotated_vectors` has shape ``(3,)``.
                - In all other cases, `rotated_vectors` has shape ``(..., 3)``,
                  where ``...`` is determined by broadcasting.
        """
        matrix = self.as_matrix()
        if inverse:
            matrix = matrix.mT
        if self._single:
            matrix = matrix.unsqueeze(0)

        if input_is_spatialdimension := isinstance(vectors, SpatialDimension):
            # sort the axis by AXIS_ORDER
            vectors_tensor = torch.stack([torch.as_tensor(getattr(vectors, axis)) for axis in AXIS_ORDER], -1)
        else:
            vectors_tensor = torch.as_tensor(vectors)
        if vectors_tensor.shape[-1] != 3:
            raise ValueError(f'Expected input of shape (..., 3), got {vectors_tensor.shape}.')
        if vectors_tensor.is_complex():
            raise ValueError('Complex vectors are not supported. The coordinates to rotate should be real numbers.')
        if vectors_tensor.dtype != matrix.dtype:
            dtype = torch.promote_types(matrix.dtype, vectors_tensor.dtype)
            matrix = matrix.to(dtype=dtype)
            vectors_tensor = vectors_tensor.to(dtype=dtype)

        try:
            result = (matrix @ vectors_tensor.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            raise ValueError(
                f'The batch-shape of the rotation, {list(matrix.shape[:-2])}, '
                f'is not compatible with the input batch shape {list(vectors_tensor.shape[:-1])}'
            ) from None

        if self._single and vectors_tensor.shape == (3,):
            # a single rotation and a single vector
            result = result[0]

        if input_is_spatialdimension:
            return SpatialDimension(
                x=result[..., AXIS_ORDER.index('x')],
                y=result[..., AXIS_ORDER.index('y')],
                z=result[..., AXIS_ORDER.index('z')],
            )
        else:
            return result

    @classmethod
    def random(
        cls,
        num: int | Sequence[int] | None = None,
        random_state: int | np.random.RandomState | np.random.Generator | None = None,
        improper: bool | Literal['random'] = False,
    ):
        """Generate uniformly distributed rotations.

        Parameters
        ----------
        num
            Number of random rotations to generate. If None (default), then a
            single rotation is generated.
        random_state
            If `random_state` is None, the `numpy.random.RandomState`
            singleton is used.
            If `random_state` is an int, a new ``RandomState`` instance is used,
            seeded with `random_state`.
            If `random_state` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.
        improper
            if True, only improper rotations are generated. If False, only proper rotations are generated.
            if "random", then a random mix of proper and improper rotations are generated.

        Returns
        -------
        random_rotation
            Contains a single rotation if `num` is None. Otherwise contains a
            stack of `num` rotations.
        """
        generator: np.random.RandomState = check_random_state(random_state)

        if num is None:
            random_sample = torch.as_tensor(generator.normal(size=4), dtype=torch.float32)
        elif isinstance(num, int):
            random_sample = torch.as_tensor(generator.normal(size=(num, 4)), dtype=torch.float32)
        else:
            random_sample = torch.as_tensor(generator.normal(size=(*num, 4)), dtype=torch.float32)
        if improper == 'random':
            inversion: torch.Tensor | bool = torch.as_tensor(
                generator.choice([True, False], size=random_sample.shape[:-1]), dtype=torch.bool
            )
        elif isinstance(improper, bool):
            inversion = improper
        else:
            raise ValueError('improper should be a boolean or "random"')
        return cls(random_sample, inversion=inversion, reflection=False, normalize=True, copy=False)

    @classmethod
    def random_vmf(
        cls,
        num: int | None = None,
        mean_axis: torch.Tensor | None = None,
        kappa: float = 0.0,
        sigma: float = math.inf,
    ):
        """
        Randomly sample rotations from a von Mises-Fisher distribution.

        Generate rotations from a von Mises-Fisher distribution with a given mean axis and concentration parameter
        and a 2pi-wrapped Gaussian distribution for the rotation angle.

        Parameters
        ----------
        mean_axis
            shape (..., 3,), the mean axis of the von Mises-Fisher distribution.
        kappa
            The concentration parameter of the von Mises-Fisher distribution.
            small kappa results in a uniform distribution, large kappa results in a peak around the mean axis.
            similar to the inverse of the variance of a Gaussian distribution.
        sigma
            Standard deviation (radians) of the 2pi-wrapped Gaussian distribution used to sample the rotation angle.
            Use `math.inf` if a uniform distribution is desired.
        num
            number of samples to generate. If None, a single rotation is generated.

        Returns
        -------
        random_rotation
            a stack of `(num, ...)` rotations.

        """
        n = 1 if num is None else num
        mu = torch.tensor((1.0, 0.0, 0.0)) if mean_axis is None else torch.as_tensor(mean_axis)
        rot_axes = sample_vmf(mu=mu, kappa=kappa, n_samples=n)
        if sigma == math.inf:
            rot_angle = torch.rand(n, *mu.shape[:-1], dtype=mu.dtype, device=mu.device) * 2 * math.pi
        else:
            rot_angle = (torch.randn(n, *mu.shape[:-1], dtype=mu.dtype, device=mu.device) * sigma) % (2 * math.pi)
        return cls.from_rotvec(rot_axes * rot_angle.unsqueeze(-1))

    def __mul__(self, other: Rotation) -> Self:
        """For compatibility with sp.spatial.transform.Rotation."""
        warnings.warn(
            'Using Rotation*Rotation is deprecated, consider Rotation@Rotation', DeprecationWarning, stacklevel=2
        )
        return self @ other

    def __matmul__(self, other: Rotation) -> Self:
        """Compose this rotation with the other.

        If `p` and `q` are two rotations, then the composition of 'q followed
        by p' is equivalent to `p * q`. In terms of rotation matrices,
        the composition can be expressed as
        ``p.as_matrix() @ q.as_matrix()``.

        Parameters
        ----------
        other
            Object containing the rotations to be composed with this one. Note
            that rotation compositions are not commutative, so ``p * q`` is
            generally different from ``q * p``.

        Returns
        -------
        composition
            This function supports composition of multiple rotations at a time.
            The following cases are possible:

            - Either ``p`` or ``q`` contains a single rotation. In this case
              `composition` contains the result of composing each rotation in
              the other object with the single rotation.
            - Both ``p`` and ``q`` contain ``N`` rotations. In this case each
              rotation ``p[i]`` is composed with the corresponding rotation
              ``q[i]`` and `output` contains ``N`` rotations.
        """
        if not isinstance(other, Rotation):
            return NotImplemented  # type: ignore[unreachable]

        p = self._quaternions
        q = other._quaternions
        p, q = torch.broadcast_tensors(p, q)
        result_quaternions = _compose_quaternions(p, q)
        result_improper = self._is_improper ^ other._is_improper

        if self._single and other._single:
            result_quaternions = result_quaternions[0]
            result_improper = result_improper[0]
        return self.__class__(result_quaternions, normalize=True, copy=False, inversion=result_improper)

    def __pow__(self, n: float, modulus: None = None):
        """Compose this rotation with itself `n` times.

        Composition of a rotation ``p`` with itself can be extended to
        non-integer ``n`` by considering the power ``n`` to be a scale factor
        applied to the angle of rotation about the rotation's fixed axis. The
        expression ``q = p ** n`` can also be expressed as
        ``q = Rotation.from_rotvec(n * p.as_rotvec())``.

        If ``n`` is negative, then the rotation is inverted before the power
        is applied. In other words, ``p ** -abs(n) == p.inv() ** abs(n)``.

        Parameters
        ----------
        n
            The number of times to compose the rotation with itself.
        modulus
            This overridden argument is not applicable to Rotations and must be
            ``None``.

        Returns
        -------
        power : `Rotation` instance
            If the input Rotation ``p`` contains ``N`` multiple rotations, then
            the output will contain ``N`` rotations where the ``i`` th rotation
            is equal to ``p[i] ** n``

        Notes
        -----
        For example, a power of 2 will double the angle of rotation, and a
        power of 0.5 will halve the angle. There are three notable cases: if
        ``n == 1`` then the original rotation is returned, if ``n == 0``
        then the identity rotation is returned, and if ``n == -1`` then
        ``p.inv()`` is returned.

        For improper rotations, the power of a rotation with a reflection is
        equivalent to the power of the rotation without the reflection, followed
        by an reflection if the power is integer and odd. If the power is
        non-integer, the reflection is never applied.
        This means that, for example a 0.5 power of a rotation with a reflection
        applied twice will result in a rotation without a reflection.

        Note that fractional powers ``n`` which effectively take a root of
        rotation, do so using the shortest path smallest representation of that
        angle (the principal root). This means that powers of ``n`` and ``1/n``
        are not necessarily inverses of each other. For example, a 0.5 power of
        a +240 degree rotation will be calculated as the 0.5 power of a -120
        degree rotation, with the result being a rotation of -60 rather than
        +120 degrees.
        """
        if modulus is not None:
            raise NotImplementedError('modulus not supported')

        # Exact short-cuts
        if n == 0:
            return Rotation.identity(None if self._single else self._quaternions.shape[:-1])
        elif n == -1:
            return self.inv()
        elif n == 1:
            if self._single:
                return self.__class__(self._quaternions[0], inversion=self._is_improper[0], copy=True)
            else:
                return self.__class__(self._quaternions, inversion=self._is_improper[0], copy=True)
        elif math.isclose(round(n), n) and round(n) % 2:
            improper: torch.Tensor | bool = self._is_improper
        else:
            improper = False

        return Rotation.from_rotvec(n * self.as_rotvec(), reflection=improper)

    def inv(self) -> Self:
        """Invert this rotation.

        Composition of a rotation with its inverse results in an identity
        transformation.

        Returns
        -------
        inverse
            Object containing inverse of the rotations in the current instance.
        """
        quaternions = self._quaternions * torch.tensor([-1, -1, -1, 1])
        improper = self._is_improper.clone()

        if self._single:
            quaternions = quaternions[0]
            improper = self._is_improper[0]

        return self.__class__(quaternions, inversion=improper, copy=False)

    def reflect(self) -> Self:
        """Reflect this rotation.

        Converts a proper rotation to an improper one, or vice versa
        by reflecting the rotation about a plane perpendicular to the rotation axis.

        Returns
        -------
        reflected
            Object containing the reflected rotations.
        """
        if self._single:
            quaternions = self._quaternions[0]
            is_improper = self._is_improper[0]
        else:
            quaternions = self._quaternions
            is_improper = self._is_improper

        return self.__class__(quaternions, copy=False, inversion=is_improper, reflection=True)

    def invert_axes(self) -> Self:
        """Invert the axes of the coordinate system.

        Converts a proper rotation to an improper one, or vice versa
        by inversion of the coordinate system.

        Note:
        This is not the same as the inverse of the rotation.
        See `inv` for that.

        Returns
        -------
        inverted_axes
            Object containing the rotation with inverted axes.
        """
        quaternions = self._quaternions.clone()
        improper = ~self._is_improper
        if self._single:
            quaternions = quaternions[0]
            improper = improper[0]
        return self.__class__(quaternions, copy=False, inversion=improper)

    def magnitude(self) -> torch.Tensor:
        """Get the magnitude(s) of the rotation(s).

        Returns
        -------
        magnitude
            Angles in radians. The magnitude will always be in the range [0, pi].
        """
        angles = 2 * torch.atan2(
            torch.linalg.vector_norm(self._quaternions[..., :3], dim=-1), torch.abs(self._quaternions[..., 3])
        )
        if self._single:
            angles = angles[0]
        return angles

    def approx_equal(self, other: Rotation, atol: float = 1e-6, degrees: bool = False) -> torch.Tensor:
        """Determine if another rotation is approximately equal to this one.

        Equality is measured by calculating the smallest angle between the
        rotations, and checking to see if it is smaller than `atol`.

        Parameters
        ----------
        other
            Object containing the rotations to measure against this one.
        atol
            The absolute angular tolerance, below which the rotations are
            considered equal.
        degrees
            If True and `atol` is given, then `atol` is measured in degrees. If
            False (default), then atol is measured in radians.

        Returns
        -------
        approx_equal :
            Whether the rotations are approximately equal, bool if object
            contains a single rotation and Tensor if object contains multiple
            rotations.
        """
        if degrees:
            atol = np.deg2rad(atol)
        angles = (other @ self.inv()).magnitude()
        return (angles < atol) & (self._is_improper == other._is_improper)

    def __getitem__(self, indexer: TorchIndexerType) -> Self:
        """Extract rotation(s) at given index(es) from object.

        Create a new `Rotation` instance containing a subset of rotations
        stored in this object.

        Parameters
        ----------
        indexer:
            Specifies which rotation(s) to extract.

        Returns
        -------
        rotation

        Raises
        ------
        TypeError if the instance was created as a single rotation.
        """
        if self._single:
            raise TypeError('Single rotation is not subscriptable.')
        if isinstance(indexer, tuple):
            indexer_quat = (*indexer, slice(None))
        else:
            indexer_quat = (indexer, slice(None))
        return self.__class__(self._quaternions[indexer_quat], normalize=False, inversion=self._is_improper[indexer])

    @property
    def quaternion_x(self) -> torch.Tensor:
        """Get x component of the quaternion."""
        axis = QUAT_AXIS_ORDER.index('x')
        if self._single:
            return self._quaternions[0, axis]
        return self._quaternions[..., axis]

    @quaternion_x.setter
    def quaternion_x(self, quat_x: torch.Tensor | float):
        """Set x component of the quaternion."""
        axis = QUAT_AXIS_ORDER.index('x')
        self._quaternions[..., axis] = quat_x

    @property
    def quaternion_y(self) -> torch.Tensor:
        """Get y component of the quaternion."""
        axis = QUAT_AXIS_ORDER.index('y')
        if self._single:
            return self._quaternions[0, axis]
        return self._quaternions[..., axis]

    @quaternion_y.setter
    def quaternion_y(self, quat_y: torch.Tensor | float):
        """Set y component of the quaternion."""
        axis = QUAT_AXIS_ORDER.index('y')
        self._quaternions[..., axis] = quat_y

    @property
    def quaternion_z(self) -> torch.Tensor:
        """Get z component of the quaternion."""
        axis = QUAT_AXIS_ORDER.index('z')
        if self._single:
            return self._quaternions[0, axis]
        return self._quaternions[..., axis]

    @quaternion_z.setter
    def quaternion_z(self, quat_z: torch.Tensor | float):
        """Set z component of the quaternion."""
        axis = QUAT_AXIS_ORDER.index('z')
        self._quaternions[..., axis] = quat_z

    @property
    def quaternion_w(self) -> torch.Tensor:
        """Get w component of the quaternion."""
        axis = QUAT_AXIS_ORDER.index('w')
        if self._single:
            return self._quaternions[0, axis]
        return self._quaternions[..., axis]

    @quaternion_w.setter
    def quaternion_w(self, quat_w: torch.Tensor | float):
        """Set w component of the quaternion."""
        axis = QUAT_AXIS_ORDER.index('w')
        self._quaternions[..., axis] = quat_w

    def __setitem__(self, indexer: TorchIndexerType, value: Rotation):
        """Set rotation(s) at given index(es) from object.

        Parameters
        ----------
        indexer
            Specifies which rotation(s) to replace.
        value
            The rotations to set.

        Raises
        ------
        TypeError if the instance was created as a single rotation.
        """
        if self._single:
            raise TypeError('Single rotation is not subscriptable.')

        if not isinstance(value, Rotation):
            raise TypeError('value must be a Rotation object')

        if isinstance(indexer, tuple):
            indexer_quat = (*indexer, slice(None))
        else:
            indexer_quat = (indexer, slice(None))
        quat, inversion = value.as_quat(improper='inversion')
        self._quaternions[indexer_quat] = quat
        self._is_improper[indexer] = inversion

    @classmethod
    def identity(cls, shape: int | None | tuple[int, ...] = None) -> Self:
        """Get identity rotation(s).

        Composition with the identity rotation has no effect.

        Parameters
        ----------
        shape
            Number of identity rotations to generate. If None (default), then a
            single rotation is generated.

        Returns
        -------
        identity : Rotation object
            The identity rotation.
        """
        match shape:
            case None:
                q = torch.zeros(4)
            case int():
                q = torch.zeros(shape, 4)
            case tuple():
                q = torch.zeros(*shape, 4)
        q[..., -1] = 1
        return cls(q, normalize=False)

    @overload
    @classmethod
    def align_vectors(
        cls,
        a: torch.Tensor | Sequence[torch.Tensor] | Sequence[float] | Sequence[Sequence[float]],
        b: torch.Tensor | Sequence[torch.Tensor] | Sequence[float] | Sequence[Sequence[float]],
        weights: torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | None = None,
        *,
        return_sensitivity: Literal[False] = False,
        allow_improper: bool = ...,
    ) -> tuple[Rotation, float]: ...

    @overload
    @classmethod
    def align_vectors(
        cls,
        a: torch.Tensor | Sequence[torch.Tensor] | Sequence[float] | Sequence[Sequence[float]],
        b: torch.Tensor | Sequence[torch.Tensor] | Sequence[float] | Sequence[Sequence[float]],
        weights: torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | None = None,
        *,
        return_sensitivity: Literal[True],
        allow_improper: bool = ...,
    ) -> tuple[Rotation, float, torch.Tensor]: ...

    @classmethod
    def align_vectors(
        cls,
        a: torch.Tensor | Sequence[torch.Tensor] | Sequence[float] | Sequence[Sequence[float]],
        b: torch.Tensor | Sequence[torch.Tensor] | Sequence[float] | Sequence[Sequence[float]],
        weights: torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | None = None,
        *,
        return_sensitivity: bool = False,
        allow_improper: bool = False,
    ) -> tuple[Rotation, float] | tuple[Rotation, float, torch.Tensor]:
        R"""Estimate a rotation to optimally align two sets of vectors.

        Find a rotation between frames A and B which best aligns a set of
        vectors `a` and `b` observed in these frames. The following loss
        function is minimized to solve for the rotation matrix :math:`R`:

        .. math::

            L(R) = \\frac{1}{2} \\sum_{i = 1}^{n} w_i \\lVert \\mathbf{a}_i -
            R \\mathbf{b}_i \\rVert^2 ,

        where :math:`w_i`'s are the `weights` corresponding to each vector.

        The rotation is estimated with Kabsch algorithm [1]_, and solves what
        is known as the "pointing problem", or "Wahba's problem" [2]_.

        There are two special cases. The first is if a single vector is given
        for `a` and `b`, in which the shortest distance rotation that aligns
        `b` to `a` is returned. The second is when one of the weights is infinity.
        In this case, the shortest distance rotation between the primary infinite weight
        vectors is calculated as above. Then, the rotation about the aligned primary
        vectors is calculated such that the secondary vectors are optimally
        aligned per the above loss function. The result is the composition
        of these two rotations. The result via this process is the same as the
        Kabsch algorithm as the corresponding weight approaches infinity in
        the limit. For a single secondary vector this is known as the
        "align-constrain" algorithm [3]_.

        For both special cases (single vectors or an infinite weight), the
        sensitivity matrix does not have physical meaning and an error will be
        raised if it is requested. For an infinite weight, the primary vectors
        act as a constraint with perfect alignment, so their contribution to
        `rssd` will be forced to 0 even if they are of different lengths.

        Parameters
        ----------
        a
            Vector components observed in initial frame A. Each row of `a`
            denotes a vector.
        b
            Vector components observed in another frame B. Each row of `b`
            denotes a vector.
        weights
            Weights describing the relative importance of the vector
            observations. If None (default), then all values in `weights` are
            assumed to be 1. One and only one weight may be infinity, and
            weights must be positive.
        return_sensitivity
            Whether to return the sensitivity matrix.
        allow_improper
            If True, allow improper rotations to be returned. If False (default),
            then the rotation is restricted to be proper.

        Returns
        -------
        rotation
            Best estimate of the rotation that transforms `b` to `a`.
        rssd
            Square root of the weighted sum of the squared distances between the given sets of
            vectors
            after alignment.
        sensitivity_matrix
            Sensitivity matrix of the estimated rotation estimate as explained
            in Notes.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Kabsch_algorithm
        .. [2] https://en.wikipedia.org/wiki/Wahba%27s_problem
        .. [3] Magner, Robert,
                "Extending target tracking capabilities through trajectory and
                momentum setpoint optimization." Small Satellite Conference,
                2018.
        """
        a_tensor = torch.stack([torch.as_tensor(el) for el in a]) if isinstance(a, Sequence) else torch.as_tensor(a)
        b_tensor = torch.stack([torch.as_tensor(el) for el in b]) if isinstance(b, Sequence) else torch.as_tensor(b)
        dtype = torch.promote_types(a_tensor.dtype, b_tensor.dtype)
        if not dtype.is_floating_point:
            # boolean or integer inputs will result in float32
            dtype = torch.float32
        a_tensor = torch.atleast_2d(a_tensor).to(dtype=dtype)
        b_tensor = torch.atleast_2d(b_tensor).to(dtype=dtype)
        if weights is None:
            weights_tensor = a_tensor.new_ones(a_tensor.shape[:-1], dtype=dtype)
        else:
            weights_tensor = torch.atleast_1d(torch.as_tensor(weights, dtype=dtype))

        if a_tensor.ndim > 2 or b_tensor.ndim > 2 or weights_tensor.ndim > 1:
            raise NotImplementedError('Batched inputs are not supported.')

        if return_sensitivity:
            rot_matrix, rssd, sensitivity = _align_vectors(a_tensor, b_tensor, weights_tensor, True, allow_improper)
            return cls.from_matrix(rot_matrix), rssd, sensitivity
        else:
            rot_matrix, rssd = _align_vectors(a_tensor, b_tensor, weights_tensor, False, allow_improper)
            return cls.from_matrix(rot_matrix), rssd

    @property
    def shape(self) -> torch.Size:
        """Return the batch shape of the Rotation."""
        if self._single:
            return torch.Size()
        return self._quaternions.shape[:-1]

    def __bool__(self):
        """Comply with Python convention for objects to be True.

        Required because `Rotation.__len__()` is defined and not always
        truthy.
        """
        return True

    def __len__(self) -> int:
        """Return the leading dimensions size of the batched Rotation."""
        if self._single:
            raise TypeError('Single rotation has no len().')
        return self.shape[0]

    def __repr__(self):
        """Return String Representation of the Rotation."""
        if self._single and not self._is_improper:
            return f'Rotation({self._quaternions.tolist()})'
        elif self._single and self._is_improper:
            return f'improper Rotation({self._quaternions.tolist()})'
        elif self._is_improper.all():
            return f'{tuple(self.shape)}-batched improper Rotation()'
        elif self._is_improper.any():
            return f'{tuple(self.shape)}-batched (mixed proper/improper) Rotation()'
        else:
            return f'{tuple(self.shape)}-batched Rotation()'

    def mean(
        self,
        weights: torch.Tensor | NestedSequence[float] | None = None,
        dim: None | int | Sequence[int] = None,
        keepdim: bool = False,
    ) -> Self:
        r"""Get the mean of the rotations.

        The mean used is the chordal L2 mean (also called the projected or
        induced arithmetic mean) [HAR2013]_. If ``A`` is a set of rotation matrices,
        then the mean ``M`` is the rotation matrix that minimizes the
        following loss function:
        :math:`L(M) = \sum_{i = 1}^{n} w_i \lVert \mathbf{A}_i - \mathbf{M} \rVert^2`,

        where :math:`w_i`'s are the `weights` corresponding to each matrix.

        Optionally, if A is a set of Rotation matrices with multiple batch dimensions,
        the dimensions to reduce over can be specified.

        If the rotations contains improper, the mean will be computed without
        considering the improper and the result will contain a reflection if
        the weighted majority of the rotations over which the mean is taken
        have improper.

        Parameters
        ----------
        weights
            Weights describing the relative importance of the rotations. If
            None (default), then all values in `weights` are assumed to be
            equal.
        dim
            Batch Dimensions to reduce over. None will always return a single Rotation.
        keepdim
            Keep reduction dimensions as length-1 dimensions.


        Returns
        -------
        mean : `Rotation` instance
            Object containing the mean of the rotations in the current
            instance.

        References
        ----------
        .. [HAR2013] Hartley R, Li H (2013) Rotation Averaging. International Journal of Computer Vision (103)
           https://link.springer.com/article/10.1007/s11263-012-0601-0

        """
        if weights is None:
            weights = torch.ones(*self.shape)
        else:
            weights = torch.as_tensor(weights)
            weights = weights.expand(self.shape)

            if torch.any(weights < 0):
                raise ValueError('`weights` must be non-negative.')

        if isinstance(dim, Sequence):
            dim = tuple(dim)

        modal_improper = (weights * self._is_improper).sum(dim=dim, keepdim=keepdim) > 0.5 * weights.sum(
            dim=dim, keepdim=keepdim
        )

        quaternions = torch.as_tensor(self._quaternions)
        if dim is None:
            quaternions = quaternions.reshape(-1, 4)
            weights = weights.reshape(-1)
            dim = list(range(len(self.shape)))
        else:
            dim = (
                [d % (quaternions.ndim - 1) for d in dim]
                if isinstance(dim, Sequence)
                else [dim % (quaternions.ndim - 1)]
            )
            batch_dims = [i for i in range(quaternions.ndim - 1) if i not in dim]
            permute_dims = (*batch_dims, *dim)
            quaternions = quaternions.permute(*permute_dims, -1).flatten(start_dim=len(batch_dims), end_dim=-2)
            weights = weights.permute(permute_dims).flatten(start_dim=len(batch_dims))
        k = (weights.unsqueeze(-2) * quaternions.mT) @ quaternions
        _, v = torch.linalg.eigh(k)
        mean_quaternions = v[..., -1]
        if keepdim:
            # unsqueeze the dimensions we removed in the reshape and product
            for d in sorted(dim):
                mean_quaternions = mean_quaternions.unsqueeze(d)

        return self.__class__(mean_quaternions, inversion=modal_improper, normalize=False)

    def reshape(self, *shape: int | Sequence[int]) -> Self:
        """Reshape the Rotation object in the batch dimensions.

        Parameters
        ----------
        shape
            The new shape of the Rotation object.

        Returns
        -------
        reshaped
            The reshaped Rotation object.
        """
        newshape = []
        for s in shape:
            if isinstance(s, int):
                newshape.append(s)
            else:
                newshape.extend(s)
        return self.__class__(
            self._quaternions.reshape(*newshape, 4), inversion=self._is_improper.reshape(newshape), copy=True
        )
