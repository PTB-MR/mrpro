"""Tests of the Rotation class."""
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

import copy
import math
import pickle
from itertools import permutations
from math import sqrt

import numpy as np
import pytest
import torch
from mrpro.data import Rotation, SpatialDimension
from mrpro.data.Rotation import AXIS_ORDER
from scipy.stats import special_ortho_group

from tests import RandomGenerator


def _norm(x):
    return torch.linalg.norm(x.float(), dim=-1, keepdim=True)


def test_from_quats():
    x = torch.tensor([[3, 4, 0, 0], [5, 12, 0, 0]])
    r = Rotation.from_quat(x)
    expected_quat = x / _norm(x)
    torch.testing.assert_close(r.as_quat(), expected_quat)


def test_from_single_1d_quaternion():
    x = torch.tensor([3, 4, 0, 0])
    r = Rotation.from_quat(x)
    expected_quat = x / _norm(x)
    torch.testing.assert_close(r.as_quat(), expected_quat)


def test_from_single_2d_quaternion():
    x = torch.tensor([[3, 4, 0, 0]])
    r = Rotation.from_quat(x)
    expected_quat = x / _norm(x)
    torch.testing.assert_close(r.as_quat(), expected_quat)


def test_from_square_quat_matrix():
    # Ensure proper norm array broadcasting
    x = torch.tensor(
        [
            [3, 0, 0, 4],
            [5, 0, 12, 0],
            [0, 0, 0, 1],
            [-1, -1, -1, 1],
            [0, 0, 0, -1],  # Check double cover
            [-1, -1, -1, -1],  # Check double cover
        ]
    )
    r = Rotation.from_quat(x)
    expected_quat = x / _norm(x)
    torch.testing.assert_close(r.as_quat(), expected_quat)


def test_quat_double_to_canonical_single_cover():
    x = torch.tensor([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1], [-1, -1, -1, -1]])
    r = Rotation.from_quat(x)
    expected_quat = torch.abs(x) / _norm(x)
    torch.testing.assert_close(r.as_quat(canonical=True), expected_quat)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_quat_double_cover():
    # See the scipy Rotation.from_quat() docstring for scope of the quaternion
    # double cover property.
    # Check from_quat and as_quat(canonical=False)
    q = torch.tensor([0.0, 0.0, 0.0, -1.0])
    r = Rotation.from_quat(q)
    assert (q == r.as_quat(canonical=False)).all()

    # Check composition and inverse
    q = torch.tensor([1.0, 0.0, 0.0, 1.0]) / sqrt(2)  # 90 deg rotation about x
    r = Rotation.from_quat(q)
    r3 = r * r * r
    torch.testing.assert_close(r.as_quat(canonical=False) * sqrt(2), torch.tensor([1, 0, 0, 1.0]))
    torch.testing.assert_close(r.inv().as_quat(canonical=False) * sqrt(2), torch.tensor([-1, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(r3.as_quat(canonical=False) * sqrt(2), torch.tensor([1.0, 0.0, 0.0, -1.0]))
    torch.testing.assert_close(r3.inv().as_quat(canonical=False) * sqrt(2), torch.tensor([-1.0, 0.0, 0.0, -1.0]))
    # these can be achieved with high precision even in float
    torch.testing.assert_close(
        (r * r.inv()).as_quat(canonical=False), torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=2e-16, rtol=0.0
    )
    torch.testing.assert_close(
        (r3 * r3.inv()).as_quat(canonical=False), torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=2e-16, rtol=0.0
    )
    torch.testing.assert_close(
        (r * r3).as_quat(canonical=False), torch.tensor([0.0, 0.0, 0.0, -1.0]), atol=2e-16, rtol=0.0
    )
    torch.testing.assert_close(
        (r.inv() * r3.inv()).as_quat(canonical=False), torch.tensor([0.0, 0.0, 0.0, -1.0]), atol=2e-16, rtol=0.0
    )


def test_malformed_1d_from_quat():
    with pytest.raises(ValueError):
        Rotation.from_quat(torch.tensor([1, 2, 3]))


def test_malformed_2d_from_quat():
    with pytest.raises(ValueError):
        Rotation.from_quat(torch.tensor([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]]))


def test_zero_norms_from_quat():
    x = torch.tensor([[3, 4, 0, 0], [0, 0, 0, 0], [5, 0, 12, 0]])
    with pytest.raises(ValueError):
        Rotation.from_quat(x)


def test_as_matrix_single_1d_quaternion():
    quat = torch.tensor([0, 0, 0, 1])
    mat = Rotation.from_quat(quat).as_matrix()
    # mat.shape == (3,3) due to 1d input
    torch.testing.assert_close(mat, torch.eye(3))


def test_as_matrix_single_2d_quaternion():
    quat = torch.tensor([[0, 0, 1, 1]])
    mat = Rotation.from_quat(quat).as_matrix()
    assert mat.shape == (1, 3, 3)
    expected_mat = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    torch.testing.assert_close(mat[0], expected_mat)


def test_as_matrix_from_square_input():
    quats = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0]])
    mat = Rotation.from_quat(quats).as_matrix()
    assert mat.shape == (4, 3, 3)

    expected0 = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).float()
    torch.testing.assert_close(mat[0], expected0)

    expected1 = torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).float()
    torch.testing.assert_close(mat[1], expected1)

    torch.testing.assert_close(mat[2], torch.eye(3))
    torch.testing.assert_close(mat[3], torch.eye(3))


def test_as_matrix_from_generic_input():
    quats = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1], [1, 2, 3, 4]]).float()
    mat = Rotation.from_quat(quats).as_matrix()
    assert mat.shape == (3, 3, 3)

    expected0 = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).float()
    torch.testing.assert_close(mat[0], expected0)

    expected1 = torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).float()
    torch.testing.assert_close(mat[1], expected1)

    expected2 = torch.tensor([[0.4, -2, 2.2], [2.8, 1, 0.4], [-1, 2, 2]]) / 3
    torch.testing.assert_close(mat[2], expected2)


def test_from_single_2d_matrix():
    mat = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).float()
    expected_quat = torch.tensor([0.5, 0.5, 0.5, 0.5])
    torch.testing.assert_close(Rotation.from_matrix(mat).as_quat(), expected_quat)


def test_from_single_3d_matrix():
    mat = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).reshape((1, 3, 3))
    expected_quat = torch.tensor([0.5, 0.5, 0.5, 0.5]).reshape((1, 4))
    torch.testing.assert_close(Rotation.from_matrix(mat).as_quat(), expected_quat)


def test_from_matrix_calculation():
    expected_quat = torch.tensor([1, 1, 6, 1]) / sqrt(39)
    mat = torch.tensor(
        [[-0.8974359, -0.2564103, 0.3589744], [0.3589744, -0.8974359, 0.2564103], [0.2564103, 0.3589744, 0.8974359]]
    )
    torch.testing.assert_close(Rotation.from_matrix(mat).as_quat(), expected_quat)
    torch.testing.assert_close(Rotation.from_matrix(mat.reshape((1, 3, 3))).as_quat(), expected_quat.reshape((1, 4)))

    torch.testing.assert_close(
        Rotation.from_matrix(mat.reshape((1, 3, 3)).expand(2, 3, 3)).as_quat(),
        expected_quat.reshape((1, 4)).expand(2, 4),
    )

    torch.testing.assert_close(
        Rotation.from_matrix(mat.reshape((1, 1, 3, 3))).as_quat(), expected_quat.reshape((1, 1, 4))
    )


def test_matrix_calculation_pipeline():
    mat = torch.as_tensor(special_ortho_group.rvs(3, size=10, random_state=0))
    torch.testing.assert_close(Rotation.from_matrix(mat).as_matrix(), mat)


def test_from_matrix_ortho_output():
    rnd = RandomGenerator(0)
    mat = rnd.float32_tensor((100, 3, 3))
    ortho_mat = Rotation.from_matrix(mat).as_matrix()

    mult_result = torch.einsum('...ij,...jk->...ik', ortho_mat, ortho_mat.permute((0, 2, 1)))

    eye = torch.eye(3)[None, ...].expand(100, 3, 3)

    torch.testing.assert_close(mult_result, eye)


def test_from_1d_single_rotvec():
    rotvec = torch.tensor([1, 0, 0])
    expected_quat = torch.tensor([0.4794255, 0, 0, 0.8775826])
    result = Rotation.from_rotvec(rotvec)
    torch.testing.assert_close(result.as_quat(), expected_quat)


def test_from_2d_single_rotvec():
    rotvec = torch.tensor([[1, 0, 0]])
    expected_quat = torch.tensor([[0.4794255, 0, 0, 0.8775826]])
    result = Rotation.from_rotvec(rotvec)
    torch.testing.assert_close(result.as_quat(), expected_quat)


def test_from_generic_rotvec():
    rotvec = [[1.0, 2.0, 2.0], [1.0, -1.0, 0.5], [0.0, 0.0, 0.0]]
    expected_quat = torch.tensor(
        [[0.3324983, 0.6649967, 0.6649967, 0.0707372], [0.4544258, -0.4544258, 0.2272129, 0.7316889], [0, 0, 0, 1]]
    )
    torch.testing.assert_close(Rotation.from_rotvec(rotvec).as_quat(), expected_quat)


def test_from_rotvec_small_angle():
    rotvec = torch.tensor([[5e-4 / sqrt(3), -5e-4 / sqrt(3), 5e-4 / sqrt(3)], [0.2, 0.3, 0.4], [0, 0, 0]])

    quat = Rotation.from_rotvec(rotvec).as_quat()
    # cos(theta/2) ~~ 1 for small theta
    torch.testing.assert_close(quat[0, 3], torch.tensor(1.0))
    # sin(theta/2) / theta ~~ 0.5 for small theta
    torch.testing.assert_close(quat[0, :3], rotvec[0] * 0.5)

    torch.testing.assert_close(quat[1, 3], torch.tensor(0.9639685))
    torch.testing.assert_close(
        quat[1, :3], torch.tensor([0.09879603932153465, 0.14819405898230198, 0.19759207864306931])
    )

    assert (quat[2] == torch.tensor([0, 0, 0, 1])).all()


def test_degrees_from_rotvec():
    rotvec1 = torch.tensor([1.0, 1.0, 1.0]) / 3 ** (1 / 3)
    rot1 = Rotation.from_rotvec(rotvec1, degrees=True)
    quat1 = rot1.as_quat()

    rotvec2 = torch.deg2rad(rotvec1)
    rot2 = Rotation.from_rotvec(rotvec2)
    quat2 = rot2.as_quat()

    torch.testing.assert_close(quat1, quat2)


def test_malformed_1d_from_rotvec():
    with pytest.raises(ValueError, match='Expected `rot_vec` to have shape'):
        Rotation.from_rotvec([1, 2])


def test_malformed_2d_from_rotvec():
    with pytest.raises(ValueError, match='Expected `rot_vec` to have shape'):
        Rotation.from_rotvec([[1, 2, 3, 4], [5, 6, 7, 8]])


def test_as_generic_rotvec():
    quat = torch.tensor([[1, 2, -1, 0.5], [1, -1, 1, 0.0003], [0, 0, 0, 1]])
    quat /= torch.linalg.vector_norm(quat, keepdim=True, dim=-1)

    rotvec = Rotation.from_quat(quat).as_rotvec()
    angle = torch.linalg.vector_norm(rotvec, dim=-1)

    torch.testing.assert_close(quat[:, 3], torch.cos(angle / 2))
    torch.testing.assert_close(torch.linalg.cross(rotvec, quat[:, :3]), torch.zeros((3, 3)))


def test_as_rotvec_single_1d_input():
    quat = torch.tensor([1, 2, -3, 2])
    expected_rotvec = torch.tensor([0.5772381, 1.1544763, -1.7317144])

    actual_rotvec = Rotation.from_quat(quat).as_rotvec()

    assert actual_rotvec.shape == (3,)
    torch.testing.assert_close(actual_rotvec, expected_rotvec)


def test_as_rotvec_single_2d_input():
    quat = torch.tensor([[1, 2, -3, 2]])
    expected_rotvec = torch.tensor([[0.5772381, 1.1544763, -1.7317144]])

    actual_rotvec = Rotation.from_quat(quat).as_rotvec()

    assert actual_rotvec.shape == (1, 3)
    torch.testing.assert_close(actual_rotvec, expected_rotvec)


def test_as_rotvec_degrees():
    # x->y, y->z, z->x
    mat = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    rot = Rotation.from_matrix(mat)
    rotvec = rot.as_rotvec(degrees=True)
    angle = torch.linalg.vector_norm(rotvec)
    torch.testing.assert_close(angle, torch.tensor(120.0))
    torch.testing.assert_close(rotvec[0], rotvec[1])
    torch.testing.assert_close(rotvec[1], rotvec[2])


def test_rotvec_calc_pipeline():
    # Include small angles
    rotvec = torch.tensor([[0, 0, 0], [1, -1, 2], [-3e-4, 3.5e-4, 7.5e-5]])
    torch.testing.assert_close(Rotation.from_rotvec(rotvec).as_rotvec(), rotvec)
    torch.testing.assert_close(Rotation.from_rotvec(rotvec, degrees=True).as_rotvec(degrees=True), rotvec)


# all mrp tests from scipy.spatial.transform.Rotation removed
# as we haven't implemented as_mrp nor from_mrp.

# all davenport  tests from scipy.spatial.transform.Rotation removed
# as we haven't implemented as_davenport nor from_davenport.


def test_from_euler_single_rotation():
    lastaxis = AXIS_ORDER[-1]
    quat = Rotation.from_euler(lastaxis.lower(), 90, degrees=True).as_quat()
    expected_quat = torch.tensor([0, 0, 1, 1]) / sqrt(2)
    torch.testing.assert_close(quat, expected_quat)


def test_single_intrinsic_extrinsic_rotation():
    lastaxis = AXIS_ORDER[-1]
    extrinsic = Rotation.from_euler(lastaxis.lower(), 90, degrees=True).as_matrix()
    intrinsic = Rotation.from_euler(lastaxis.upper(), 90, degrees=True).as_matrix()
    torch.testing.assert_close(extrinsic, intrinsic)


def test_from_euler_rotation_order():
    # Intrinsic rotation is same as extrinsic with order reversed
    rnd = RandomGenerator(0)
    axes = AXIS_ORDER
    a = rnd.float32_tensor(low=0, high=180, size=(6, 3))
    b = torch.flip(a, (-1,))
    x = Rotation.from_euler(axes.lower(), a, degrees=True).as_quat()
    y = Rotation.from_euler(axes[::-1].upper(), b, degrees=True).as_quat()
    torch.testing.assert_close(x, y)


def test_from_euler_elementary_extrinsic_rotation():
    # Simple test to check if extrinsic rotations are implemented correctly
    axes = AXIS_ORDER[2] + AXIS_ORDER[0]
    mat = Rotation.from_euler(axes, [90, 90], degrees=True).as_matrix()
    expected_mat = torch.tensor([[0, -1, 0], [0, 0, -1], [1, 0, 0]]).float()
    torch.testing.assert_close(mat, expected_mat)


def test_from_euler_intrinsic_rotation_201():
    angles = [[30, 60, 45], [30, 60, 30], [45, 30, 60]]
    axes = (AXIS_ORDER[2] + AXIS_ORDER[0] + AXIS_ORDER[1]).upper()
    mat = Rotation.from_euler(axes, angles, degrees=True).as_matrix()

    torch.testing.assert_close(
        mat[0],
        torch.tensor(
            [[0.3061862, -0.2500000, 0.9185587], [0.8838835, 0.4330127, -0.1767767], [-0.3535534, 0.8660254, 0.3535534]]
        ),
    )

    torch.testing.assert_close(
        mat[1],
        torch.tensor(
            [[0.5334936, -0.2500000, 0.8080127], [0.8080127, 0.4330127, -0.3995191], [-0.2500000, 0.8660254, 0.4330127]]
        ),
    )

    torch.testing.assert_close(
        mat[2],
        torch.tensor(
            [[0.0473672, -0.6123725, 0.7891491], [0.6597396, 0.6123725, 0.4355958], [-0.7500000, 0.5000000, 0.4330127]]
        ),
    )


def test_from_euler_intrinsic_rotation_202():
    angles = [[30, 60, 45], [30, 60, 30], [45, 30, 60]]
    axes = (AXIS_ORDER[2] + AXIS_ORDER[0] + AXIS_ORDER[2]).upper()
    mat = Rotation.from_euler(axes, angles, degrees=True).as_matrix()
    expect0 = torch.tensor(
        [
            [0.43559574, -0.78914913, 0.4330127],
            [0.65973961, -0.04736717, -0.750000],
            [0.61237244, 0.61237244, 0.50000000],
        ]
    )
    torch.testing.assert_close(mat[0], expect0)
    expect1 = torch.tensor(
        [
            [0.6250000, -0.64951905, 0.4330127],
            [0.64951905, 0.1250000, -0.7500000],
            [0.4330127, 0.75000000, 0.50000000],
        ]
    )
    torch.testing.assert_close(
        mat[1],
        expect1,
    )
    expect2 = torch.tensor(
        [
            [-0.1767767, -0.91855865, 0.35355339],
            [0.88388348, -0.30618622, -0.35355339],
            [0.4330127, 0.250000000, 0.8660254],
        ]
    )
    torch.testing.assert_close(mat[2], expect2)


def test_from_euler_extrinsic_rotation_201():
    angles = [[30, 60, 45], [30, 60, 30], [45, 30, 60]]
    axes = (AXIS_ORDER[2] + AXIS_ORDER[0] + AXIS_ORDER[1]).lower()
    mat = Rotation.from_euler(axes, angles, degrees=True).as_matrix()

    torch.testing.assert_close(
        mat[0],
        torch.tensor(
            [
                [0.91855865, 0.1767767, 0.35355339],
                [0.25000000, 0.4330127, -0.8660254],
                [-0.30618622, 0.88388348, 0.35355339],
            ]
        ),
    )

    torch.testing.assert_close(
        mat[1],
        torch.tensor(
            [
                [0.96650635, -0.0580127, 0.2500000],
                [0.25000000, 0.4330127, -0.8660254],
                [-0.0580127, 0.89951905, 0.4330127],
            ]
        ),
    )

    torch.testing.assert_close(
        mat[2],
        torch.tensor(
            [
                [0.65973961, -0.04736717, 0.7500000],
                [0.61237244, 0.61237244, -0.5000000],
                [-0.43559574, 0.78914913, 0.4330127],
            ]
        ),
    )


def test_from_euler_extrinsic_rotation_202():
    angles = [[30, 60, 45], [30, 60, 30], [45, 30, 60]]
    axes = (AXIS_ORDER[2] + AXIS_ORDER[0] + AXIS_ORDER[2]).lower()

    mat = Rotation.from_euler(axes, angles, degrees=True).as_matrix()

    torch.testing.assert_close(
        mat[0],
        torch.tensor(
            [
                [0.43559574, -0.65973961, 0.61237244],
                [0.78914913, -0.04736717, -0.61237244],
                [0.4330127, 0.75000000, 0.500000],
            ]
        ),
    )

    torch.testing.assert_close(
        mat[1],
        torch.tensor(
            [
                [0.62500000, -0.64951905, 0.4330127],
                [0.64951905, 0.12500000, -0.750000],
                [0.4330127, 0.75000000, 0.500000],
            ]
        ),
    )

    torch.testing.assert_close(
        mat[2],
        torch.tensor(
            [
                [-0.1767767, -0.88388348, 0.4330127],
                [0.91855865, -0.30618622, -0.250000],
                [0.35355339, 0.35355339, 0.8660254],
            ]
        ),
    )


def _test_stats(error: torch.Tensor, mean_max: float, rms_max: float) -> None:
    # helper function for mean error tests
    mean = torch.mean(error, dim=0)
    std = torch.std(error, dim=0)
    rms = torch.hypot(mean, std)
    assert torch.all(torch.abs(mean) < mean_max)
    assert torch.all(rms < rms_max)


@pytest.mark.parametrize('seq_tuple', permutations('xyz'), ids=str)
@pytest.mark.parametrize('intrinsic', [False, True])
def test_as_euler_asymmetric_axes(seq_tuple, intrinsic):
    rnd = RandomGenerator(0)
    n = 1000
    angles = torch.empty((n, 3), dtype=torch.float64)
    angles[:, 0] = rnd.float64_tensor(low=-torch.pi, high=torch.pi, size=(n,))
    angles[:, 1] = rnd.float64_tensor(low=-torch.pi / 2, high=torch.pi / 2, size=(n,))
    angles[:, 2] = rnd.float64_tensor(low=-torch.pi, high=torch.pi, size=(n,))
    seq = ''.join(seq_tuple)
    if intrinsic:
        # Extrinsic rotation (wrt to global world) at lower case
        # intrinsic (WRT the object itself) lower case.
        seq = seq.upper()
    rotation = Rotation.from_euler(seq, angles)
    angles_quat = rotation.as_euler(seq)
    torch.testing.assert_close(angles, angles_quat, atol=0, rtol=1e-11)
    _test_stats(angles_quat - angles, 1e-15, 1e-14)


@pytest.mark.parametrize('seq_tuple', permutations('xyz'), ids=str)
@pytest.mark.parametrize('intrinsic', [False, True])
def test_as_euler_symmetric_axes(seq_tuple, intrinsic):
    rnd = RandomGenerator(0)
    n = 1000
    angles = torch.empty((n, 3), dtype=torch.float64)
    angles[:, 0] = rnd.float64_tensor(low=-torch.pi, high=torch.pi, size=(n,))
    angles[:, 1] = rnd.float64_tensor(low=0, high=torch.pi, size=(n,))
    angles[:, 2] = rnd.float64_tensor(low=-torch.pi, high=torch.pi, size=(n,))

    # Rotation of the form A/B/A are rotation around symmetric axes
    seq = ''.join([seq_tuple[0], seq_tuple[1], seq_tuple[0]])
    if intrinsic:
        seq = seq.upper()
    rotation = Rotation.from_euler(seq, angles)
    angles_quat = rotation.as_euler(seq)

    torch.testing.assert_close(angles, angles_quat, atol=0, rtol=1e-11)
    _test_stats(angles_quat - angles, 1e-16, 1e-14)


@pytest.mark.parametrize('seq_tuple', permutations('xyz'), ids=str)
@pytest.mark.parametrize('intrinsic', [False, True])
def test_as_euler_degenerate_asymmetric_axes(seq_tuple, intrinsic):
    # Since we cannot check for angle equality, we check for rotation matrix
    # equality
    angles = torch.tensor([[45, 90, 35], [35, -90, 20], [35, 90, 25], [25, -90, 15]])

    seq = ''.join(seq_tuple)
    if intrinsic:
        # Extrinsic rotation (wrt to global world) at lower case
        # Intrinsic (WRT the object itself) upper case.
        seq = seq.upper()
    rotation = Rotation.from_euler(seq, angles, degrees=True)
    mat_expected = rotation.as_matrix()

    # with pytest.warns(UserWarning, match='Gimbal lock'):
    angle_estimates = rotation.as_euler(seq, degrees=True)
    mat_estimated = Rotation.from_euler(seq, angle_estimates, degrees=True).as_matrix()

    torch.testing.assert_close(mat_expected, mat_estimated)


@pytest.mark.parametrize('seq_tuple', permutations('xyz'), ids=str)
@pytest.mark.parametrize('intrinsic', [False, True])
def test_as_euler_degenerate_symmetric_axes(seq_tuple, intrinsic):
    # Since we cannot check for angle equality, we check for rotation matrix
    # equality
    angles = torch.tensor([[15, 0, 60], [35, 0, 75], [60, 180, 35], [15, -180, 25]])

    # Rotation of the form A/B/A are rotation around symmetric axes
    seq = ''.join([seq_tuple[0], seq_tuple[1], seq_tuple[0]])
    if intrinsic:
        # Extrinsic rotation (wrt to global world) at lower case
        # Intrinsic (WRT the object itself) upper case.
        seq = seq.upper()
    rotation = Rotation.from_euler(seq, angles, degrees=True)
    mat_expected = rotation.as_matrix()

    # with pytest.warns(UserWarning, match='Gimbal lock'):
    angle_estimates = rotation.as_euler(seq, degrees=True)
    mat_estimated = Rotation.from_euler(seq, angle_estimates, degrees=True).as_matrix()

    torch.testing.assert_close(mat_expected, mat_estimated)


def test_inv():
    rnd = np.random.RandomState(0)
    n = 10
    p = Rotation.random(num=n, random_state=rnd)
    q = p.inv()

    p_mat = p.as_matrix()
    q_mat = q.as_matrix()
    result1 = torch.einsum('...ij,...jk->...ik', p_mat, q_mat)
    result2 = torch.einsum('...ij,...jk->...ik', q_mat, p_mat)

    eye3d = torch.empty((n, 3, 3))
    eye3d[:] = torch.eye(3)

    torch.testing.assert_close(result1, eye3d)
    torch.testing.assert_close(result2, eye3d)


def test_inv_single_rotation():
    rnd = np.random.RandomState(0)
    p = Rotation.random(random_state=rnd)
    q = p.inv()

    p_mat = p.as_matrix()
    q_mat = q.as_matrix()
    res1 = torch.matmul(p_mat, q_mat)
    res2 = torch.matmul(q_mat, p_mat)

    eye = torch.eye(3)

    torch.testing.assert_close(res1, eye)
    torch.testing.assert_close(res2, eye)

    x = Rotation.random(num=1, random_state=rnd)
    y = x.inv()

    x_matrix = x.as_matrix()
    y_matrix = y.as_matrix()
    result1 = torch.einsum('...ij,...jk->...ik', x_matrix, y_matrix)
    result2 = torch.einsum('...ij,...jk->...ik', y_matrix, x_matrix)

    eye3d = torch.empty((1, 3, 3))
    eye3d[:] = torch.eye(3)

    torch.testing.assert_close(result1, eye3d)
    torch.testing.assert_close(result2, eye3d)


@pytest.mark.parametrize('n', [10, (2, 3, 4), 1])
def test_identity_magnitude(n):
    torch.testing.assert_close(Rotation.identity(n).magnitude(), torch.zeros(n))
    torch.testing.assert_close(Rotation.identity(n).inv().magnitude(), torch.zeros(n))


def test_single_identity_magnitude():
    assert Rotation.identity().magnitude() == 0
    assert Rotation.identity().inv().magnitude() == 0


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_identity_invariance():
    n = 10
    p = Rotation.random(n, random_state=0)

    result = p * Rotation.identity(n)
    torch.testing.assert_close(p.as_quat(), result.as_quat())

    result = result * p.inv()
    torch.testing.assert_close(result.magnitude(), torch.zeros(n))


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_single_identity_invariance():
    n = 10
    p = Rotation.random(n, random_state=0)

    result = p * Rotation.identity()
    torch.testing.assert_close(p.as_quat(), result.as_quat())

    result = result * p.inv()
    torch.testing.assert_close(result.magnitude(), torch.zeros(n))


def test_magnitude():
    r = Rotation.from_quat(torch.eye(4))
    result = r.magnitude()
    torch.testing.assert_close(result, torch.tensor([torch.pi, torch.pi, torch.pi, 0]))

    r = Rotation.from_quat(-torch.eye(4))
    result = r.magnitude()
    torch.testing.assert_close(result, torch.tensor([torch.pi, torch.pi, torch.pi, 0]))


def test_magnitude_single_rotation():
    r = Rotation.from_quat(torch.eye(4))
    result1 = r[0].magnitude()
    torch.testing.assert_close(result1, torch.tensor(torch.pi))

    result2 = r[3].magnitude()
    torch.testing.assert_close(result2, torch.tensor(0.0))


def test_approx_equal():
    rng = np.random.RandomState(0)
    p = Rotation.random(10, random_state=rng)
    q = Rotation.random(10, random_state=rng)
    r = p @ q.inv()
    r_mag = r.magnitude()
    atol = torch.median(r_mag)  # ensure we get mix of Trues and Falses
    assert (p.approx_equal(q, atol) == (r_mag < atol)).all()


def test_approx_equal_single_rotation():
    # also tests passing single argument to approx_equal
    p = Rotation.from_rotvec([0, 0, 1e-9])  # less than atol of 1e-8
    q = Rotation.from_quat(torch.eye(4))
    assert p.approx_equal(q[3])
    assert not p.approx_equal(q[0])

    # test passing atol and using degrees
    assert not p.approx_equal(q[3], atol=1e-10)
    assert not p.approx_equal(q[3], atol=1e-8, degrees=True)


def test_apply_single_spatialdim():
    vec = SpatialDimension(1.0, 2.0, 3.0)
    mat = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).float()
    r_1d = Rotation.from_matrix(mat)
    r_2d = Rotation.from_matrix(mat.unsqueeze(0))
    v_1d = r_1d(vec)
    v_2d = r_2d(vec)

    assert isinstance(v_1d, SpatialDimension)
    torch.testing.assert_close(getattr(v_1d, AXIS_ORDER[0]), torch.tensor(-2.0))
    torch.testing.assert_close(getattr(v_1d, AXIS_ORDER[1]), torch.tensor(1.0))
    torch.testing.assert_close(getattr(v_1d, AXIS_ORDER[2]), torch.tensor(3.0))

    assert isinstance(v_2d, SpatialDimension)
    torch.testing.assert_close(getattr(v_2d, AXIS_ORDER[0]), torch.tensor([-2.0]))
    torch.testing.assert_close(getattr(v_2d, AXIS_ORDER[1]), torch.tensor([1.0]))
    torch.testing.assert_close(getattr(v_2d, AXIS_ORDER[2]), torch.tensor([3.0]))


def test_apply_single_rotation_single_point():
    mat = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    r_1d = Rotation.from_matrix(mat)
    r_2d = Rotation.from_matrix(mat.unsqueeze(0))

    v_1d = torch.tensor([1, 2, 3])
    v_2d = v_1d.unsqueeze(0)
    v1d_rotated = torch.tensor([-2, 1, 3]).float()
    v2d_rotated = v1d_rotated.unsqueeze(0)

    torch.testing.assert_close(r_1d(v_1d), v1d_rotated)
    torch.testing.assert_close(r_1d(v_2d), v2d_rotated)
    torch.testing.assert_close(r_2d(v_1d), v2d_rotated)
    torch.testing.assert_close(r_2d(v_2d), v2d_rotated)

    v1d_inverse = torch.tensor([2, -1, 3]).float()
    v2d_inverse = v1d_inverse.unsqueeze(0)

    torch.testing.assert_close(r_1d(v_1d, inverse=True), v1d_inverse)
    torch.testing.assert_close(r_1d(v_2d, inverse=True), v2d_inverse)
    torch.testing.assert_close(r_2d(v_1d, inverse=True), v2d_inverse)
    torch.testing.assert_close(r_2d(v_2d, inverse=True), v2d_inverse)


def test_apply_single_rotation_multiple_points():
    mat = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    r1 = Rotation.from_matrix(mat)
    r2 = Rotation.from_matrix(mat.unsqueeze(0))

    v = torch.tensor([[1, 2, 3], [4, 5, 6]])
    v_rotated = torch.tensor([[-2, 1, 3], [-5, 4, 6]]).float()

    torch.testing.assert_close(r1(v), v_rotated)
    torch.testing.assert_close(r2(v), v_rotated)

    v_inverse = torch.tensor([[2, -1, 3], [5, -4, 6]]).float()

    torch.testing.assert_close(r1(v, inverse=True), v_inverse)
    torch.testing.assert_close(r2(v, inverse=True), v_inverse)


def test_apply_multiple_rotations_single_point():
    mat = torch.empty((2, 3, 3))
    mat[0] = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    mat[1] = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    r = Rotation.from_matrix(mat)

    v1 = torch.tensor([1, 2, 3])
    v2 = v1.unsqueeze(0)

    v_rotated = torch.tensor([[-2, 1, 3], [1, -3, 2]]).float()

    torch.testing.assert_close(r(v1), v_rotated)
    torch.testing.assert_close(r(v2), v_rotated)

    v_inverse = torch.tensor([[2, -1, 3], [1, 3, -2]]).float()

    torch.testing.assert_close(r(v1, inverse=True), v_inverse)
    torch.testing.assert_close(r(v2, inverse=True), v_inverse)


def test_apply_multiple_rotations_multiple_points():
    mat = torch.empty((2, 3, 3))
    mat[0] = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    mat[1] = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    r = Rotation.from_matrix(mat)

    v = torch.tensor([[1, 2, 3], [4, 5, 6]])
    v_rotated = torch.tensor([[-2, 1, 3], [4, -6, 5]]).float()
    torch.testing.assert_close(r(v), v_rotated)

    v_inverse = torch.tensor([[2, -1, 3], [4, 6, -5]]).float()
    torch.testing.assert_close(r(v, inverse=True), v_inverse)


def test_getitem():
    mat = torch.empty((2, 3, 3))
    mat[0] = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    mat[1] = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    r = Rotation.from_matrix(mat)

    torch.testing.assert_close(r[0].as_matrix(), mat[0])
    torch.testing.assert_close(r[1].as_matrix(), mat[1])
    torch.testing.assert_close(r[:-1].as_matrix(), mat[0].unsqueeze(0))


def test_getitem_single():
    with pytest.raises(TypeError, match='not subscriptable'):
        Rotation.identity()[0]


def test_setitem_single():
    r = Rotation.identity()
    with pytest.raises(TypeError, match='not subscriptable'):
        r[0] = Rotation.identity()


def test_setitem_slice():
    rng = np.random.RandomState(seed=0)
    r1 = Rotation.random(10, random_state=rng)
    r2 = Rotation.random(5, random_state=rng)
    r1[1:6] = r2
    assert (r1[1:6].as_quat() == r2.as_quat()).all()


def test_setitem_integer():
    rng = np.random.RandomState(seed=0)
    r1 = Rotation.random(10, random_state=rng)
    r2 = Rotation.random(random_state=rng)
    r1[1] = r2
    assert (r1[1].as_quat() == r2.as_quat()).all()


def test_setitem_wrong_type():
    r = Rotation.random(10, random_state=0)
    with pytest.raises(TypeError, match='Rotation object'):
        r[0] = 1


def test_n_rotations():
    mat = torch.empty((2, 3, 3))
    mat[0] = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    mat[1] = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    r = Rotation.from_matrix(mat)

    assert len(r) == 2
    assert len(r[:-1]) == 1


def test_random_rotation_shape():
    rnd = np.random.RandomState(0)
    assert Rotation.random(random_state=rnd).as_quat().shape == (4,)
    assert Rotation.random(None, random_state=rnd).as_quat().shape == (4,)
    assert Rotation.random(1, random_state=rnd).as_quat().shape == (1, 4)
    assert Rotation.random(5, random_state=rnd).as_quat().shape == (5, 4)
    assert Rotation.random((2, 3), random_state=rnd).as_quat().shape == (2, 3, 4)


def test_align_vectors_no_rotation():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]]).float()
    y = x.clone()

    r, rssd = Rotation.align_vectors(x, y)
    torch.testing.assert_close(r.as_matrix(), torch.eye(3))
    assert math.isclose(rssd, 0.0, abs_tol=1e-6, rel_tol=1e-4)


def test_align_vectors_no_noise():
    rnd = np.random.RandomState(0)
    c = Rotation.random(random_state=rnd)
    b = torch.tensor(rnd.normal(size=(5, 3)))
    a = c(b)

    est, rssd = Rotation.align_vectors(a, b)
    torch.testing.assert_close(c.as_quat().double(), est.as_quat().double())
    assert math.isclose(rssd, 0.0, abs_tol=1e-6, rel_tol=1e-4)


def test_align_vectors_improper_rotation():
    """Test for scipy issue #10444"""
    x = torch.tensor([[0.89299824, -0.44372674, 0.0752378], [0.60221789, -0.47564102, -0.6411702]]).double()
    y = torch.tensor([[0.02386536, -0.82176463, 0.5693271], [-0.27654929, -0.95191427, -0.1318321]]).double()

    est, rssd = Rotation.align_vectors(x, y)
    torch.testing.assert_close(x, est(y), atol=1e-7, rtol=0)
    torch.testing.assert_close(rssd, torch.tensor(0.0, dtype=torch.float64), atol=1e-7, rtol=0)


def test_align_vectors_rssd_sensitivity():
    rssd_expected = 0.141421356237308
    sens_expected = torch.tensor([[0.2, 0.0, 0.0], [0.0, 1.5, 1.0], [0.0, 1.0, 1.0]])
    a = [[0, 1, 0], [0, 1, 1], [0, 1, 1]]
    b = [[1.0, 0.0, 0.0], [1.0, 1.1, 0.0], [1.0, 0.9, 0.0]]
    rot, rssd, sens = Rotation.align_vectors(a, b, return_sensitivity=True)
    assert math.isclose(rssd, rssd_expected, abs_tol=1e-6, rel_tol=1e-4)
    assert torch.allclose(sens, sens_expected, atol=1e-6, rtol=1e-4)


def test_align_vectors_scaled_weights():
    n = 10
    a = Rotation.random(n, random_state=0)([1, 0, 0])
    b = Rotation.random(n, random_state=1)([1, 0, 0])
    scale = 2

    est1, rssd1, cov1 = Rotation.align_vectors(a, b, torch.ones(n), return_sensitivity=True)
    est2, rssd2, cov2 = Rotation.align_vectors(a, b, scale * torch.ones(n), return_sensitivity=True)

    torch.testing.assert_close(est1.as_matrix(), est2.as_matrix())
    torch.testing.assert_close(sqrt(scale) * rssd1, rssd2, atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(cov1, cov2)


def test_align_vectors_noise():
    rnd = np.random.RandomState(0)
    n_vectors = 100
    rot = Rotation.random(random_state=rnd)
    vectors = torch.tensor(rnd.normal(size=(n_vectors, 3)), dtype=torch.float32)
    result = rot(vectors)

    # The paper adds noise as independently distributed angular errors
    sigma = np.deg2rad(1)
    tolerance = 1.5 * sigma
    noise = Rotation.from_rotvec(torch.tensor(rnd.normal(size=(n_vectors, 3), scale=sigma), dtype=torch.float32))

    # Attitude errors must preserve norm. Hence apply individual random
    # rotations to each vector.
    noisy_result = noise(result)

    est, rssd, cov = Rotation.align_vectors(noisy_result, vectors, return_sensitivity=True)

    # Use rotation compositions to find out closeness
    error_vector = (rot @ est.inv()).as_rotvec()
    torch.testing.assert_close(error_vector, torch.zeros(3), atol=tolerance, rtol=0)

    # Check error bounds using covariance matrix
    cov *= sigma
    torch.testing.assert_close(torch.diag(cov), torch.zeros(3), atol=tolerance, rtol=0)
    torch.testing.assert_close(torch.sum((noisy_result - est(vectors)) ** 2) ** 0.5, rssd)


def test_align_vectors_invalid_input():
    with pytest.raises(ValueError, match='Expected inputs to have same shapes'):
        Rotation.align_vectors([1, 2, 3, 4], [1, 2, 3])

    with pytest.raises(ValueError, match='Expected inputs to have same shapes'):
        Rotation.align_vectors([[1, 2, 3], [4, 5, 6]], [[1, 2, 3]])

    with pytest.raises(ValueError, match='Expected inputs to have shape'):
        Rotation.align_vectors([1, 2, 3, 4], [1, 2, 3, 4])

    with pytest.raises(ValueError, match='Invalid weights: expected shape'):
        Rotation.align_vectors([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]], weights=[1, 2, 3])

    with pytest.raises(ValueError, match='Invalid weights: expected shape'):
        Rotation.align_vectors([[1, 2, 3]], [[1, 2, 3]], weights=[-1])

    with pytest.raises(ValueError, match='Only one infinite weight is allowed'):
        Rotation.align_vectors([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]], weights=[torch.inf, torch.inf])

    with pytest.raises(ValueError, match='Cannot align zero length primary vectors'):
        Rotation.align_vectors([[0, 0, 0]], [[1, 2, 3]])

    with pytest.raises(ValueError, match='Cannot return sensitivity matrix'):
        Rotation.align_vectors(
            [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]], return_sensitivity=True, weights=[torch.inf, 1]
        )

    with pytest.raises(ValueError, match='Cannot return sensitivity matrix'):
        Rotation.align_vectors([[1, 2, 3]], [[1, 2, 3]], return_sensitivity=True)


def test_align_vectors_align_constrain():
    """Align the primary +X B axis with the primary +Y A axis, and rotate about
    it such that the +Y B axis (residual of the [1, 1, 0] secondary b vector)
    is aligned with the +Z A axis (residual of the [0, 1, 1] secondary a
    vector)"""

    atol = 1e-6
    b = [[1, 0, 0], [1, 1, 0]]
    a = [[0, 1, 0], [0, 1, 1]]
    m_expected = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).float()
    r, rssd = Rotation.align_vectors(a, b, weights=[torch.inf, 1])
    torch.testing.assert_close(r.as_matrix(), m_expected, atol=atol, rtol=0)
    torch.testing.assert_close(r(b), torch.tensor(a).float(), atol=atol, rtol=0)  # Pri and sec align exactly
    assert math.isclose(rssd, 0, abs_tol=atol)

    # Do the same but with an inexact secondary rotation
    b = [[1, 0, 0], [1, 2, 0]]
    rssd_expected = 1.0
    r, rssd = Rotation.align_vectors(a, b, weights=[torch.inf, 1])
    torch.testing.assert_close(r.as_matrix(), m_expected, atol=atol, rtol=0)
    torch.testing.assert_close(r(b)[0], torch.tensor(a[0]).float(), atol=atol, rtol=0)  # Only pri aligns exactly
    assert math.isclose(rssd, rssd_expected, abs_tol=atol)
    a_expected = torch.tensor([[0, 1, 0], [0, 1, 2]]).float()
    torch.testing.assert_close(r(b), a_expected, atol=atol, rtol=0)

    # Check random vectors
    b = [[1, 2, 3], [-2, 3, -1]]
    a = [[-1, 3, 2], [1, -1, 2]]
    rssd_expected = 1.3101595297515016
    r, rssd = Rotation.align_vectors(a, b, weights=[torch.inf, 1])
    torch.testing.assert_close(r(b)[0], torch.tensor(a[0]).float(), atol=atol, rtol=0)  # Only pri aligns exactly
    assert math.isclose(rssd, rssd_expected, abs_tol=atol)


def test_align_vectors_near_inf():
    """align_vectors should return near the same result for high weights as for
    infinite weights. rssd will be different with floating point error on the
    exactly aligned vector being multiplied by a large non-infinite weight
    """
    n = 100
    mats = []
    for i in range(6):
        mats.append(Rotation.random(n, random_state=10 + i).as_matrix())

    for i in range(n):
        # Get random pairs of 3-element vectors
        a = [1 * mats[0][i][0], 2 * mats[1][i][0]]
        b = [3 * mats[2][i][0], 4 * mats[3][i][0]]

        r, _ = Rotation.align_vectors(a, b, weights=[1e10, 1])
        r2, _ = Rotation.align_vectors(a, b, weights=[torch.inf, 1])
        torch.testing.assert_close(r.as_matrix(), r2.as_matrix(), atol=1e-4, rtol=0.0)

    for i in range(n):
        # Get random triplets of 3-element vectors
        a = [1 * mats[0][i][0], 2 * mats[1][i][0], 3 * mats[2][i][0]]
        b = [4 * mats[3][i][0], 5 * mats[4][i][0], 6 * mats[5][i][0]]

        r, _ = Rotation.align_vectors(a, b, weights=[1e10, 2, 1])
        r2, _ = Rotation.align_vectors(a, b, weights=[torch.inf, 2, 1])
        torch.testing.assert_close(r.as_matrix(), r2.as_matrix(), atol=1e-4, rtol=0.0)


def test_align_vectors_parallel():
    atol = 1e-6

    a = [[1, 0, 0], [0, 1, 0]]
    b = [[0, 1, 0], [0, 1, 0]]
    m_expected = torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]).float()
    r, _ = Rotation.align_vectors(a, b, weights=[torch.inf, 1])
    torch.testing.assert_close(r.as_matrix(), m_expected, atol=atol, rtol=0)

    r, _ = Rotation.align_vectors(a[0], b[0])
    torch.testing.assert_close(r.as_matrix(), m_expected, atol=atol, rtol=0)
    torch.testing.assert_close(r(b[0]), torch.tensor(a[0]).float(), atol=atol, rtol=0)

    b = [[1, 0, 0], [1, 0, 0]]
    m_expected = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).float()

    r, _ = Rotation.align_vectors(a, b, weights=[torch.inf, 1])
    torch.testing.assert_close(r.as_matrix(), m_expected, atol=atol, rtol=0)

    r, _ = Rotation.align_vectors(a[0], b[0])
    torch.testing.assert_close(r.as_matrix(), m_expected, atol=atol, rtol=0)
    torch.testing.assert_close(r(b[0]), torch.tensor(a[0]).float(), atol=atol, rtol=0)

    atol = 1e-6
    mats_a = Rotation.random(100, random_state=0).as_matrix()
    mats_b = Rotation.random(100, random_state=1).as_matrix()
    for mat_a, mat_b in zip(mats_a, mats_b, strict=False):
        # random 3-element unit vectors
        a = mat_a[0]
        b = mat_b[0]

        # Compare to align_vectors with primary only
        r, rssd = Rotation.align_vectors(a, b)
        torch.testing.assert_close(r(b), a, atol=atol, rtol=0)
        assert math.isclose(rssd, 0, abs_tol=atol)


def test_multiplication_stability():
    qs = Rotation.random(50, random_state=0)
    rs = Rotation.random(1000, random_state=1)
    for q in qs:
        rs @= q @ rs
        torch.testing.assert_close(torch.linalg.norm(rs.as_quat(), axis=1), torch.ones(1000))


@pytest.mark.parametrize('n', [-5, -2, -1, 0, 1, 2, 5])
def test_pow_integer(n):
    """Test Rotation**n"""
    # Test the short-cuts and other integers
    atol = 1e-6
    p = Rotation.random(10, random_state=0)
    p_inv = p.inv()

    # Test accuracy
    q = p**n
    r = Rotation.identity(10)
    for _ in range(abs(n)):
        r = r @ (p if n > 0 else p_inv)

    torch.testing.assert_close(r.as_quat(True), q.as_quat(True), atol=atol, rtol=0)

    # Test shape preservation
    r = Rotation.from_quat([0, 0, 0, 1])
    assert (r**n).as_quat().shape == (4,)
    r = Rotation.from_quat([[0, 0, 0, 1]])
    assert (r**n).as_quat().shape == (1, 4)


@pytest.mark.parametrize('n', [-1.5, -0.5, -0.0, 0.0, 0.5, 1.5])
def test_pow_fraction(n):
    """Large angle test cases for pow"""
    atol = 1e-7
    p = Rotation.random(10, random_state=0)
    q = p**n
    r = Rotation.from_rotvec(n * p.as_rotvec())
    torch.testing.assert_close(q.as_quat(), r.as_quat(), atol=atol, rtol=0)


def pow_small_angle():
    """Small angle test case for pow"""
    atol = 1e-7
    p = Rotation.from_rotvec([1e-6, 0, 0])
    n = 3
    q = p**n
    r = Rotation.from_rotvec(n * p.as_rotvec())
    torch.testing.assert_close(q.as_quat(), r.as_quat(), atol=atol, rtol=0)


def test_pow_modulus():
    """Modulus is not implemented"""
    p = Rotation.random(random_state=0)
    with pytest.raises(NotImplementedError, match='modulus not supported'):
        pow(p, 1, 1)


def test_rotation_within_numpy_object_array():
    """Rotation objects can be saved as objects in an numpy array"""

    single = Rotation.random(random_state=0)
    multiple = Rotation.random(2, random_state=1)

    array = np.array(single)
    assert array.shape == ()

    array = np.array(multiple)
    assert array.shape == (2,)
    torch.testing.assert_close(array[0].as_matrix(), multiple[0].as_matrix())
    torch.testing.assert_close(array[1].as_matrix(), multiple[1].as_matrix())

    array = np.array([single])
    assert array.shape == (1,)
    assert array[0] == single

    array = np.array([multiple])
    assert array.shape == (1, 2)
    torch.testing.assert_close(array[0, 0].as_matrix(), multiple[0].as_matrix())
    torch.testing.assert_close(array[0, 1].as_matrix(), multiple[1].as_matrix())

    array = np.array([single, multiple], dtype=object)
    assert array.shape == (2,)
    assert array[0] == single
    assert array[1] == multiple

    array = np.array([multiple, multiple, multiple])
    assert array.shape == (3, 2)


# Needed because of bug in torch 2.4.0. Should be fixed with 2.4.1
@pytest.mark.filterwarnings('ignore::FutureWarning')
def test_pickling():
    """Test pickling a Rotation"""
    r = Rotation.from_quat([0, 0, np.sin(torch.pi / 4), np.cos(torch.pi / 4)])
    pkl = pickle.dumps(r)
    unpickled = pickle.loads(pkl)  # noqa: S301
    torch.testing.assert_close(r.as_matrix(), unpickled.as_matrix())


def test_deepcopy():
    """Test copying a rotation"""
    r = Rotation.from_quat([0, 0, np.sin(torch.pi / 4), np.cos(torch.pi / 4)])
    r1 = copy.deepcopy(r)
    torch.testing.assert_close(r.as_matrix(), r1.as_matrix())


def test_as_euler_contiguous():
    """Test if euler angles return contiguous tensor"""
    r = Rotation.from_quat([0, 0, 0, 1])
    e1 = r.as_euler('xyz')  # extrinsic euler rotation
    e2 = r.as_euler('XYZ')  # intrinsic
    assert e1.is_contiguous()
    assert e2.is_contiguous()


def test_concatenate():
    """Test Rotation"""
    rotation = Rotation.random(10, random_state=0)
    sizes = [1, 2, 3, 1, 3]
    starts = [0, *np.cumsum(sizes)]
    split = [rotation[i : i + n] for i, n in zip(starts, sizes, strict=False)]
    result = Rotation.concatenate(split)
    assert (rotation.as_quat() == result.as_quat()).all()


def test_concatenate_wrong_type():
    """Test concatenation with non-Rotation objects"""
    with pytest.raises(TypeError, match='Rotation objects only'):
        Rotation.concatenate([Rotation.identity(), 1, None])  # type: ignore[list-item]


def test_len_and_bool():
    """Test __len__ and __bool__"""
    # Regression test for scipy gh-16663
    rotation_multi_empty = Rotation(torch.empty((0, 4)))
    rotation_multi_one = Rotation([[0, 0, 0, 1]])
    rotation_multi = Rotation([[0, 0, 0, 1], [0, 0, 0, 1]])
    rotation_single = Rotation([0, 0, 0, 1])

    assert len(rotation_multi_empty) == 0
    assert len(rotation_multi_one) == 1
    assert len(rotation_multi) == 2
    with pytest.raises(TypeError, match='Single rotation has no len().'):
        len(rotation_single)

    # Rotation should always be truthy. See scigh-16663
    assert rotation_multi_empty
    assert rotation_multi_one
    assert rotation_multi
    assert rotation_single


@pytest.mark.parametrize('theta', [0.0, np.pi / 8, np.pi / 4, np.pi / 3, np.pi / 2])
def test_mean(theta):
    "Basic test for mean"
    axes = np.concatenate((-np.eye(3), np.eye(3)))
    r = Rotation.from_rotvec(theta * axes)
    assert math.isclose(r.mean().magnitude(), 0.0)


@pytest.mark.parametrize('theta', [0.0, np.pi / 8, np.pi / 4, np.pi / 3, np.pi / 2])
def test_weighted_mean(theta):
    """Test that doubling a weight is equivalent to including a rotation twice."""
    axes = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
    rw = Rotation.from_rotvec(theta * axes[:2])
    mw = rw.mean(weights=[1, 2])
    r = Rotation.from_rotvec(theta * axes)
    m = r.mean()
    assert math.isclose((m @ mw.inv()).magnitude(), 0, abs_tol=1e-12)


@pytest.mark.parametrize(
    ('shape', 'keepdim', 'dim', 'expected_shape'),
    [
        ((3, 2, 4), True, 0, (1, 2, 4)),
        ((4, 3, 2), False, 0, (3, 2)),
        ((5, 4, 2), True, [0, -1], (1, 4, 1)),
        ((3, 1, 2), True, None, (1, 1, 1)),
        ((3, 4, 2), False, None, ()),
        ((3,), True, -1, (1,)),
        ((3,), False, -1, ()),
    ],
)
def test_weighted_mean_dims(shape, keepdim, dim, expected_shape):
    """Tests Rotation.mean for different combinations dim and shape.

    Checks the resulting shape and tests if multiplying a weight by N is
    equivalent to including a rotation Ntimes.
    """

    rnd = RandomGenerator(0)
    rotvectors1 = rnd.float32_tensor(size=(*shape[1:], 3))
    rotvectors2 = rnd.float32_tensor(size=(*shape[1:], 3))
    rotvectors = torch.stack([rotvectors1, *([rotvectors2] * (shape[0] - 1))], 0)
    weights = torch.ones(2, *[1 for _ in shape[1:]])
    weights[-1, ...] = shape[0] - 1

    # only include rotvectors2 one time, but weight it higher
    rotations_weight = Rotation.from_rotvec(rotvectors[:2])
    mean1 = rotations_weight.mean(weights=weights, keepdim=keepdim, dim=dim)
    # include rotvectors2 multiple times, but no weights
    rotations_full = Rotation.from_rotvec(rotvectors)
    mean2 = rotations_full.mean(weights=None, keepdim=keepdim, dim=dim)

    assert mean1.shape == expected_shape, 'Shape does not match'
    assert mean1.approx_equal(mean2).all(), 'Weighting a Rotation 2x is not the same as including it twice'


def test_mean_invalid_weights():
    """Test mean with invalid weights"""
    r = Rotation.from_quat(torch.eye(4))
    with pytest.raises(ValueError, match='non-negative'):
        r.mean(weights=-torch.ones(4))


def test_repr():
    """Test string representation"""
    assert repr(Rotation.identity(None)) == 'Rotation([[0.0, 0.0, 0.0, 1.0]])'
    assert repr(Rotation.identity(1)) == '(1,)-batched Rotation()'
    assert repr(Rotation.identity(1).reflect()) == '(1,)-batched improper Rotation()'


def test_quaternion_properties_single():
    """Test quaternion_x, quaternion_y, quaternion_z, quaternion_w"""
    quat = torch.tensor([1.0, 2.0, 3.0, 4.0])
    quat /= quat.norm()
    r = Rotation(quat, normalize=False)
    assert r.quaternion_x == quat[AXIS_ORDER.index('x')]
    assert r.quaternion_y == quat[AXIS_ORDER.index('y')]
    assert r.quaternion_z == quat[AXIS_ORDER.index('z')]
    assert r.quaternion_w == quat[-1]
    r.quaternion_x = 1.0  # type: ignore[assignment]
    r.quaternion_y = torch.tensor(2.0)
    r.quaternion_z = 3  # type: ignore[assignment]
    r.quaternion_w = 4.0  # type: ignore[assignment]
    torch.testing.assert_close(r.quaternion_x, torch.tensor(1.0))
    torch.testing.assert_close(r.quaternion_y, torch.tensor(2.0))
    torch.testing.assert_close(r.quaternion_z, torch.tensor(3.0))
    torch.testing.assert_close(r.quaternion_w, torch.tensor(4.0))


def test_quaternion_properties_batch():
    """Test quaternion_x, quaternion_y, quaternion_z, quaternion_w"""
    quat = torch.arange(10 * 2 * 4).reshape(10, 2, 4).float()
    quat /= quat.norm(dim=-1, keepdim=True)
    r = Rotation(quat, normalize=False)
    assert torch.equal(r.quaternion_x, quat[..., AXIS_ORDER.index('x')])
    assert torch.equal(r.quaternion_y, quat[..., AXIS_ORDER.index('y')])
    assert torch.equal(r.quaternion_z, quat[..., AXIS_ORDER.index('z')])
    assert torch.equal(r.quaternion_w, quat[..., -1])
    r.quaternion_x = 1.0 * torch.ones(10, 2)
    r.quaternion_y = 2.0 * torch.ones(10, 2)
    r.quaternion_z = 3.0 * torch.ones(10, 2)
    r.quaternion_w = 4.0 * torch.ones(10, 2)
    torch.testing.assert_close(r.quaternion_x, 1 * torch.ones(10, 2))
    torch.testing.assert_close(r.quaternion_y, 2 * torch.ones(10, 2))
    torch.testing.assert_close(r.quaternion_z, 3 * torch.ones(10, 2))
    torch.testing.assert_close(r.quaternion_w, 4 * torch.ones(10, 2))


def test_axis_order_zyx():
    """Check that the axis order is set to zyx"""
    assert AXIS_ORDER == 'zyx'


def test_from_to_directions():
    """Test that from_directions and as_directions are inverse operations"""
    one = torch.ones(1, 2, 3, 4)

    # must be a rotation
    b1 = SpatialDimension(one * (0.8146), one * (0.4707), one * (-0.3388))
    b2 = SpatialDimension(one * (-0.4432), one * (0.8820), one * (0.1599))
    b3 = SpatialDimension(one * (-0.3741), one * (-0.0199), one * (-0.9272))

    r = Rotation.from_directions(b1, b2, b3)
    torch.testing.assert_close(b1.zyx, r.as_directions()[0].zyx, atol=1e-4, rtol=0)
    torch.testing.assert_close(b2.zyx, r.as_directions()[1].zyx, atol=1e-4, rtol=0)
    torch.testing.assert_close(b3.zyx, r.as_directions()[2].zyx, atol=1e-4, rtol=0)


def test_as_directions():
    """Test conversion to basis vectors"""
    r = Rotation.random(10, random_state=0)
    matrix = r.as_matrix()
    directions = r.as_directions()
    for col, basis in enumerate(directions):
        for row, axis in enumerate(AXIS_ORDER):
            expected = matrix[:, row, col]
            actual = getattr(basis, axis)
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0)


def test_random_improper():
    """Test improper rotations"""
    r = Rotation.random(10, random_state=0, improper=True)
    matrix = r.as_matrix()
    det = torch.linalg.det(matrix)
    torch.testing.assert_close(det, -torch.ones(10))


def test_reflect():
    """Test improper rotations"""
    r = Rotation.random(None, random_state=0)
    r2 = r.reflect()
    r3 = r2.reflect()
    det = torch.linalg.det(r2.as_matrix())
    torch.testing.assert_close(det, torch.tensor(-1.0))
    torch.testing.assert_close(r.as_matrix(), r3.as_matrix())


def test_invert_axes():
    """Test inversion of axes"""
    r = Rotation.random(None, random_state=0)
    r2 = r.invert_axes()
    r3 = r2.invert_axes()
    det = torch.linalg.det(r2.as_matrix())
    torch.testing.assert_close(det, torch.tensor(-1.0))
    torch.testing.assert_close(r.as_matrix(), r3.as_matrix())
    torch.testing.assert_close(r.as_matrix(), -r2.as_matrix())


def test_improper_quat_inversion():
    """Test improper quaternions with inversion"""
    r = Rotation.random(10, random_state=0, improper='random')
    q, inv = r.as_quat(improper='inversion')
    assert torch.equal(r.is_improper, inv)
    r2 = Rotation.from_quat(q, inversion=inv)
    assert r2.approx_equal(r).all()


def test_improper_quat_reflection():
    """Test improper quaternions with reflection"""
    r = Rotation.random(10, random_state=0, improper='random')
    q, ref = r.as_quat(improper='reflection')
    assert torch.equal(r.is_improper, ref)
    r2 = Rotation.from_quat(q, reflection=ref)
    assert r2.approx_equal(r).all()


def test_improper_quat_warn():
    """Test improper quaternions with warning"""
    r = Rotation.random(10, random_state=0, improper=True)
    with pytest.warns(UserWarning, match='Rotation contains improper'):
        _ = r.as_quat(improper='warn')


def test_improper_euler_reflection():
    """Test improper euler angles with reflection"""
    r = Rotation.random(10, random_state=0, improper=True)
    angle, ref = r.as_euler('xyz', improper='reflection')
    r2 = Rotation.from_euler('xyz', angle, reflection=ref)
    assert r2.approx_equal(r, atol=1e-5).all()  # loss of precision in reflection conversion


def test_improper_euler_inversion():
    """Test improper euler angles with inversion"""
    r = Rotation.random(10, random_state=0, improper=True)
    angle, inv = r.as_euler('xyz', improper='inversion')
    r2 = Rotation.from_euler('xyz', angle, inversion=inv)
    assert r2.approx_equal(r).all()


def test_improper_euler_warn():
    """Test improper euler angles with warning"""
    r = Rotation.random(10, random_state=0, improper=True)
    with pytest.warns(UserWarning, match='Rotation contains improper'):
        _ = r.as_euler('xyz', improper='warn')


def test_improper_as_rotvec_reflection():
    """Test improper as_rotvec with reflection"""
    r = Rotation.random(10, random_state=0, improper=True)
    expected = r.reflect().as_rotvec()
    actual, _ = r.as_rotvec(improper='reflection')

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)


def test_improper_from_rotvec_reflection():
    """Test improper from_rotvec with reflection"""
    # Test the shortcut in from_rotvec
    r = Rotation.random(10, random_state=0, improper=False)
    rotvec = r.as_rotvec()
    actual = Rotation.from_rotvec(rotvec, reflection=True)
    expected = Rotation.from_rotvec(rotvec).reflect()
    assert actual.approx_equal(expected).all()


def test_improper_rotvec_inversion():
    """Test improper rotvec with inversion"""
    r = Rotation.random(10, random_state=0, improper=True)
    rotvec, inv = r.as_rotvec(improper='inversion')
    r2 = Rotation.from_rotvec(rotvec, inversion=inv)
    assert r2.approx_equal(r).all()


def test_improper_rotvec_reflection():
    """Test improper rotvec with inversion"""
    r = Rotation.random(1, random_state=0, improper=False)
    rotvec = r.as_rotvec()
    r2 = r.reflect()
    r3 = Rotation.from_rotvec(rotvec, reflection=True)
    assert r2.approx_equal(r3).all()


def test_improper_rotvec_warn():
    """Test improper rotvec with warning"""
    r = Rotation.random(10, random_state=0, improper=True)
    with pytest.warns(UserWarning, match='Rotation contains improper'):
        _ = r.as_rotvec(improper='warn')


def test_apply_scipy():
    """Test apply to vector (scipy style apply)"""
    r = Rotation.random(10, random_state=0)
    v = RandomGenerator(0).float32_tensor(size=(10, 3))
    with pytest.warns(UserWarning, match='Consider using Rotation'):
        actual = r.apply(v)
    expected = (r.as_matrix() @ v.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(expected, actual)


def test_apply_torch():
    """Test apply with callable (torch style apply)"""
    r = Rotation.random(10, random_state=0)
    r.apply(lambda x: x.double())
    assert r._quaternions.dtype == torch.float64


def test_random_vmf_uniform():
    """Test random rotations with a uniform distribution"""
    mean = torch.tensor([0, 0, 1.0])
    # vmf does not support a seed, as torch.distribution do not support it
    prev_rng_state = torch.random.get_rng_state()
    torch.manual_seed(0)
    r = Rotation.random_vmf(10000, mean, kappa=0, sigma=math.inf)
    torch.random.set_rng_state(prev_rng_state)
    assert r.shape == (10000,)
    assert r.mean().magnitude() < 0.1


def test_random_vmf_peaked():
    """Test random rotations with a peaked distribution"""
    mean = torch.tensor([0.0, 1.0, 0.0])
    # vmf does not support a seed, as torch.distribution do not support it
    prev_rng_state = torch.random.get_rng_state()
    torch.manual_seed(0)
    r = Rotation.random_vmf(5000, mean, kappa=50, sigma=20)
    torch.random.set_rng_state(prev_rng_state)
    assert r.shape == (5000,)
    torch.testing.assert_close(torch.linalg.cross(r.mean().as_rotvec(), mean), torch.zeros(3), atol=3e-3, rtol=0)


def test_apply_improper():
    """Test apply with improper rotations"""
    r = Rotation.random(10, random_state=0, improper=False)
    v = RandomGenerator(0).float32_tensor(size=(10, 3))
    actual = r.invert_axes()(v)
    expected = (-r.as_matrix() @ v.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(expected, actual)


def test_reshape():
    r = Rotation.random((1, 2, 3), random_state=0, improper=False)
    reshaped = r.reshape(3, 2, 1, 1)
    assert reshaped.shape == (3, 2, 1, 1)
    assert r.shape == (1, 2, 3)
    rereshaped = reshaped.reshape(1, 2, 3)
    torch.testing.assert_close(r._quaternions, rereshaped._quaternions)
