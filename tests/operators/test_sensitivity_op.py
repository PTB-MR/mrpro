"""Tests for Sensitivity operator."""

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

import pytest
import torch
from mrpro.data import CsmData
from mrpro.data import QHeader
from mrpro.data import SpatialDimension
from mrpro.operators import SensitivityOp

from tests import RandomGenerator


def test_sensitivity_op_adjointness():
    """Sensitivity Operator adjoint property."""

    random_generator = RandomGenerator(seed=0)
    n_z, n_y, n_x = 100, 100, 100
    n_coils = 4

    # Generate sensitivity operator
    random_tensor = random_generator.complex64_tensor(size=(1, n_coils, n_z, n_y, n_x))
    random_csm = CsmData(data=random_tensor, header=QHeader(fov=SpatialDimension(1.0, 1.0, 1.0)))
    sensitivity_op = SensitivityOp(random_csm)

    # Check adjoint property
    u = random_generator.complex64_tensor(size=(1, 1, n_z, n_y, n_x))
    v = random_generator.complex64_tensor(size=(1, n_coils, n_z, n_y, n_x))
    (forward,) = sensitivity_op.forward(u)
    (adjoint,) = sensitivity_op.adjoint(v)
    torch.testing.assert_close(
        torch.vdot(forward.ravel(), v.ravel()),
        torch.vdot(u.ravel(), adjoint.ravel()),
        rtol=1e-03,
        atol=1e-03,
    )


@pytest.mark.parametrize(('csm_other_dim', 'img_other_dim'), [(1, 1), (1, 6), (6, 6)])
def test_sensitivity_op_other_dim_compatibility_pass(csm_other_dim, img_other_dim):
    """Test paired-dimensions that have to pass applying the sensitivity
    operator."""

    random_generator = RandomGenerator(seed=0)
    n_z, n_y, n_x = 100, 100, 100
    n_coils = 4

    # Generate sensitivity operator
    random_tensor = random_generator.complex64_tensor(size=(csm_other_dim, n_coils, n_z, n_y, n_x))
    random_csm = CsmData(data=random_tensor, header=QHeader(fov=SpatialDimension(1.0, 1.0, 1.0)))
    sensitivity_op = SensitivityOp(random_csm)

    u = random_generator.complex64_tensor(size=(img_other_dim, 1, n_z, n_y, n_x))
    v = random_generator.complex64_tensor(size=(img_other_dim, n_coils, n_z, n_y, n_x))
    (forward,) = sensitivity_op.forward(u)
    (adjoint,) = sensitivity_op.adjoint(v)
    assert forward.shape == (img_other_dim, n_coils, n_z, n_y, n_x)
    assert adjoint.shape == (img_other_dim, 1, n_z, n_y, n_x)


@pytest.mark.parametrize(('csm_other_dim', 'img_other_dim'), [(6, 3), (3, 6)])
def test_sensitivity_op_other_dim_compatibility_fail(csm_other_dim, img_other_dim):
    """Test paired-dimensions that have to raise error for the sensitivity
    operator."""
    random_generator = RandomGenerator(seed=0)
    n_z, n_y, n_x = 100, 100, 100
    n_coils = 4

    # Generate sensitivity operator
    random_tensor = random_generator.complex64_tensor(size=(csm_other_dim, n_coils, n_z, n_y, n_x))
    random_csm = CsmData(data=random_tensor, header=QHeader(fov=SpatialDimension(1.0, 1.0, 1.0)))
    sensitivity_op = SensitivityOp(random_csm)

    u = random_generator.complex64_tensor(size=(img_other_dim, 1, n_z, n_y, n_x))
    with pytest.raises(RuntimeError, match='The size of tensor'):
        sensitivity_op.forward(u)

    v = random_generator.complex64_tensor(size=(img_other_dim, n_coils, n_z, n_y, n_x))
    with pytest.raises(RuntimeError, match='The size of tensor'):
        sensitivity_op.adjoint(v)
