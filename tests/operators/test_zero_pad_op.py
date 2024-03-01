"""Tests for Zero Pad Operator class."""

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
from mrpro.operators import ZeroPadOp

from tests import RandomGenerator


def test_zero_pad_op_content():
    """Test correct padding."""
    dshape_orig = (2, 100, 3, 200, 50, 2)
    dshape_new = (2, 80, 3, 100, 240, 2)
    generator = RandomGenerator(seed=0)
    dorig = generator.complex64_tensor(dshape_orig)
    pad_dim = (-5, -3, -2)
    POp = ZeroPadOp(
        dim=pad_dim,
        orig_shape=tuple([dshape_orig[d] for d in pad_dim]),
        padded_shape=tuple([dshape_new[d] for d in pad_dim]),
    )
    (dnew,) = POp.forward(dorig)

    # Compare overlapping region
    torch.testing.assert_close(dorig[:, 10:90, :, 50:150, :, :], dnew[:, :, :, :, 95:145, :])


@pytest.mark.parametrize(
    'u_shape, v_shape',
    [
        ((101, 201, 50), (13, 221, 64)),
        ((100, 200, 50), (14, 220, 64)),
        ((101, 201, 50), (14, 220, 64)),
        ((100, 200, 50), (13, 221, 64)),
    ],
)
def test_zero_pad_op_adjoint(u_shape, v_shape):
    """Test adjointness of pad operator."""
    generator = RandomGenerator(seed=0)
    u = generator.complex64_tensor(u_shape)
    v = generator.complex64_tensor(v_shape)
    POp = ZeroPadOp(dim=(-3, -2, -1), orig_shape=u_shape, padded_shape=v_shape)
    (Au,) = POp.forward(u)
    (AHv,) = POp.adjoint(v)

    assert torch.isclose(torch.vdot(Au.flatten(), v.flatten()), torch.vdot(u.flatten(), AHv.flatten()), rtol=1e-3)
