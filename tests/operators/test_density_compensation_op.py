"""Tests for density compensation operator."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
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

import torch
from mrpro.data import DcfData
from mrpro.operators import DensityCompensationOp

from tests import RandomGenerator
from tests.helper import dotproduct_adjointness_test


def test_dcf_op_adjointness():
    """Test density operator adjoint property."""
    random_generator = RandomGenerator(seed=0)

    ns_zyx = (2, 3, 4)
    ns_other = (5, 6, 7)
    n_coils = 8

    # Generate sensitivity operator
    random_tensor = random_generator.complex64_tensor(size=(*ns_other, *ns_zyx))
    random_dcf = DcfData(data=random_tensor)
    dcf_op = DensityCompensationOp(random_dcf)

    # Check adjoint property
    u = random_generator.complex64_tensor(size=(*ns_other, n_coils, *ns_zyx))
    v = random_generator.complex64_tensor(size=(*ns_other, n_coils, *ns_zyx))
    dotproduct_adjointness_test(dcf_op, u, v)


def test_sensitivity_op_dcfdata_tensor():
    """Test matching result after creation via tensor and DcfData."""
    random_generator = RandomGenerator(seed=0)

    ns_zyx = (2, 3, 4)
    ns_other = (5, 6, 7)
    n_coils = 8

    # Generate sensitivity operators
    random_tensor = random_generator.complex64_tensor(size=(*ns_other, *ns_zyx))
    random_dcf = DcfData(data=random_tensor)

    dcf_op_tensor = DensityCompensationOp(random_tensor)
    dcf_op_dcfdata = DensityCompensationOp(random_dcf)
    dcf_op_dcfdata_asop = random_dcf.as_operator()

    # Check equality
    u = random_generator.complex64_tensor(size=(*ns_other, n_coils, *ns_zyx))
    v = random_generator.complex64_tensor(size=(*ns_other, n_coils, *ns_zyx))
    assert torch.equal(*dcf_op_tensor(u), *dcf_op_dcfdata(u))
    assert torch.equal(*dcf_op_tensor.H(v), *dcf_op_dcfdata.H(v))
    assert torch.equal(*dcf_op_tensor(u), *dcf_op_dcfdata_asop(u))
    assert torch.equal(*dcf_op_tensor.H(v), *dcf_op_dcfdata_asop.H(v))


def test_sensitivity_forward():
    """Test result of forward."""
    random_generator = RandomGenerator(seed=0)
    ns_zyx = (2, 3, 4)
    ns_other = (5, 6, 7)
    n_coils = 8
    random_tensor = random_generator.complex64_tensor(size=(*ns_other, *ns_zyx))
    dcf_op = DensityCompensationOp(random_tensor)
    u = random_generator.complex64_tensor(size=(*ns_other, n_coils, *ns_zyx))
    # forward should be a multiplication with the dcf
    excepted = random_tensor.unsqueeze(-4) * u
    torch.testing.assert_close(dcf_op(u), excepted)
