"""Tests for non-linear optimization algorithms."""

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

import pytest
import torch
from mrpro.algorithms.optimizers import adam
from mrpro.algorithms.optimizers import lbfgs
from mrpro.operators import ConstraintsOp
from tests.operators._OptimizationTestFunctions import Rosenbrock


@pytest.mark.parametrize('enforce_bounds_on_x1', [True, False])
@pytest.mark.parametrize(
    ('optimizer', 'optimizer_kwargs'), [(adam, {'lr': 0.02, 'max_iter': 10000}), (lbfgs, {'lr': 1.0})]
)
#@pytest.mark.filterwarnings('ignore:allow_ops_in_compiled_graph')
def test_optimizers_rosenbrock(optimizer, enforce_bounds_on_x1, optimizer_kwargs):
    # use Rosenbrock function as test case with 2D test data
    a, b = 1.0, 100.0
    rosen_brock = Rosenbrock(a, b)

    # initial point of optimization
    x1 = torch.tensor([a / 3.14])
    x2 = torch.tensor([3.14])
    x1.grad = torch.tensor([2.78])
    x2.grad = torch.tensor([-1.0])
    params_init = [x1, x2]

    # save to compare with later as optimization should not change the initial points
    params_init_before = [i.detach().clone() for i in params_init]
    params_init_grad_before = [i.grad.detach().clone() if i.grad is not None else None for i in params_init]

    if enforce_bounds_on_x1:
        # the analytical solution for x_1 will be a, thus we can limit it into [0,2a]
        constrain_op = ConstraintsOp(bounds=((0, 2 * a),))
        functional = rosen_brock @ constrain_op
    else:
        functional = rosen_brock

    # minimizer of Rosenbrock function
    analytical_solution = torch.tensor([a, a**2])

    params_result = optimizer(functional, params_init, **optimizer_kwargs)

    if enforce_bounds_on_x1:
        # the parameters are currently the unbounded values, by applying the operator again
        # we obtain the bounded true values
        params_result = constrain_op(*params_result)

    # obtained solution should match analytical
    torch.testing.assert_close(torch.tensor(params_result), analytical_solution)

    for p, before, grad_before in zip(params_init, params_init_before, params_init_grad_before, strict=True):
        assert p == before, 'the initial parameter should not have changed during optimization'
        assert p.grad == grad_before, 'the initial parameters gradient should not have changed during optimization'
