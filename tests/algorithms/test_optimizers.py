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
from mrpro.algorithms import adam
from mrpro.algorithms import lbfgs
from mrpro.operators import ConstraintsOp
from tests.operators._OptimizationTestFunctions import Rosenbrock


@pytest.mark.parametrize('enforce_bounds_on_x1', [True, False])
@pytest.mark.parametrize(
    'optimizer',
    [
        adam,
        lbfgs,
    ],
)
def test_optimizers_rosenbrock(optimizer, enforce_bounds_on_x1):
    # TODO: remove once fixed in pytorch. see also issue #132 on GitHub
    with pytest.raises(ImportWarning):
        # use Rosenbrock function as test case with 2D test data
        a, b = 1, 100
        rosen_brock = Rosenbrock(a, b)

        # initial point of optimization
        x1 = torch.tensor([42.0])
        x2 = torch.tensor([3.14])
        x1.grad = torch.tensor([2.7])
        x2.grad = torch.tensor([-1.0])
        params_init = [x1, x2]

        # save to compare with later as optimization should not change the initial points
        params_init_before = [i.detach().clone for i in params_init]

        if enforce_bounds_on_x1:
            # the analytical solution for x_1 will be a, thus we can limit it into [0,2a]
            constrain_op = ConstraintsOp(bounds=((0, 2 * a),))
            functional = rosen_brock @ constrain_op
        else:
            functional = rosen_brock

        # hyperparams for optimizer
        lr = 1e-2
        max_iter = 250

        # minimizer of Rosenbrock function
        analytical_solution = torch.tensor([a, a**2])

        # estimate minimizer
        params_result = optimizer(
            functional,
            params_init,
            max_iter=max_iter,
            lr=lr,
        )

        # obtained solution should match analytical
        torch.testing.assert_close(torch.tensor(params_result), analytical_solution)

        for p, before in zip(params_init, params_init_before, strict=True):
            assert p == before, 'the initial parameter should not have changed during optimization'
            assert p.grad == before, 'the inital paramters gradient should not have changed during optimization'
