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
from tests import RandomGenerator
from tests.operators._OptimizationTestFunctions import Rosenbrock


@pytest.mark.parametrize(
    'optimizer_name, bounds_flag',
    [
        ('adam', 0),
        ('adam', 1),
        ('lbfgs', 0),
        ('lbfgs', 1),
    ],
)
def test_optimizers_rosenbrock(optimizer_name, bounds_flag):

    # TODO: currently required locally; check if required on GitHub;
    with pytest.raises(ImportWarning):

        random_generator = RandomGenerator(seed=0)

        # generate two-dimensional test data
        x1 = random_generator.float32_tensor(size=(1,))
        x2 = random_generator.float32_tensor(size=(1,))

        # enable gradient calculation
        x1.requires_grad = True
        x2.requires_grad = True
        params_init = [x1, x2]

        # define Rosenbrock function
        a, b = 1, 100
        rosen_brock = Rosenbrock(a, b)

        # set
        cop = ConstraintsOp(bounds=((-1, 1),)) if bounds_flag else None

        def f(x):  # TODO: Use @ later
            return rosen_brock(x) if cop is None else rosen_brock(cop(x))

        # hyperparams for optimizer
        lr = 1e-2
        max_iter = 250

        if optimizer_name == 'adam':
            optimizer = adam
        elif optimizer_name == 'lbfgs':
            optimizer = lbfgs

        # minimizer of Rosenbrock function
        analytical_sol = torch.tensor([a, a**2])

        # estimate minimizer
        params_result = optimizer(
            f,
            params_init,
            max_iter=max_iter,
            lr=lr,
        )

        torch.testing.assert_close(torch.tensor(params_result), analytical_sol)
