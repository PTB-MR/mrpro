"""Tests for LBFGS."""

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
import torch.nn.functional as F

from mrpro.algorithms import lbfgs
from mrpro.data import SpatialDimension
from mrpro.operators import ConstraintsOp
from mrpro.operators import Operator
from mrpro.operators.functionals import L2_data_discrepancy
from mrpro.operators.models import SaturationRecovery
from mrpro.phantoms import EllipsePhantom
from tests import RandomGenerator
from tests.operators._OptimizationTestFunctions import Rosenbrock


@pytest.mark.parametrize(
    'a, b',
    [(1.0, 100.0), (1.0, 50.0), (1.0, 25.0), (1.0, 2.0)],
)
def test_lbfgs_rosenbrock(a, b):
    """Test lbfgs functionality."""

    # with pytest.raises(ImportWarning):  # ToDo: remove this when fixed
    # hyperparams or lbfgs
    lr = 1.0
    max_iter = 2000

    random_generator = RandomGenerator(seed=0)

    # generate two-dimensional test data
    x1 = random_generator.float32_tensor(size=(1,))
    x2 = random_generator.float32_tensor(size=(1,))

    # enable gradient calculation
    x1.requires_grad = True
    x2.requires_grad = True

    params = [x1, x2]

    # define Rosenbrock function
    f = Rosenbrock(a, b)

    # call lbfgs
    params = lbfgs(
        f,
        params,
        max_iter=max_iter,
        lr=lr,
    )

    # minimizer of Rosenbrock function
    sol = torch.tensor([a, a**2])

    # obtained solution
    x = torch.tensor(params)

    # check if they are close
    tol = 1.0
    mse = F.mse_loss(sol, x)
    assert mse < tol
