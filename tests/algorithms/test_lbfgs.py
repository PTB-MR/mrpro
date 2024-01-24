"""Tests lbfgs."""

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
import torch.nn as nn

from mrpro.algorithms import lbfgs
from tests import RandomGenerator


class Rosenbrock(nn.Module):
    def __init__(self, a=1, b=100):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x1, x2):
        fval = (self.a - x1) ** 2 + self.b * (x[:, 1] - x2**2) ** 2

        return fval


@pytest.mark.parametrize(
    'a, b',
    [
        (1, 100),
    ],
)
def test_lbfgs(a, b):
    """Test lbfgs functionality."""

    with pytest.raises(ImportWarning):  # ToDo: remove this when fixed
        # hyperparams or lbfgs
        lr = 1
        max_iter = 120
        max_eval = 120
        tolerance_grad = 1e-07
        tolerance_change = 1e-09
        history_size = 10
        line_search_fn = 'strong_wolfe'

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
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
        )

        # minimizer of Rosenbrock function
        sol = torch.tensor([a, a**2])
        x = torch.tensor(params)
        assert torch.isclose(x, sol, rtol=1e-4)
