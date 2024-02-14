"""Tests for Adam."""

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

from mrpro.algorithms import adam
from mrpro.data import SpatialDimension
from mrpro.operators import ConstraintsOp
from mrpro.operators.functionals import L2_data_discrepancy
from tests import RandomGenerator
from tests.operators._OptimizationTestFunctions import Rosenbrock
from tests.phantoms._SaturationRecoveryEllipsePhantom import SaturationRecoveryEllipsePhantom2D


@pytest.mark.parametrize(
    'a, b',
    [(1.0, 100.0), (2.0, 50.0), (5.0, 10.0), (10.0, 5.0)],
)
def test_adam_rosenbrock(a, b):
    """Test adam functionality."""

    with pytest.raises(ImportWarning):  # ToDo: remove this when fixed
        # hyperparams or adam
        lr = 1e-2
        max_iter = 1000

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

        # call adam
        params = adam(
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
        tol = 1e-1
        mse = F.mse_loss(sol, x)
        assert mse < tol


@pytest.mark.parametrize(
    'ny, nx, bounds_flag',
    [
        (128, 128, 0),
        (128, 128, 1),
        (128, 256, 0),
        (128, 256, 1),
        (32, 64, 0),
        (32, 64, 1),
    ],
)
def test_adam_sat_recovery(ny, nx, bounds_flag):
    """Test lbfgs functionality for a saturation recovery example."""

    # image dimension
    im_shape = SpatialDimension(z=1, y=ny, x=nx)

    # create inversion times
    ti0, tiN, Nti = 0.1, 2.0, 36
    ti = torch.linspace(ti0, tiN, Nti)

    # create phantom object
    saturation_recovery_ellipse_phantom2d = SaturationRecoveryEllipsePhantom2D(im_shape, ti)

    # retrospetively generate data
    noise_std = 0.01
    m0_true, t1_true, data = saturation_recovery_ellipse_phantom2d.generate_data(noise_std)

    # bounds for variables
    if bounds_flag:
        # make sure that the interval defined by bounds contains the t1-maps point-wise
        bounds_t1 = ((1 - 0.05) * t1_true.min().item(), (1 + 0.05) * t1_true.max().item())  # in s
        bounds = ((None, None), bounds_t1)

        # hyper parameters for the variable transform operator
        beta_sigmoid = 0.3
        beta_softplus = 1.0
        cop = ConstraintsOp(bounds, beta_sigmoid=beta_sigmoid, beta_softplus=beta_softplus)
    else:
        cop = None

    # initializations for m0 and t1
    m0_init = data[[0], ...].clone()
    m0_init.requires_grad = True
    t1_init = (
        cop.sigmoid_transf_inv(torch.ones(t1_true.shape, dtype=torch.float32), bounds=bounds_t1, beta=beta_sigmoid)
        if cop is not None
        else torch.ones(t1_true.shape)
    )
    t1_init.requires_grad = True

    # list of parameters
    params = [m0_init, t1_init]

    # functional to be minimized
    L2 = L2_data_discrepancy(data)

    # define functional to be minimized
    q = saturation_recovery_ellipse_phantom2d.get_sat_recovery_model()

    def f(m0, t1):  # TODO: Use @ later
        return L2(q(m0, t1)) if cop is None else L2(q(*cop((m0, t1))))

    # hyperparams for adam
    lr = 1e-1
    max_iter = 3000

    # call adam
    params = adam(f, params=params, max_iter=max_iter, lr=lr)

    if cop is not None:
        # transform back the variables
        params = cop(tuple(params))

    p_adam = torch.cat(params, dim=0).detach()
    p_true = torch.cat([m0_true, t1_true], dim=0).detach()

    # mask for where there is no signal
    mask = torch.heaviside(m0_true.abs(), torch.tensor(0.0))

    mse = F.mse_loss(torch.view_as_real(mask * p_true), torch.view_as_real(mask * p_adam))
    assert mse < 0.025
