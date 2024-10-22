"""Tests for non-linear optimization algorithms."""

import pytest
import torch
from mrpro.algorithms.optimizers import OptimizerStatus, adam, lbfgs
from mrpro.operators import ConstraintsOp
from tests.operators import Rosenbrock


@pytest.mark.parametrize('enforce_bounds_on_x1', [True, False])
@pytest.mark.parametrize(
    ('optimizer', 'optimizer_kwargs'), [(adam, {'lr': 0.02, 'max_iter': 10000}), (lbfgs, {'lr': 1.0})]
)
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


@pytest.mark.parametrize('optimizer', [adam, lbfgs])
def test_callback_optimizers(optimizer):
    """Test if a callback function is called within the optimizers."""

    # use Rosenbrock function as test case with 2D test data
    a, b = 1.0, 100.0
    rosen_brock = Rosenbrock(a, b)

    # initial point of optimization
    parameter1 = torch.tensor([a / 3.14], requires_grad=True)
    parameter2 = torch.tensor([3.14], requires_grad=True)
    parameters = [parameter1, parameter2]

    # maximum number of iterations
    max_iter = 10

    # callback function; if the function is called during the iterations, the
    # test is successful
    def callback(optimizer_status: OptimizerStatus) -> None:
        _, _ = optimizer_status['iteration_number'], optimizer_status['solution'][0]
        assert True

    # run optimizer
    _ = optimizer(rosen_brock, parameters, max_iter=max_iter, callback=callback)
