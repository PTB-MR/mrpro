"""Test for the adjoint as backward option in the LinearOperator class."""

import pytest
import torch
from mrpro.operators import LinearOperator
from torch.autograd.gradcheck import GradcheckError

from tests import RandomGenerator
from tests.helper import dotproduct_adjointness_test


class NonDiffentiableOperator(LinearOperator, adjoint_as_backward=False):
    """Dummy operator for testing purposes."""

    def forward(self, x):
        python_float = x.item()
        tensor = torch.tensor([2 * python_float], dtype=x.dtype)  # breaks the autograd
        return (tensor,)

    def adjoint(self, x):
        python_float = x.item()
        tensor = torch.tensor([2 * python_float], dtype=x.dtype)  # breaks the autograd
        return (tensor,)


class DiffentiableOperator(LinearOperator, adjoint_as_backward=True):
    """Dummy operator for testing purposes. Should be differentiable due to adjoint_as_backward=True."""

    def forward(self, x):
        python_float = x.item()
        tensor = torch.tensor([2 * python_float], dtype=x.dtype)  # breaks the autograd
        return (tensor,)

    def adjoint(self, x):
        python_float = x.item()
        tensor = torch.tensor([2 * python_float], dtype=x.dtype)  # breaks the autograd
        return (tensor,)


def test_linop_autograd_dummy_operator():
    """Ensure correct dummy operator."""
    rng = RandomGenerator(seed=0)
    nondiff = NonDiffentiableOperator()
    diff = DiffentiableOperator()

    u = rng.float32_tensor((1,))
    v = rng.float32_tensor((1,))

    # Differentiable operator and non differentiable operator should be the same
    torch.testing.assert_close(nondiff.adjoint(u), nondiff.adjoint(u))
    torch.testing.assert_close(diff.adjoint(u), diff.adjoint(u))

    # and the adjoin should be correct
    dotproduct_adjointness_test(nondiff, u, v)


def test_linop_autograd_nondifferentiable():
    """Test the forward of the non differentiable operator."""
    nondiff = NonDiffentiableOperator()
    # The non differentiable operator should raise an error,
    # because the forward is non differentiable (python float)
    x = torch.tensor(1.0, requires_grad=True, dtype=torch.float64)
    with pytest.raises(GradcheckError):
        torch.autograd.gradcheck(nondiff, x)


def test_linop_autograd_differentiable_forward():
    """Test the backward of the forward of the differentiable operator."""
    diff = DiffentiableOperator()
    # The differentiable operator should work, as
    # the adjoint can be used as backward.
    x = torch.tensor(1.0, requires_grad=True, dtype=torch.float64)
    torch.autograd.gradcheck(diff, x)


def test_linop_autograd_differentiable_adjoint():
    """Test the backward of the adjoint of the differentiable operator."""
    diff = DiffentiableOperator()
    # Should work, as the forward will be used as the backward
    # of the adjoint.
    x = torch.tensor(1.0, requires_grad=True, dtype=torch.float64)
    torch.autograd.gradcheck(diff.H, x)


def test_linop_autograd_differentiable_gradgrad():
    """Test the gradgrad of the differentiable operator."""
    diff = DiffentiableOperator()
    # The backward of the backward should be the working
    # as it should be using the forward
    x = torch.tensor(1.0, requires_grad=True, dtype=torch.float64)
    torch.autograd.gradgradcheck(diff, x)
