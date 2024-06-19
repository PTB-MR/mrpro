"""Helper/Utilities for test functions."""

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

from typing import Any
from typing import TypeVarTuple

import torch
from mrpro.operators.LinearOperator import LinearOperator
from mrpro.operators.Operator import Operator

Tin = TypeVarTuple('Tin')


def relative_image_difference(img1: torch.Tensor, img2: torch.Tensor):
    """Calculate mean absolute relative difference between two images.

    Parameters
    ----------
    img1
        first image
    img2
        second image

    Returns
    -------
        mean absolute relative difference between images
    """
    image_difference = torch.mean(torch.abs(img1 - img2))
    image_mean = 0.5 * torch.mean(torch.abs(img1) + torch.abs(img2))
    if image_mean == 0:
        raise ValueError('average of images should be larger than 0')
    return image_difference / image_mean


def dotproduct_adjointness_test(
    linear_operator: LinearOperator,
    u: torch.Tensor,
    v: torch.Tensor,
    relative_tolerance: float = 1e-3,
    absolute_tolerance=1e-5,
):
    """Test the adjointness of operator and operator.H

    Test if
         <Operator(u),v> == <u, Operator^H(v)>
         for one u ∈ domain and one v ∈ range of Operator.
    and if the shapes match.

    Note: This property should hold for all u and v.
    Commonly, this function is called with two random vectors u and v.


    Parameters
    ----------
    linear_operator
        linear operator
    u
        element of the domain of the operator
    v
        element of the range of the operator
    relative_tolerance
        default is pytorch's default for float16
    absolute_tolerance
        default is pytorch's default for float16

    Raises
    ------
    AssertionError
        if the adjointness property does not hold
    AssertionError
        if the shape of linear_operator(u) and v does not match
        if the shape of u and linear_operator.H(v) does not match

    """
    (forward_u,) = linear_operator(u)
    (adjoint_v,) = linear_operator.adjoint(v)

    # explicitly check the shapes, as flatten makes the dot product insensitive to wrong shapes
    assert forward_u.shape == v.shape
    assert adjoint_v.shape == u.shape

    dotproduct_range = torch.vdot(forward_u.flatten(), v.flatten())
    dotproduct_domain = torch.vdot(u.flatten().flatten(), adjoint_v.flatten())
    torch.testing.assert_close(dotproduct_range, dotproduct_domain, rtol=relative_tolerance, atol=absolute_tolerance)


def gradient_of_linear_operator_test(
    linear_operator: LinearOperator,
    u: torch.Tensor,
    v: torch.Tensor,
    relative_tolerance: float = 1e-3,
    absolute_tolerance=1e-5,
):
    """Test the gradient of a linear operator is the adjoint.

    Note: This property should hold for all u and v.
    Commonly, this function is called with two random vectors u and v.


    Parameters
    ----------
    linear_operator
        linear operator
    u
        element of the domain of the operator
    v
        element of the range of the operator
    relative_tolerance
        default is pytorch's default for float16
    absolute_tolerance
        default is pytorch's default for float16

    Raises
    ------
    AssertionError
        if the gradient is not the adjoint


    """
    # Gradient of the forward via vjp
    (_, vjpfunc) = torch.func.vjp(linear_operator.forward, u)
    assert torch.allclose(
        vjpfunc((v,))[0], linear_operator.adjoint(v)[0], rtol=relative_tolerance, atol=absolute_tolerance
    )

    # Gradient of the adjoint via vjp
    (_, vjpfunc) = torch.func.vjp(linear_operator.adjoint, v)
    assert torch.allclose(
        vjpfunc((u,))[0], linear_operator.forward(u)[0], rtol=relative_tolerance, atol=absolute_tolerance
    )


def forward_mode_autodiff_of_linear_operator_test(
    linear_operator: LinearOperator,
    u: torch.Tensor,
    v: torch.Tensor,
    relative_tolerance: float = 1e-3,
    absolute_tolerance=1e-5,
):
    """Test the forward-mode autodiff calculation.

    Verifies that the Jacobian-vector product (jvp) is equivalent to applying the operator.

    Note: This property should hold for all u and v.
    Commonly, this function is called with two random vectors u and v.


    Parameters
    ----------
    linear_operator
        linear operator
    u
        element of the domain of the operator
    v
        element of the range of the operator
    relative_tolerance
        default is pytorch's default for float16
    absolute_tolerance
        default is pytorch's default for float16

    Raises
    ------
    AssertionError
        if the jvp yields different results than applying the operator


    """
    # jvp of the forward
    assert torch.allclose(
        torch.func.jvp(linear_operator.forward, (u,), (u,))[0][0],
        linear_operator.forward(u)[0],
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )

    # jvp of the adjoint
    assert torch.allclose(
        torch.func.jvp(linear_operator.adjoint, (v,), (v,))[0][0],
        linear_operator.adjoint(v)[0],
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )


def autodiff_of_operator_test(
    operator: Operator[*Tin, tuple[torch.Tensor,]],
    *u: Any,
):
    """Test autodiff an operator is working.

    This test does not check that the gradient is correct but simply that it can be calculated using autodiff.

    Parameters
    ----------
    operator
        operator
    u
        element(s) of the domain of the operator

    Raises
    ------
    AssertionError
        if autodiff fails


    """
    # Forward-mode autodiff using jvp
    assert torch.func.jvp(operator.forward, u, u)

    # Backward-mode autodiff using vjp
    assert torch.func.vjp(operator.forward, *u)
