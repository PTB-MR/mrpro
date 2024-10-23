"""Helper/Utilities for test functions."""

from typing import Any

import torch
from mrpro.operators import LinearOperator, Operator


def relative_image_difference(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
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
    """Test the adjointness of linear operator and operator.H.

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


def linear_operator_isometry_test(
    linear_operator: LinearOperator, u: torch.Tensor, relative_tolerance: float = 1e-3, absolute_tolerance=1e-5
):
    """Test the isometry of a linear operator.

    Test if
         ||Operator(u)|| == ||u||
         for u ∈ domain of Operator.

    Parameters
    ----------
    linear_operator
        linear operator
    u
        element of the domain of the operator
    relative_tolerance
        default is pytorch's default for float16
    absolute_tolerance
        default is pytorch's default for float16

    Raises
    ------
    AssertionError
        if the adjointness property does not hold
    """
    torch.testing.assert_close(
        torch.norm(u), torch.norm(linear_operator(u)[0]), rtol=relative_tolerance, atol=absolute_tolerance
    )


def linear_operator_unitary_test(
    linear_operator: LinearOperator, u: torch.Tensor, relative_tolerance: float = 1e-3, absolute_tolerance=1e-5
):
    """Test if a linear operator is unitary.

    Test if
         Operator.adjoint(Operator(u)) == u
         for u ∈ domain of Operator.

    Parameters
    ----------
    linear_operator
        linear operator
    u
        element of the domain of the operator
    relative_tolerance
        default is pytorch's default for float16
    absolute_tolerance
        default is pytorch's default for float16

    Raises
    ------
    AssertionError
        if the adjointness property does not hold
    """
    torch.testing.assert_close(
        u, linear_operator.adjoint(linear_operator(u)[0])[0], rtol=relative_tolerance, atol=absolute_tolerance
    )


def autodiff_of_operator_test(
    operator: Operator[*tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]],
    *u: Any,
):
    """Test if autodiff of an operator is working.
    This test does not check that the gradient is correct but simply that it can be calculated using autodiff.
    torch.autograd.detect_anomaly will raise the Warning:
    Anomaly Detection has been enabled. This mode will increase the runtime and should only be enabled for debugging.
    If you want to add this function in a test, use the decorator:
    @pytest.mark.filterwarnings("ignore:Anomaly Detection has been enabled")
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
    with torch.autograd.detect_anomaly():
        v_range, _ = torch.func.jvp(operator.forward, u, u)

    # Backward-mode autodiff using vjp
    with torch.autograd.detect_anomaly():
        (_, vjpfunc) = torch.func.vjp(operator.forward, *u)
        vjpfunc(v_range)


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
