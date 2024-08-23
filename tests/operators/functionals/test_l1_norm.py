"""Tests for L1-functional."""

import pytest
import torch
from mrpro.operators.functionals.l1 import L1Norm, L1NormViewAsReal


@pytest.mark.parametrize(
    (
        'x',
        'dim',
        'keepdim',
        'shape_forward_x',
    ),
    [
        (
            torch.rand(16,8,1,160,160),
            (-2,-1),
            True,
            torch.Size([16,8,1,1,1]),
        ),
        (
            torch.rand(16,8,1,160,160),
            (-2,-1),
            False,
            torch.Size([16,8,1]),
        ),
        (
            torch.rand(16,8,1,160,160),
            None,
            False,
            torch.Size([]),
        ),
        (
            torch.rand(16,8,1,160,160),
            None,
            True,
            torch.Size([1,1,1,1,1]),
        ),
    ],
)
def test_l1_functional_shape(
    x,
    dim,
    keepdim,
    shape_forward_x,
):
    """Test if L1 norm matches expected values."""
    l1_norm = L1Norm(weight=1)
    torch.testing.assert_close(l1_norm.forward(x,dim=dim,keep_dim=keepdim)[0].shape,shape_forward_x , rtol=1e-3, atol=1e-3)
    l1_norm = L1Norm(weight=1, divide_by_n=True)
    torch.testing.assert_close(l1_norm.forward(x,dim=dim,keep_dim=keepdim)[0].shape,shape_forward_x , rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    (
        'x',
    ),
    [
        (
            torch.rand(10,10,10),
        ),
    ],
)
def test_prox_equals_unity_matrix(
    x,
):
    """Test if Prox of l1 norm is the identity if sigma is 0 or close to 0."""
    l1_norm = L1Norm(weight=1, divide_by_n=True)
    torch.testing.assert_close(l1_norm.prox(x, sigma=0)[0], x, rtol=1e-3, atol=1e-3)
    l1_norm = L1Norm(weight=1, divide_by_n=False)
    torch.testing.assert_close(l1_norm.prox(x, sigma=0)[0], x, rtol=1e-3, atol=1e-3)
    
    
@pytest.mark.parametrize(
    (
        'x',
    ),
    [
        (
            torch.rand(10,10,10),
        ),
    ],
)
def test_W_equals_0(
    x,
):
    """Test if Prox of l1 norm is the identity if W=0."""
    l1_norm = L1Norm(weight=0, divide_by_n=True)
    torch.testing.assert_close(l1_norm.prox(x, sigma=1)[0], x, rtol=1e-3, atol=1e-3)
    l1_norm = L1Norm(weight=0, divide_by_n=False)
    torch.testing.assert_close(l1_norm.prox(x, sigma=1)[0], x, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    (
        'x',
        'forward_x_sum',
        'forward_x_mean',
        'prox',
        'prox_complex_conjugate',
    ),
    [
        (
            torch.tensor([2.0, -2.0]),
            torch.tensor((4), dtype=torch.float32),
            torch.tensor((2), dtype=torch.float32),
            torch.tensor([1.5, -1.5], dtype=torch.float32),
            torch.tensor([1, -1], dtype=torch.float32),
        ),
    ],
)
def test_l1_known_cases_2(
    x,
    forward_x_sum,
    forward_x_mean,
    prox,
    prox_complex_conjugate,
):
    """Test if L1 norm matches expected values."""
    l1_norm = L1Norm(weight=1, keep_dim=False)
    torch.testing.assert_close(l1_norm.forward(x, divide_by_n=False)[0], forward_x_sum, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(l1_norm.forward(x, divide_by_n=True)[0], forward_x_mean, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(l1_norm.prox(x, sigma=0.5)[0], prox, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(l1_norm.prox_convex_conj(x, sigma=0.5)[0], prox_complex_conjugate, rtol=1e-3, atol=1e-3)
    
    
@pytest.mark.parametrize(
    (
        'x',
        'forward_x_sum',
        'forward_x_mean',
        'prox',
        'prox_complex_conjugate',
    ),
    [
        (
            torch.tensor([2.0, -2.0]),
            torch.tensor((2), dtype=torch.float32),
            torch.tensor((1), dtype=torch.float32),
            torch.tensor([1.5, -1.5], dtype=torch.float32),
            torch.tensor([1, -1], dtype=torch.float32),
        ),
    ],
)
def test_l1_known_cases(
    x,
    forward_x_sum,
    forward_x_mean,
    prox,
    prox_complex_conjugate,
):
    """Test if L1 norm matches expected values."""
    l1_norm = L1Norm(weight=1, target=torch.tensor([1,-1]), keep_dim=False)
    torch.testing.assert_close(l1_norm.forward(x, divide_by_n=False)[0], forward_x_sum, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(l1_norm.forward(x, divide_by_n=True)[0], forward_x_mean, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(l1_norm.prox(x, sigma=0.5)[0], prox, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(l1_norm.prox_convex_conj(x, sigma=0.5)[0], prox_complex_conjugate, rtol=1e-3, atol=1e-3)

@pytest.mark.parametrize(
    (
        'x',
    ),
    [
        (
            100 * torch.randn(16,8,1,160,160),
        ),
    ],
)
def test_l1_moreau(
    x,
):
    """Test if L1 norm matches expected values."""
    divide_by_n = False
    dim = (-2,-1)
    l1_norm = L1Norm(weight=1, keep_dim=True, dim=dim, divide_by_n=divide_by_n)
    sigma = 0.5
    x_new = (l1_norm.prox(x, sigma=sigma)[0] + sigma * (l1_norm.prox_convex_conj(x / sigma, 1./sigma))[0])
    torch.testing.assert_close(x, x_new, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    (
        'x',
        'forward_x',
        'prox',
        'prox_complex_conjugate',
    ),
    [
        (
            torch.tensor([1.0, -1.0], dtype=torch.float32),
            torch.tensor((1), dtype=torch.float32),
            torch.tensor([0.9, -0.9], dtype=torch.complex64),
            torch.tensor([0.95, -0.95], dtype=torch.complex64),
        ),
    ],
)
def test_l1_functional_mixed_real_complex(
    x,
    forward_x,
    prox,
    prox_complex_conjugate,
):
    """Test if L1 norm matches expected values."""
    l1_norm = L1Norm(weight=1, target=torch.tensor([0.5+0j, -0.5+0j], dtype=torch.complex64))
    torch.testing.assert_close(l1_norm.forward(x)[0], forward_x, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(l1_norm.prox(x, sigma=0.1)[0], prox, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(l1_norm.prox_convex_conj(x, sigma=0.1)[0], prox_complex_conjugate, rtol=1e-3, atol=1e-3)

@pytest.mark.parametrize(
    (
        'x',
        'forward_x',
        'prox',
        'prox_complex_conjugate',
    ),
    [
        (
            torch.tensor([1.0+0j, -1.0+0j], dtype=torch.complex64),
            torch.tensor((1), dtype=torch.float32),
            torch.tensor([0.9, -0.9], dtype=torch.complex64),
            torch.tensor([0.95, -0.95], dtype=torch.complex64),
        ),
    ],
)
def test_l1_functional_mixed_real_complex(
    x,
    forward_x,
    prox,
    prox_complex_conjugate,
):
    """Test if L1 norm matches expected values."""
    l1_norm = L1Norm(weight=1, target=torch.tensor([0.5, -0.5], dtype=torch.float32), keep_dim=False)
    torch.testing.assert_close(l1_norm.forward(x)[0], forward_x, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(l1_norm.prox(x, sigma=0.1)[0], prox, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(l1_norm.prox_convex_conj(x, sigma=0.1)[0], prox_complex_conjugate, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    (
        'x',
        'forward_x',
        'prox',
        'prox_complex_conjugate',
    ),
    [
        (
            torch.tensor([1.0 + 0.5j, 2.0 + 0.5j, -1.0 - 0.5j], dtype=torch.complex64),
            torch.tensor((2.1213), dtype=torch.float32),
            torch.tensor([0.929 + 0.429j, 1.929 + 0.429j, -0.929 - 0.429j], dtype=torch.complex64),
            torch.tensor([0.884 + 0.465j, 0.965 + 0.261j, -0.884 - 0.465j], dtype=torch.complex64),
        ),
    ],
)
def test_l1_functional(
    x,
    forward_x,
    prox,
    prox_complex_conjugate,
):
    """Test if L1 norm matches expected values."""
    l1_norm = L1Norm(weight=1, target=torch.tensor([0.5 + 0j, 1.5 + 0j, -0.5 + 0j]), keep_dim=False)
    torch.testing.assert_close(l1_norm.forward(x)[0], forward_x, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(l1_norm.prox(x, sigma=0.1)[0], prox, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(l1_norm.prox_convex_conj(x, sigma=0.1)[0], prox_complex_conjugate, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    (
        'x',
        'forward_x',
        'prox',
        'prox_complex_conjugate',
    ),
    [
        (
            torch.tensor([1.0 + 0.5j, 2.0 + 0.5j, -1.0 - 0.5j], dtype=torch.complex64),
            torch.tensor((3), dtype=torch.float32),
            torch.tensor([0.4 + 0.4j, 0.4 + 0.4j, -0.4 - 0.4j], dtype=torch.complex64),
            torch.tensor([0.95 + 0.5j, 1.0 + 0.5j, -0.95 - 0.5j], dtype=torch.complex64),
        ),
    ],
)
def test_l1_functional_componentwise(
    x,
    forward_x,
    prox,
    prox_complex_conjugate,
):
    """Test if L1 norm matches expected values."""

    l1_norm = L1NormViewAsReal(weight=1, target=torch.tensor([0.5 + 0j, 1.5 + 0j, -0.5 + 0j]))
    torch.testing.assert_close(l1_norm.forward(x,keep_dim=False)[0], forward_x, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(l1_norm.prox(x, sigma=0.1)[0], prox, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(l1_norm.prox_convex_conj(x, sigma=0.1)[0], prox_complex_conjugate, rtol=1e-3, atol=1e-3)
