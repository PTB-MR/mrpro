"""Tests for L1-functional."""

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
from mrpro.operators.functionals.l1 import L1Norm
from mrpro.operators.functionals.l1 import L1NormViewAsReal


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
    l1_norm = L1Norm(weight=1, target=torch.tensor([0.5, -0.5], dtype=torch.float32))
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
    l1_norm = L1Norm(weight=1, target=torch.tensor([0.5 + 0j, 1.5 + 0j, -0.5 + 0j]))
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
    torch.testing.assert_close(l1_norm.forward(x)[0], forward_x, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(l1_norm.prox(x, sigma=0.1)[0], prox, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(l1_norm.prox_convex_conj(x, sigma=0.1)[0], prox_complex_conjugate, rtol=1e-3, atol=1e-3)
