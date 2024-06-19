"""Tests for the proximal gradient descent."""

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

import torch
from mrpro.algorithms.optimizers import pgd
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.operators import FastFourierOp
from mrpro.operators.functionals import L1Norm
from mrpro.operators.functionals import L2NormSquared
from mrpro.phantoms import EllipsePhantom


def test_pgd_convergence_fft_example():
    dim = SpatialDimension(x=100, y=100, z=1)
    ellipse = EllipsePhantom()
    image = ellipse.image_space(dim)
    fft = FastFourierOp()
    (kspace,) = fft(image)

    l2 = L2NormSquared(target=kspace)
    f = l2 @ fft
    g = L1Norm()

    initial_value = torch.ones_like(image)
    pgd_solution = pgd(
        f=f,
        g=g,
        initial_value=initial_value,
        stepsize=0.5,
        reg_parameter=0.01,
        max_iterations=200,
        backtrack_factor=1.0,
    )

    assert f(pgd_solution)[0] + g(pgd_solution)[0] < f(initial_value)[0] + g(initial_value)[0]


def test_pgd_convergence_backtracking_fft_example():
    dim = SpatialDimension(x=100, y=100, z=1)
    ellipse = EllipsePhantom()
    image = ellipse.image_space(dim)
    fft = FastFourierOp()
    (kspace,) = fft(image)

    l2 = L2NormSquared(target=kspace)
    f = l2 @ fft
    g = L1Norm()

    initial_value = torch.ones_like(image)
    pgd_solution = pgd(
        f=f,
        g=g,
        initial_value=initial_value,
        stepsize=2.0,
        reg_parameter=0.01,
        max_iterations=200,
        backtrack_factor=0.75,
    )

    assert f(pgd_solution)[0] + g(pgd_solution)[0] < f(initial_value)[0] + g(initial_value)[0]


def test_pgd_convergence_backtracking_denoising_example():
    dim = SpatialDimension(x=100, y=100, z=1)
    ellipse = EllipsePhantom()
    image = ellipse.image_space(dim)
    noise = torch.randn_like(image)
    noisy_image = image + noise

    f = L2NormSquared(target=noisy_image)
    g = L1Norm()

    initial_value = torch.ones_like(image)
    pgd_solution = pgd(
        f=f,
        g=g,
        initial_value=initial_value,
        stepsize=2.0,
        reg_parameter=0.01,
        max_iterations=200,
        backtrack_factor=0.75,
    )

    assert f(pgd_solution)[0] + g(pgd_solution)[0] < f(initial_value)[0] + g(initial_value)[0]
