"""Tests for the conjugate gradient method."""

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

from mrpro.algorithms import cg
from mrpro.operators import LinearOperator
from tests import RandomGenerator


def create_self_adjoint_op(N, complex_valued_flag):
    """Create self-adjoint operator H and wrap as LinearOperator."""
    random_generator = RandomGenerator(seed=0)
    N = 32

    # generate random matrices (avoid zeros by using ab offset of 1.)
    if complex_valued_flag:
        A = 1 + random_generator.complex64_tensor(size=(N, N))
        H = A.conj().T @ A
    else:
        A = 1 + random_generator.float32_tensor(size=(N, N))
        H = A.T @ A

    class MatMult(LinearOperator):
        """Simple Linear Operator that implements matrix multiplication."""

        def __init__(self, Mat: torch.Tensor) -> None:
            super().__init__()
            self.Mat = Mat

        def forward(self, x) -> tuple[torch.Tensor]:
            return (self.Mat @ x,)

        def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor]:
            return (self.Mat.conj().T @ x,) if self.Mat.is_complex() else (self.Mat.T @ x,)

    HOp = MatMult(H)
    return HOp


@pytest.mark.parametrize('complex_valued_flag', [True, False])
def test_cg_convergence(complex_valued_flag):
    """Test if CG delivers accurate solution."""
    N = 32
    random_generator = RandomGenerator(seed=0)
    HOp = create_self_adjoint_op(N, complex_valued_flag)
    xtrue = (
        1 + random_generator.complex64_tensor(size=(N,))
        if complex_valued_flag
        else 5 * random_generator.float32_tensor(size=(N,))
    )
    (b,) = HOp(xtrue)
    x0 = torch.ones_like(xtrue)
    xcg = cg(HOp, b, x0=x0, max_iter=N)
    rtol = 1e-3 * torch.vdot(b.flatten(), b.flatten())
    atol = 1e-3

    # test if solution is accurate
    torch.testing.assert_close(xcg, xtrue, rtol=rtol, atol=atol)

    # test if cg stops if the ground-truth is the initial guess
    xcg_one_iter = cg(HOp, b, x0=xtrue, max_iter=1)
    assert (xtrue == xcg_one_iter).all()


def test_invalid_shapes():

    # check if wrongly set-up linear system throws error
    N = 32
    random_generator = RandomGenerator(seed=0)
    HOp = create_self_adjoint_op(N, complex_valued_flag=False)
    x0 = random_generator.complex64_tensor(size=(N,))
    b = random_generator.complex64_tensor(size=(N + 1,))
    with pytest.raises(ValueError, match='incompatible'):
        _ = cg(HOp, b, x0=x0, max_iter=N)
