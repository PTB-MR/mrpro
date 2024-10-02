"""Primal-Dual Hybrid Gradient Algorithm (PDHG)"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import overload

import torch

from mrpro.algorithms.optimizers import OptimizerStatus


@dataclass
class PDHGStatus(OptimizerStatus):
    dual: Sequence[torch.Tensor]
    x_relaxed: torch.Tensor
    theta: float
    sigma: float


@overload  # 1
def pdhg(
    x0: torch.Tensor,
    f: ProximableFunctional | None,
    g: ProximableFunctional,
    L: LinearOprator | None,
    n_iterations: int = 10,
    sigma: float | None = None,
    tau: float | None = None,
    theta: float = 1.0,
    x_relax: None | torch.Tensor = None,
    y=None | tuple[torch.Tensor],
    callback: Callable[[PDHGStatus], None] | None = None,
): ...
@overload  # 1 as tuple
def pdhg(
    x0: torch.Tensor,
    f: ProximableFunctional | None,
    g: tuple[ProximableFunctional],
    L: tuple[LinearOperator] | None,
    n_iterations: int = 10,
    sigma: float | None = None,
    tau: float | None = None,
    theta: float = 1.0,
    x_relax: None | torch.Tensor = None,
    y=None | tuple[torch.Tensor,],
    callback: Callable[[PDHGStatus], None] | None = None,
): ...


@overload  # 2
def pdhg(
    x0: torch.Tensor,
    f: ProximableFunctional | None,
    g: tuple[ProximableFunctional, ProximableFunctional],
    L: tuple[LinearOperator, LinearOperator] | None,
    n_iterations: int = 10,
    sigma: float | None = None,
    tau: float | None = None,
    theta: float = 1.0,
    x_relax: None | torch.Tensor = None,
    y=None | tuple[torch.Tensor, torch.Tensor],
    callback: Callable[[PDHGStatus], None] | None = None,
): ...


@overload  # 3
def pdhg(
    x0: torch.Tensor,
    f: ProximableFunctional | None,
    g: tuple[ProximableFunctional, ProximableFunctional, ProximableFunctional],
    L: tuple[LinearOperator, LinearOperator, LinearOperator] | None,
    n_iterations: int = 10,
    sigma: float | None = None,
    tau: float | None = None,
    theta: float = 1.0,
    x_relax: None | torch.Tensor = None,
    y=None | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    callback: Callable[[PDHGStatus], None] | None = None,
): ...


@overload  # 4 or more
def pdhg(
    x0: torch.Tensor,
    f: ProximableFunctional | None,
    g: tuple[
        ProximableFunctional,
        ProximableFunctional,
        ProximableFunctional,
        ProximableFunctional,
        *tuple[ProximableFunctional, ...],
    ],
    L: tuple[LinearOperator, LinearOperator, LinearOperator, LinearOperator, *tuple[LinearOperator, ...]] | None,
    n_iterations: int = 10,
    sigma: float | None = None,
    tau: float | None = None,
    theta: float = 1.0,
    x_relax: None | torch.Tensor = None,
    y=None | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, *tuple[torch.Tensor, ...]],
    callback: Callable[[PDHGStatus], None] | None = None,
): ...


def pdhg(
    x0: torch.Tensor,
    f: ProximableFunctional | None,
    g: ProximableFunctional | tuple[ProximableFunctional, ...],
    L: tuple[LinearOperator, ...] | LinearOprator | None,
    n_iterations: int = 10,
    sigma: float | None = None,
    tau: float | None = None,
    theta: float = 1.0,
    x_relax=None,
    y=None,
    callback: Callable[[PDHGStatus], None] | None = None,
):
    """Primal-Dual Hybrid Gradient Algorithm (PDHG)
    Solves min_x f(x) + g(L x)
    with linear operator L and proximable functionals f and g.
    L can be supplied as a tuple of LinearOperators, interpreted as a vertical stack: L':=(L[0],L[1],...)^T.
    In that case, g should also be a tuple of functionals and g is interpreted as the separable sum: g'(y[0],y[1],...)=g[0]y[0] + g[0]y[1] +  ...
    For example, for TV regularized MR reconstruction .....

    If sigma and tau are not supplied, they are chose as 1/|L|_op.

    Parameters
    ----------
    """
    gs = g if isinstance(g, Iterable) else (g,)
    if L is None:
        L = (IdentityOp,) * len(gs)
    elif isinstance(L, Iterable):
        Ls = L
    else:
        Ls = (L,)
    if len(Ls) != len(g):
        raise ValueError('g and L must have same length')
    if x_relax is None:
        x_relaxed = x0
    if y is None:
        ys = [L(x) * 0.0 for L in Ls]
    elif len(ys) != len(gs):
        raise ValueError('if dual y is supplied, it should be a tuple of same length as the tuple of g')
    else:
        ys = y
    x = x0
    for i in range(n_iterations):
        ys = [g.prox_convex_conj(y + sigma * L * x_relax, sigma=sigma) for g, y, L in zip(gs, ys, Ls, strict=True)]

        x_new = x - tau * sum(L.adjoint(y) for g, y, L in zip(ys, Ls, strict=False))
        if f is not None:
            x_new = f.prox(x_new, sigma=tau)
        x_relaxed = x + theta * (x_new - x)
        x = x_new
        if callback is not None:
            status = PDHGStatus(iteration_number=i, sigma=sigma, tau=tau, dual=ys, solution=x, x_relaxed=x_relax)
            callback(status)
    return x
