"""MSE-Functional."""

from collections.abc import Sequence

import torch

from mrpro.operators.functionals.L2NormSquared import L2NormSquared


class MSE(L2NormSquared):
    r"""Functional class for the mean squared error."""

    def __init__(
        self,
        target: torch.Tensor | None | complex = None,
        weight: torch.Tensor | complex = 1.0,
        dim: int | Sequence[int] | None = None,
        divide_by_n: bool = True,
        keepdim: bool = False,
    ) -> None:
        r"""Initialize MSE Functional.

        The MSE functional is given by
        :math:`f: C^N -> [0, \infty), x -> 1/N \| W (x-b)\|_2^2`,
        where :math:`W` is either a scalar or tensor that corresponds to a (block-) diagonal operator
        that is applied to the input. The division by `N` can be disabled by setting `divide_by_n=False`
        For more details also see :class:`mrpro.operators.functionals.L2NormSquared`

        Parameters
        ----------
        target
            target element - often data tensor (see above)
        weight
            weight parameter (see above)
        dim
            dimension(s) over which functional is reduced.
            All other dimensions of  `weight ( x - target)` will be treated as batch dimensions.
        divide_by_n
            if true, the result is scaled by the number of elements of the dimensions index by `dim` in
            the tensor `weight ( x - target)`. If true, the functional is thus calculated as the mean,
            else the sum.
        keepdim
            if true, the dimension(s) of the input indexed by dim are maintained and collapsed to singeltons,
            else they are removed from the result.

        """
        super().__init__(weight=weight, target=target, dim=dim, divide_by_n=divide_by_n, keepdim=keepdim)
