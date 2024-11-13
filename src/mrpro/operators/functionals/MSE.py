"""MSE-Functional."""

from collections.abc import Sequence

import torch

from mrpro.operators.functionals.L2NormSquared import L2NormSquared


class MSE(L2NormSquared):
    r"""Functional class for the mean square error.

    This makes use of the functional L2NormSquared.
    """

    def __init__(
        self,
        weight: torch.Tensor | complex = 1.0,
        target: torch.Tensor | None | complex = None,
        dim: int | Sequence[int] | None = None,
        divide_by_n: bool = True,
        keepdim: bool = False,
    ) -> None:
        r"""Initialize a Functional.

        We assume that functionals are given in the form
        :math:`f(x) = \phi ( weight ( x - target))`
        for some functional :math:`\phi`.

        Parameters
        ----------
         functional
            functional to be employed
        weight
            weight parameter (see above)
        target
            target element - often data tensor (see above)
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
