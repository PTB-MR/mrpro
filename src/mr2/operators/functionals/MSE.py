"""MSE-Functional."""

from collections.abc import Sequence

import torch

from mr2.operators.functionals.L2NormSquared import L2NormSquared


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
        :math:`f: C^N \rightarrow [0, \infty), x \rightarrow 1/N \| W (x-b)\|_2^2`,
        where :math:`W` is either a scalar or tensor that corresponds to a (block-) diagonal operator
        that is applied to the input. The division by `N` can be disabled by setting `divide_by_n` to `False`.
        For more details also see :class:`mr2.operators.functionals.L2NormSquared`.

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
            If `True`, the result is scaled by the number of elements of the dimensions in the
            tensor `weight ( x - target)` indexed by `dim`. The functional is thus calculated as the mean, else the sum.
        keepdim
            If `True`, the dimension(s) of the input indexed by dim are maintained and collapsed to singletons,
            else they are removed from the result.

        """
        super().__init__(weight=weight, target=target, dim=dim, divide_by_n=divide_by_n, keepdim=keepdim)

    def __call__(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        r"""Compute the Mean Squared Error (MSE).

        Calculates :math:`1/N \| W * (x - b) \|_2^2`, where :math:`W` is `weight`,
        :math:`b` is `target`, and `N` is the number of elements over which the
        mean is computed (if `divide_by_n` is `True` at initialization of L2NormSquared).
        The squared norm is computed along dimensions specified by `dim`.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
            The MSE. If `keepdim` is `True`, the dimensions `dim` are retained
            with size 1; otherwise, they are reduced.
        """
        return super().__call__(x)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Apply forward of MSE.

        .. note::
            Prefer calling the instance of the MSE as ``operator(x)`` over directly calling this method.
            See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        # MSE uses the forward implementation of L2NormSquared
        return super().forward(x)
