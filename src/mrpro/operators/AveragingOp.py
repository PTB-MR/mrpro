"""Averaging operator."""

from collections.abc import Sequence
from warnings import warn

import torch

from mrpro.operators.LinearOperator import LinearOperator


class AveragingOp(LinearOperator):
    """Averaging operator.

    This operator averages the input tensor along a specified dimension.
    The averaging is performed over groups of elements defined by the `idx` parameter.
    The output tensor will have the same shape as the input tensor, except for the `dim` dimension,
    which will have a size equal to the number of groups specified in `idx`. For each group,
    the average of the elements in that group is computed.

    For example, this operator can be used to simulate the effect of a sliding window average
    in a signal model.
    """

    def __init__(
        self, dim: int, idx: Sequence[Sequence[int] | torch.Tensor | slice], domain_size: int | None = None
    ) -> None:
        """Initialize the averaging operator.

        Parameters
        ----------
        dim
            The dimension along which to average.
        idx
            The indices of the input tensor to average over. Each element of the sequence will result in a
            separate entry in the `dim` dimension of the output tensor.
            The entries can be either a sequence of integers or an integer tensor, a slice object, or a boolean tensor.
        domain_size
            The size of the input along `dim`. It is only used in the `adjoint` method.
            If not set, the size will be guessed from the input tensor during the forward pass.
        """
        super().__init__()
        self.domain_size = domain_size
        self._last_domain_size = domain_size
        self.idx = idx
        self.dim = dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the averaging operator to the input tensor."""
        if self.domain_size and self.domain_size != x.shape[self.dim]:
            raise ValueError(f'Expected domain size {self.domain_size}, got {x.shape[self.dim]}')
        self._last_domain_size = x.shape[self.dim]

        placeholder = (slice(None),) * (self.dim % x.ndim)
        averaged = torch.stack([x[*placeholder, i].mean(self.dim) for i in self.idx], self.dim)
        return (averaged,)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the adjoint of the averaging operator to the input tensor."""
        if self.domain_size is None:
            if self._last_domain_size is None:
                raise ValueError('Domain size is not set. Please set it explicitly.')
            warn(
                'Domain size is not set. Guessing the last used input size of the forward pass. '
                'Consider setting the domain size explicitly.',
                stacklevel=2,
            )
            self.domain_size = self._last_domain_size

        adjoint = x.new_zeros(*x.shape[: self.dim], self.domain_size, *x.shape[self.dim + 1 :])
        placeholder = (slice(None),) * (self.dim % x.ndim)
        for i, group in enumerate(self.idx):
            if isinstance(group, slice):
                n = len(range(*group.indices(self.domain_size)))
            elif isinstance(group, torch.Tensor) and group.dtype == torch.bool:
                n = group.sum()
            else:
                n = len(group)

            adjoint[*placeholder, group] += (
                x[*placeholder, i, None].expand(*x.shape[: self.dim], n, *x.shape[self.dim + 1 :]) / n
            )

        return (adjoint,)
