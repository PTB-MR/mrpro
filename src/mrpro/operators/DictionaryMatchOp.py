"""Dictionary Matching Operator."""

from collections.abc import Callable

import einops
import torch
from typing_extensions import Self, TypeVarTuple, Unpack

from mrpro.operators import Operator

Tin = TypeVarTuple('Tin')


class DictionaryMatchOp(Operator[torch.Tensor, tuple[*Tin]]):
    """Dictionary Matching Operator class."""

    def __init__(self, generating_function: Callable[[Unpack[Tin]], tuple[torch.Tensor,]]):
        super().__init__()
        self._f = generating_function
        self.x: list[torch.Tensor] = []
        self.y = torch.tensor([])

    def append(self, *x: Unpack[Tin]) -> Self:
        (newy,) = self._f(*x)
        newy = newy.to(dtype=torch.complex64)
        newy = newy / torch.linalg.norm(newy, dim=0, keepdim=True)
        newy = newy.flatten(start_dim=1)
        newx = [x.flatten() for x in torch.broadcast_tensors(*x)]
        if not self.x:
            self.x = newx
            self.y = newy
            return self
        self.x = [torch.cat(old, new) for old, new in zip(self.x, newx, strict=True)]
        self.y = torch.cat((self.y, newy))
        return

    def forward(self, input_signal: torch.Tensor) -> tuple[Unpack[Tin]]:
        similar = einops.einsum(input_signal, self.y, 't ..., t idx  -> idx ...')
        idx = torch.argmax(torch.abs(similar), dim=0)
        match = [x[idx] for x in self.x]
        return match
