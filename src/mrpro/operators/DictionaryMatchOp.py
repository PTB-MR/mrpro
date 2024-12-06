from collections.abc import Callable

import einops
import torch
from typing_extensions import Self, TypeVarTuple, Unpack, reveal_type

from mrpro.operators import Operator


def f(x: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor]:
    t = torch.tensor([1, 2, 3])
    x = einops.einsum(t, x, 't, ... -> t ...')
    return (x.sin() + x2,)


Tin = TypeVarTuple('Tin')


class DictionaryMatchOp(Operator[torch.Tensor, tuple[*Tin]]):
    def __init__(self, generating_function: Callable[[Unpack[Tin]], tuple[torch.Tensor,]]):
        super().__init__()
        self._f = generating_function
        self.x: list[torch.Tensor] = []
        self.y = torch.tensor([])

    def append(self, *x: Unpack[Tin]) -> Self:
        (newy,) = self._f(*x)

        newy = newy / torch.linalg.norm(newy, dim=0, keepdim=True)
        newy = newy.flatten(start_dim=1)
        newx = [x.flatten() for x in torch.broadcast_tensors(*x)]

        if not self.x:
            self.x = newx
            self.y = newy
            return self

        self.x = [torch.cat(old, new) for old, new in zip(self.x, newx, strict=True)]
        self.y = torch.cat((self.y, newy))
        return self

    def forward(self, y: torch.Tensor) -> tuple[Unpack[Tin]]:
        y = y / torch.linalg.norm(y, dim=0, keepdim=True)
        similarity = einops.einsum(y, self.y, 't ..., t idx -> idx ...')
        idx = torch.argmax(similarity)
        x = tuple(x[idx] for x in self.x)
        return x


if __name__ == '__main__':
    torch.random.manual_seed(0)
    d = DictionaryMatchOp(f).append(torch.rand(1, 200), torch.rand(100, 1))
    reveal_type(d)

    true = torch.rand(2, 1)
    (y,) = f(*true)
    pred = d(y)

    print(pred, true)
