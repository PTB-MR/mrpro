import torch

from mrpro.operators import LinearOperator


class WaveletOp(LinearOperator):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        return (x,)
