from functools import partial

import torch
from torch.nn import Module, ModuleList

from mrpro.nn.EmbMixin import call_with_emb


class UNetBase(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels_emb: int,
        dim: int,
        num_blocks: int,
    ) -> None: ...

    input_blocks: ModuleList
    down_blocks: ModuleList
    skip_blocks: ModuleList
    middle_block: Module
    output_blocks: ModuleList
    up_blocks: ModuleList
    concat_blocks: ModuleList
    last: Module
    first: Module

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Apply to Network."""
        call = partial(call_with_emb, emb=emb)
        x = call(self.first, x)
        xs = []
        for block, down, skip in zip(self.input_blocks, self.down_blocks, self.skip_blocks, strict=True):
            x = call(block, x)
            xs.append(call(skip, x))
            x = call(down, x)
        x = call(self.middle_block, x)
        for block, up, concat in zip(self.output_blocks, self.up_blocks, self.concat_blocks, strict=True):
            x = call(up, x)
            x = concat(x, xs.pop())
            x = call(block, x)
        return call(self.last, x)

    def __call__(self, x: torch.Tensor, emb: torch.Tensor | None) -> torch.Tensor:
        """Apply to Network.

        Parameters
        ----------
        x
            The input tensor.
        emb
            The embedding tensor.

        Returns
        -------
            The output tensor.
        """
        return self(x, emb)
