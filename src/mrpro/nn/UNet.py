from functools import partial

import torch
from torch.nn import Module

from mrpro.nn.layers import call_with_emb


class UNetBase(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels_emb: int,
        dim: int,
        num_blocks: int,
    ) -> None: ...

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Apply to Network."""
        call = partial(call_with_emb, emb=emb)
        x = call(self.first, x)
        xs = []
        for block, down, skip in zip(self.input_blocks, self.down_blocks, self.skip_blocks, strict=False):
            x = call(block, x)
            xs.append(call(skip, x))
            x = call(down, x)
        x = call(self.middel_block, x)
        for block, up in (self.output_blocks, self.up_blocks):
            x = call(up, x)
            x = torch.cat([x, xs.pop()], dim=1)
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
