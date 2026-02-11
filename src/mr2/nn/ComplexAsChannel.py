"""ComplexAsChannel: handling complex-valued tensors as channels."""

import torch
from einops import rearrange
from torch.nn import Module

from mr2.nn.CondMixin import CondMixin, call_with_cond


class ComplexAsChannel(CondMixin, Module):
    """Wrap module to treat complex numbers as a channel dimension."""

    def __init__(self, module: Module, convert_back: bool = True):
        """Initialize the ComplexAsChannel module.

        Wraps a module to treat complex numbers as a channel dimension.
        For each complex tensor in the input, real and imaginary parts are concatenated along the channel dimension
        before being passed to the wrapped module.


        Parameters
        ----------
        module
            The module to wrap. Should output a single real tensor.
        convert_back
            If True, the output is converted back to a complex tensor.
            The output should have a number of channels that is a multiple of 2.
        """
        super().__init__()
        self.module = module
        self.convert_back = convert_back

    def __call__(self, *x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the module.

        Parameters
        ----------
        x
            The input tensor.
        cond
            The conditioning tensor (if used by the wrapped module)
        """
        return super().__call__(*x, cond=cond)

    def forward(self, *x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the module."""
        x_real = [
            rearrange(torch.view_as_real(c), 'batch channel ... complex -> batch (channel complex) ...')
            if c.is_complex()
            else c
            for c in x
        ]

        y = call_with_cond(self.module, *x_real, cond=cond)

        if self.convert_back:
            y = rearrange(y, 'b (channel complex) ... -> b channel ... complex', complex=2).contiguous()
            y = torch.view_as_complex(y)
        return y
