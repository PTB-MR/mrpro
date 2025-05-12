"""Transposed Attention from Restormer."""

import torch
from einops import rearrange
from torch.nn import Identity, Linear, Module, Parameter, ReLU, Sequential, Sigmoid, SiLU

from mrpro.nn.NDModules import AdaptiveAvgPoolND, ConvND, InstanceNormND
from mrpro.utils.reshape import unsqueeze_tensors_right
from mrpro.operators import Operator


class TransposedAttention(Module):
    def __init__(self, dim: int, channels: int, num_heads: int):
        """Transposed Self Attention from Restormer.

        Implements the transposed self-attention, i.e. channel-wise multihead self-attention,
        layer from Restormer [ZAM22]_.

        References
        ----------
        ..[ZAM22] Zamir, Syed Waqas, et al. "Restormer: Efficient transformer for high-resolution image restoration."
          CVPR 2022, https://arxiv.org/pdf/2111.09881.pdf

        Parameters
        ----------
        dim
            input dimension
        channels
            input channels
        num_heads
            number of attention heads
        """
        super().__init__()
        self.num_heads = num_heads
        self.temperature = Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = ConvND(dim)(channels, channels * 3, kernel_size=1, bias=True)
        self.qkv_dwconv = ConvND(dim)(
            channels * 3,
            channels * 3,
            kernel_size=3,
            groups=channels * 3,
            bias=False,
        )
        self.project_out = ConvND(dim)(channels, channels, kernel_size=1, bias=True)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transposed attention.

        Parameters
        ----------
        x
            The input tensor.

        Returns
        -------
            The output tensor.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transposed Attention."""
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = rearrange(qkv, 'b (qkv head c) ... -> qkv b head (...) c', head=self.num_heads, qkv=3)
        q = torch.nn.functional.normalize(q, dim=-1) * self.temperature
        k = torch.nn.functional.normalize(k, dim=-1)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=1.0)
        out = rearrange(out, '... head points c -> ... (head c) points').reshape(x.shape)
        out = self.project_out(out)
        return out
