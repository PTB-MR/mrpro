"""Diffusion Transformer (DiT)."""

from collections.abc import Sequence
from math import prod

import torch
from torch.nn import Linear, Module, Parameter, SiLU

from mr2.nn.attention.MultiHeadAttention import MultiHeadAttention
from mr2.nn.CondMixin import CondMixin
from mr2.nn.LayerNorm import LayerNorm
from mr2.nn.nets.MLP import MLP
from mr2.nn.Sequential import Sequential
from mr2.operators.PatchOp import PatchOp
from mr2.utils.to_tuple import to_tuple


class DiTBlock(CondMixin, Module):
    """DiT block with adaptive layer normalization and residual gating.

    References
    ----------
    .. [DiT] Peebles, W., & Xie, S. Scalable Diffusion Models with Transformers.
       ICCV 2023, https://arxiv.org/abs/2212.09748
    """

    features_last: bool

    def __init__(
        self,
        n_channels: int,
        n_heads: int,
        cond_dim: int,
        mlp_ratio: float = 4.0,
        features_last: bool = True,
    ):
        """Initialize a DiT block.

        Parameters
        ----------
        n_channels
            Number of channels in the input and output.
        n_heads
            Number of attention heads.
        cond_dim
            Number of channels in the conditioning tensor.
        mlp_ratio
            Ratio of hidden MLP channels to input channels.
        features_last
            Whether the features are in the last dimension of the input tensor.
        """
        super().__init__()
        self.features_last = features_last
        self.norm1 = LayerNorm(n_channels, features_last=True, cond_dim=cond_dim)
        self.attn = MultiHeadAttention(
            n_channels_in=n_channels,
            n_channels_out=n_channels,
            n_heads=n_heads,
            features_last=True,
        )
        self.norm2 = LayerNorm(n_channels, features_last=True, cond_dim=cond_dim)
        self.mlp = MLP(
            n_channels_in=n_channels,
            n_channels_out=n_channels,
            n_features=(int(n_channels * mlp_ratio),),
            norm='none',
            activation='gelu',
            cond_dim=0,
            features_last=True,
        )
        self.gate = Sequential(
            SiLU(),
            Linear(cond_dim, 2 * n_channels),
        )
        linear = self.gate[-1]
        if isinstance(linear, Linear):
            torch.nn.init.zeros_(linear.weight)
            torch.nn.init.zeros_(linear.bias)

    def __call__(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the DiT block.

        Parameters
        ----------
        x
            Input tensor.
        cond
            Conditioning tensor.

        Returns
        -------
            Output tensor.
        """
        return super().__call__(x, cond=cond)

    def forward(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the DiT block."""
        if not self.features_last:
            x = x.moveaxis(1, -1)

        gate_msa, gate_mlp = self.gate(cond).unsqueeze(-2).chunk(2, dim=-1) if cond is not None else (1.0, 1.0)
        x = x + gate_msa * self.attn(self.norm1(x, cond=cond))
        x = x + gate_mlp * self.mlp(self.norm2(x, cond=cond))

        if not self.features_last:
            x = x.moveaxis(-1, 1)

        return x


class DiT(Module):
    """DiT model.

    DiT is a vision transformer popularized by [DiT]_.
    Often used for latent diffusion models, but also suitable for image restoration etc.

    References
    ----------
    .. [DiT] Peebles, W., & Xie, S. Scalable Diffusion Models with Transformers.
       ICCV 2023, https://arxiv.org/abs/2212.09748

    """

    grid_size: tuple[int, ...]
    patch_size: tuple[int, ...]
    n_channels_out: int

    def __init__(
        self,
        n_dim: int,
        n_channels_in: int,
        cond_dim: int,
        input_size: int | Sequence[int] = 32,
        patch_size: int | Sequence[int] = 2,
        n_channels_out: int | None = None,
        hidden_dim: int = 1152,
        depth: int = 28,
        n_heads: int = 16,
        mlp_ratio: float = 4.0,
    ) -> None:
        """Initialize DiT.

        Parameters
        ----------
        n_dim
            Number of spatial dimensions.
        n_channels_in
            Number of channels in the input tensor.
        cond_dim
            Dimension of the conditioning tensor.
        input_size
            Input spatial size. If scalar, the same size is used for all spatial dimensions.
        patch_size
            Patch size. If scalar, the same patch size is used for all spatial dimensions.
        n_channels_out
            Number of output channels. If `None`, use `n_channels_in`.
        hidden_dim
            Transformer hidden dimension.
        depth
            Number of transformer blocks.
        n_heads
            Number of attention heads.
        mlp_ratio
            Ratio of hidden MLP channels to input channels.
        """
        super().__init__()
        self.n_dim = n_dim
        self.input_size = to_tuple(n_dim, input_size)
        self.patch_size = to_tuple(n_dim, patch_size)

        if any(s % p != 0 for s, p in zip(self.input_size, self.patch_size, strict=True)):
            raise ValueError(f'Input size {self.input_size} must be divisible by patch size {self.patch_size}.')
        if hidden_dim % (2 * n_dim) != 0:
            raise ValueError(f'Hidden dimension {hidden_dim} must be divisible by 2 * {n_dim=}.')

        self.grid_size = tuple(s // p for s, p in zip(self.input_size, self.patch_size, strict=True))
        self.n_patches = prod(self.grid_size)
        self.hidden_dim = hidden_dim

        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_in if n_channels_out is None else n_channels_out

        spatial_dim = tuple(range(2, 2 + n_dim))
        self.patch_op = PatchOp(
            dim=spatial_dim,
            patch_size=self.patch_size,
            stride=self.patch_size,
            dilation=1,
            domain_size=self.input_size,
        )

        patch_volume = prod(self.patch_size)
        self.in_proj = Linear(n_channels_in * patch_volume, hidden_dim)
        self.pos_embed = Parameter(torch.zeros(self.n_patches, hidden_dim), requires_grad=False)

        self.blocks = Sequential(
            *(
                DiTBlock(
                    n_channels=hidden_dim,
                    n_heads=n_heads,
                    cond_dim=cond_dim,
                    mlp_ratio=mlp_ratio,
                    features_last=True,
                )
                for _ in range(depth)
            )
        )

        self.final_layer = Sequential(
            LayerNorm(hidden_dim, features_last=True, cond_dim=cond_dim),
            Linear(hidden_dim, patch_volume * self.n_channels_out),
        )

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize network weights."""

        def _basic_init(module: Module) -> None:
            if isinstance(module, Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        self.apply(_basic_init)

        w = self.in_proj.weight.data
        torch.nn.init.xavier_uniform_(w.reshape(w.shape[0], -1))
        if self.in_proj.bias is not None:
            torch.nn.init.zeros_(self.in_proj.bias)

        for block in self.blocks:
            if isinstance(block, DiTBlock):
                gate_linear = block.gate[-1]
                if isinstance(gate_linear, Linear):
                    torch.nn.init.zeros_(gate_linear.weight)
                    torch.nn.init.zeros_(gate_linear.bias)

        w = 1.0 / (10000 ** torch.linspace(0, 1, self.hidden_dim // (2 * len(self.grid_size))))
        x = torch.stack(torch.meshgrid(*[torch.arange(s).float() for s in self.grid_size], indexing='ij'), dim=-1)
        wx = w * x.unsqueeze(-1)
        pos_embed = torch.cat([torch.sin(wx), torch.cos(wx)], dim=-1).reshape(-1, self.hidden_dim)
        self.pos_embed.data.copy_(pos_embed.to(self.pos_embed.data))

    def forward(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply DiT.

        Parameters
        ----------
        x
            Input tensor with shape `batch, channels, *spatial_dims`.
        cond
            Conditioning tensor.

        Returns
        -------
            Output tensor with shape `batch, out_channels, *spatial_dims`.
        """
        x = self.patch_op(x)[0].swapaxes(0, 1).flatten(2)
        x = self.in_proj(x)
        x = x + self.pos_embed
        x = self.blocks(x, cond=cond)
        x = self.final_layer(x, cond=cond)
        x = x.unflatten(-1, (self.n_channels_out, *self.patch_size)).swapaxes(0, 1)
        (x,) = self.patch_op.adjoint(x)
        return x
