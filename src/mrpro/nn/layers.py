import torch
from einops import rearrange
from torch.nn import Identity, Linear, Module, Parameter, ReLU, Sequential, Sigmoid, SiLU

from mrpro.nn.NDModules import AdaptiveAvgPoolND, ConvND
from mrpro.utils.reshape import unsqueeze_tensors_right


class EmbMixin: ...


class SqueezeExcitation(Module):
    """Squeeze-and-Excitation block.

    Sequeeze-and-Excitation block from [SE]_.

    References
    ----------
    ..[SE] Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks." CVPR 2018, https://arxiv.org/abs/1709.01507
    """

    def __init__(self, dim: int, input_channels: int, squeeze_channels: int) -> None:
        """Initialize SqueezeExcitation.

        Parameters
        ----------
        dim
            The dimension of the input tensor.
        input_channels
            The number of channels in the input tensor.
        squeeze_channels
            The number of channels in the squeeze tensor.
        """
        super().__init__()
        self.scale = Sequential(
            AdaptiveAvgPoolND(dim, 1),
            ConvND(dim, input_channels, squeeze_channels, 1),
            ReLU(),
            ConvND(dim, squeeze_channels, input_channels, 1),
            Sigmoid(),
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SqueezeExcitation.

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
        """Apply SqueezeExcitation."""
        return x * self.scale(x)


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
        self.qkv = ConvND(dim, channels, channels * 3, kernel_size=1, bias=True)
        self.qkv_dwconv = ConvND(
            dim,
            channels * 3,
            channels * 3,
            kernel_size=3,
            groups=channels * 3,
            bias=False,
        )
        self.project_out = ConvND(dim, channels, channels, kernel_size=1, bias=True)

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
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=self.temperature)
        out = rearrange(out, '... head points c -> ... (head c) points').reshape(x.shape)
        out = self.project_out(out)
        return out


class GroupNorm32(torch.nn.GroupNorm):
    """A 32-bit GroupNorm.

    Casts to float32 before calling the parent class to avoid instabilities in mixed precision training.
    """

    def __init__(self, channels: int, groups: int | None = None):
        """Initialize GroupNorm32.

        Parameters
        ----------
        channels
            The number of channels in the input tensor.
        groups
            The number of groups to use. If None, the number of groups is determined automatically as
            a power of 2 that is less than or equal to 32 and leaves at least 4 channels per group.
        """
        if groups is None:
            groups_ = channels & -channels
            while (groups_ >= channels // 4) or groups_ > 32:
                groups_ //= 2
        else:
            groups_ = groups
        super().__init__(groups_, channels)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x.float()).type(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super(x.float).type(x.dtype)


class EmbSequential(Sequential):
    def __call__(self, x: torch.Tensor, emb: torch.Tensor | None) -> torch.Tensor:
        return super().__call__(x, emb)

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None) -> torch.Tensor:
        for module in self:
            if isinstance(module, EmbMixin):
                x = module(x, emb)
            else:
                x = module(x)
        return x


def call_with_emb(module: Module, x: torch.Tensor, emb: torch.Tensor | None) -> torch.Tensor:
    if isinstance(module, EmbMixin):
        return module(x, emb)
    return module(x)


class FiLM(Module, EmbMixin):
    def __init__(self, channels: int, channels_emb: int) -> None:
        super().__init__()
        self.project = Sequential(
            SiLU(),
            Linear(channels_emb, 2 * channels),
        )

    def __call__(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return super().__call__(x, emb)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        emb = self.project(emb)
        scale, shift = emb.chunk(2, dim=1)
        scale, shift = unsqueeze_tensors_right(scale, shift, ndim=x.ndim)
        return x * (1 + scale) + shift


class ResBlock(Module, EmbMixin):
    def __init__(self, channels_in: int, channels_out: int, channels_emb: int, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.rezero = torch.nn.Parameter(torch.tensor(1e-6))
        self.modules = EmbSequential(
            GroupNorm32(channels_in),
            SiLU(),
            ConvND(dim, channels_in, channels_out, 3),
            GroupNorm32(channels_out),
            SiLU(),
            ConvND(dim, channels_out, channels_out, 3),
        )
        if channels_emb > 0:
            self.modules.insert(-3, FiLM(channels_out, channels_emb))

        if channels_out == channels_in:
            self.skip_connection = Identity()
        else:
            self.skip_connection = ConvND(dim, channels_in, channels_out, 1)

    def __call__(self, x: torch.Tensor, emb: torch.Tensor | None) -> torch.Tensor:
        return super().__call__(x, emb)

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None) -> torch.Tensor:
        h = self.modules(x, emb)
        x = self.skip_connection(x) + h
        return x
