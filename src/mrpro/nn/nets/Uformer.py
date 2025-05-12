import torch
from torch.nn import Module, GELU, Linear, Sequential, Conv2d, ConvTranspose2d
from mrpro.nn.NDModules import ConvND
from mrpro.utils.sliding_window import sliding_window

import torch
from mrpro.utils.sliding_window import sliding_window
from torch.nn import Module
from einops import rearrange
from mrpro.nn.NDModules import ConvND


class LeFF(Module):
    """Locally-enhanced Feed-Forward Network.

    Part of the Uformer architecture.
    """

    def __init__(
        self,
        dim: int,
        channels_in: int = 32,
        channels_out: int = 32,
        expand_ratio: float = 4,
    ) -> None:
        """Initialize the LeFF module.

        Parameters
        ----------
        dim : int
            2 or 3, for 2D or 3D input
        channels_in : int
            Input feature dimension
        channels_out : int
            Output feature dimension
        expand_ratio : float
            Expansion ratio of the hidden dimension
        """
        super().__init__()
        hidden_dim = int(dim * expand_ratio)
        self.block = Sequential(
            ConvND(dim)(channels_in, hidden_dim, 1),
            GELU(),
            ConvND(dim)(hidden_dim, hidden_dim, kernel_size=3, groups=hidden_dim, stride=1, padding=1),
            GELU(),
            ConvND(dim)(hidden_dim, channels_out, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LeWinTransformerBlock(Module):
    def __init__(
        self,
        dim,
        channels,
        input_resolution,
        num_heads,
        win_size=8,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        token_projection='linear',
    ):
        super().__init__()
        self.channels = channels
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        self.modulator = Embedding(win_size * win_size, channels)  # modulator
        self.norm1 = norm_layer(channels)
        self.attn = WindowAttention(
            channels,
            win_size=to_2tuple(self.win_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            token_projection=token_projection,
        )

        self.norm2 = norm_layer(channels)
        mlp_hidden_dim = int(channels * mlp_ratio)
        self.mlp = LeFF(channels, mlp_hidden_dim)

    def extra_repr(self) -> str:
        return (
            f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, '
            f'win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio},modulator={self.modulator}'
        )

    def forward(self, x, mask=None):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))

        ## input mask

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        shifted_x = x
        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C
        wmsa_in = self.with_pos_embed(x_windows, self.modulator.weight)
        attn_windows = self.attn(wmsa_in, mask=attn_mask)  # nW*B, win_size*win_size, C
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class SAM(Module):
    """Spatial Attention Module.

    Part of the Uformer architecture.
    """

    def __init__(self, dim, channels):
        super().__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img
