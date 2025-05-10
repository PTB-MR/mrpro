class LeFF(nn.Module):
    """Fast Locally-enhanced Feed-Forward Network."""

    def __init__(
        self,
        dim: int = 32,
        hidden_dim: int = 128,
        act_layer: Callable[[], nn.Module] = nn.GELU,
    ) -> None:
        """
        Parameters
        ----------
        dim : int
            Input and output feature dimension.
        hidden_dim : int
            Hidden feature dimension.
        act_layer : Callable
            Activation function.
        """
        super().__init__()
        from torch_dwconv import DepthwiseConv2d  # Local import for optional dependency

        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim), act_layer())
        self.dwconv = nn.Sequential(
            DepthwiseConv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer(),
        )
        self.linear2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))
        x = self.linear1(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=hh, w=hh)
        x = self.dwconv(x)
        x = rearrange(x, 'b c h w -> b (h w) c', h=hh, w=hh)
        x = self.linear2(x)
        return x
