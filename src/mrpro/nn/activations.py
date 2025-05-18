import torch
from torch.nn import Linear, Module


class GEGLU(Module):
    r"""Gated linear unit activation function.

    References
    ----------
    ..[GLU] Shazeer, N. (2020). GLU variants improve transformer. https://arxiv.org/abs/2002.05202
    """

    def __init__(self, in_features: int, out_features: int | None = None):
        """Initialize the GEGLU activation function.

        Parameters
        ----------
        in_features : int
            The number of input features.
        out_features : int
            The number of output features. If None, the number of output features is the same as the number of input features.
        """
        super().__init__()
        self.proj = Linear(in_features, out_features * 2)

    def forward(self, x):
        h, gate = self.proj(x).chunk(2, dim=-1)
        gate = torch.nn.functional.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)
        return h * gate
