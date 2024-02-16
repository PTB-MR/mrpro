import torch
from einops import rearrange

from mrpro.operators import SignalModel


class FatWater(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor]):
    """Signal model for Fat Water."""

    def __init__(self, fat_mod: torch.Tensor):
        """Parameters needed for fat water separation.

        Parameters
        ----------
        fat_mod
            dephasor for the fat signal --> depends on chosen spectral model
        """
        super().__init__()
        self.fat_mod = torch.nn.Parameter(fat_mod, requires_grad=fat_mod.requires_grad)

    def forward(self, w: torch.Tensor, f: torch.Tensor, phasor: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the forward model.

        Parameters
        ----------
            Image data tensors w, f with dimensions (other, c, z, y, x)
            Quantitative parameter phasor with dimensions (echoes, c, z, y, x)

        Returns
        -------
            Image data tensor (other, c, z, y, x)
        """
        # TODO: do fat_mod & phasor need other dim? Atm they do not have it.
        echoes = (w[None, :] + self.fat_mod[:, None] * f[None, :]) * phasor[:, None]  # 0th dim is new echo dim
        echoes = rearrange(echoes, 't ... c z y x -> (... t) c z y x')  # join echo dim with other dim
        return (echoes,)
