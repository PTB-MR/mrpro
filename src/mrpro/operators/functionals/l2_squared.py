

import torch
import torch.nn.functional as F  # noqa: N812

from mrpro.operators._Operator import Operator

class L2NormSquared(Operator):
    
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return x.inner(x)
    
    def convex_conj(self):
        return 1/4 * L2NormSquared()
    
    def prox(self):
        return None
    
    def prox_convex_conj(self):
        return self.prox(self.convex_conj())