

import torch
import torch.nn.functional as F  # noqa: N812

from mrpro.operators._Functional import Functional

class L1Norm(Functional):
    
    def __init__(self, g:torch.Tensor=None):
        super().__init__(lam=1)
        self.g = g

    def forward(self, x):
        return x.abs().inner(torch.ones(x))
    
    def prox(self, x:torch.Tensor, sigma):
        # diff = x - g
        if self.g is not None:
            diff = x - self.g
        else:
            diff = x
        # x - (x - g) / max(|x -g|/sigma*lam, 1)
        denom = diff.abs()
        denom /= sigma*self.lam
        denom = torch.max(torch.ones(denom), denom)
        # out = (x - g)/denom
        out = torch.div(diff,denom)
        return x-out
        
    def prox_convex_conj(self, x, sigma):
        # diff = x - sigma * g
        if self.g is not None:
            diff = x-sigma*self.g
        else:
            diff = x
        # out = max( |x - sigma * g|, lam) / lam
        out = max(diff.abs(),self.lam)/self.lam
        return diff/out
        