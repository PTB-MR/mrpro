import torch
import torch.nn.functional as F  # noqa: N812
from abc import ABC, abstractmethod
from mrpro.operators._Operator import Operator

class Functional(Operator[torch.Tensor,tuple[torch.Tensor]]):
    
    def __init__(self, lam:float=1.):
        super().__init__()
        self.lam = lam
        
class ProximableFunctional(Functional):
    def prox(self, x:torch.Tensor, sigma:torch.Tensor) -> tuple[torch.Tensor]:
        """Apply forward operator."""
        return x-self.prox_convex_conj(x, sigma)
        
    def prox_convex_conj(self, x:torch.Tensor, sigma:torch.Tensor) -> tuple[torch.Tensor]:
        """Apply forward operator."""
        return x-self.prox(x, sigma)
        ...

        
    
        
