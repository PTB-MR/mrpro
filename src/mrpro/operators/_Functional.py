import torch
import torch.nn.functional as F  # noqa: N812
from abc import abstractmethod
from mrpro.operators._Operator import Operator

class Functional(Operator[tuple[torch.Tensor], tuple[torch.Tensor]]):
    
    def __init__(self, lam=1):
        super().__init__()
        self.lam = lam
        
    @abstractmethod
    def prox(self, *args:torch.Tensor) -> torch.Tensor:
        """Apply forward operator."""
        ...
        
    @abstractmethod
    def prox_convex_conj(self, *args:torch.Tensor) -> torch.Tensor:
        """Apply forward operator."""
        ...