

import torch
import torch.nn.functional as F  # noqa: N812

from mrpro.operators._Functional import ProximableFunctional

class L2NormSquared(ProximableFunctional):
    
    def __init__(self, lam:float=1.0, g:torch.Tensor=torch.tensor([0])):
        super().__init__(lam=lam)
        self.g = g

    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor]:
        return (x.inner(x),) # does this work for complex? or should it be vdot
    
    def convex_conj(self):
        return 1/4 * L2NormSquared()
    
    def prox(self, x:torch.Tensor, sigma:torch.Tensor) -> tuple[torch.Tensor]:
        is_complex = x.is_complex()
        if is_complex:
            x = torch.view_as_real(x)
            self.lam = torch.tensor([self.lam]).unsqueeze(-1)
            sigma = torch.tensor([sigma]).unsqueeze(-1)
            self.g = torch.tensor([self.g]).unsqueeze(-1)
        x = (x+sigma*self.g)/(1+2*self.lam*sigma)
        if is_complex:
            x = torch.view_as_complex(x)
        return (x,)
                
    def prox_convex_conj(self, x:torch.Tensor, sigma:torch.Tensor) -> tuple[torch.Tensor]:
        is_complex = x.is_complex()
        if is_complex:
            x = torch.view_as_real(x)
            self.lam = torch.tensor([self.lam]).unsqueeze(-1)
            sigma = torch.tensor([sigma]).unsqueeze(-1)
            self.g = torch.tensor([self.g]).unsqueeze(-1)
        x = (x-sigma*self.g)/(1+0.5 / self.lam*sigma)
        if is_complex:
            x = torch.view_as_complex(x)
        return (x,)
                