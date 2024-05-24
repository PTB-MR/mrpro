import torch
import torch.nn.functional as F  # noqa: N812

from mrpro.operators._Functional import ProximableFunctional

class L2NormSquared(ProximableFunctional):
    
    def __init__(self, lam:float=1.0, g:torch.Tensor=torch.tensor([0]), dim:list=(None)):
        super().__init__(lam=lam)
        self.g = g
        self.dim = dim

    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor]:
        return (torch.pow(torch.linalg.norm(x.flatten(), ord=2, dim=self.dim, keepdim=True),2),)
    
    def convex_conj(self):
        return 1/4 * L2NormSquared()
    
    def prox(self, x:torch.Tensor, sigma:torch.Tensor) -> tuple[torch.Tensor]:
        x_out = ((x+sigma*self.g)/(1+2*self.lam*sigma))
        return (x_out,)
                
    def prox_convex_conj(self, x:torch.Tensor, sigma:torch.Tensor) -> tuple[torch.Tensor]:
        x_out = (x-sigma*self.g)/(1+0.5 / self.lam*sigma)
        return (x_out,)