import torch
import torch.nn.functional as F  # noqa: N812

from mrpro.operators._Functional import ProximableFunctional

class L1Norm(ProximableFunctional):
    
    def __init__(self, lam=1, g:torch.Tensor=torch.tensor([0])):
        super().__init__(lam=lam)
        self.g = g

    def forward(self, x):
        return (x.abs().sum(),)
    
    def prox(self, x:torch.Tensor, sigma):
        is_complex = x.is_complex()
        if is_complex:
            x = torch.view_as_real(x)
            threshold = torch.tensor([self.lam*sigma]).unsqueeze(-1)
        else:
            threshold = torch.tensor([self.lam*sigma])
        x = torch.clamp(x, -threshold, threshold)
        if is_complex:
            x = torch.view_as_complex(x)
        return x
        
    def prox_convex_conj(self, x, sigma):
        is_complex = x.is_complex()
        if is_complex:
            x = torch.view_as_real(x)
            self.lam = torch.tensor([self.lam]).unsqueeze(-1)
            sigma = torch.tensor([sigma]).unsqueeze(-1)
            self.g = torch.tensor([self.g]).unsqueeze(-1)
        x = torch.clamp((x-self.g*sigma), -self.lam, self.lam)
        if is_complex:
            x = torch.view_as_complex(x)
        return x
        