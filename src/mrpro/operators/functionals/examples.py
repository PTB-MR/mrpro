#%%
import torch
from mrpro.operators.functionals.l1 import L1Norm
from mrpro.operators.functionals.l2_squared import L2NormSquared

l1 = L1Norm(lam=1) 
l2_squared = L2NormSquared(lam=1)
# %%
print(l1.forward(torch.tensor([1,2,3])))
print(l2_squared.forward(torch.tensor([1,2,3])))
# %%
print(l1.prox(torch.tensor([1,2,3]),sigma=1))
print(l2_squared.prox(torch.tensor([1,2,3]),sigma=1))
#%%
print(l1.prox_convex_conj(torch.tensor([1,2,3]),sigma=1))
print(l2_squared.prox_convex_conj(torch.tensor([1,2,3]),sigma=1))
#%%