#%%
import torch
from mrpro.operators.functionals.l1 import L1Norm
from mrpro.operators.functionals.l2_squared import L2NormSquared

l1 = L1Norm(lam=1) 
l2_squared = L2NormSquared(lam=1)

test = torch.ones((3,3,3), dtype=torch.complex64)
# %%
print(l1.forward(test))
print(l1.prox(test,sigma=1))
#%%
print(l2_squared.forward(test))
print(l2_squared.prox(test,sigma=0.5))
print(l2_squared.prox_convex_conj(test,sigma=0.5))
# %%
#print(l1.prox(torch.tensor([1,2,3]),sigma=1))
#print(l2_squared.prox(torch.tensor([1,2,3]),sigma=1))
l2_squared = L2NormSquared(lam=1, dim=[0,1])
print(l2_squared.prox(torch.real(test),sigma=1))
print(l2_squared.prox(torch.imag(test),sigma=1))
print(l2_squared.prox(test,sigma=1))
#%%
print(l2_squared.prox_convex_conj(torch.real(test),sigma=1))
print(l2_squared.prox_convex_conj(torch.imag(test),sigma=1))
print(l2_squared.prox_convex_conj(test,sigma=1))
#%%
print(l1.prox_convex_conj(torch.tensor([1,2,3]),sigma=1))
print(l2_squared.prox_convex_conj(torch.tensor([1,2,3]),sigma=1))
#%%

ex_tensor = torch.rand([3,4,126,126])
# %%
