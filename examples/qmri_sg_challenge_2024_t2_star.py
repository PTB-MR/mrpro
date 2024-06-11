# %% [markdown]
# # QMRI Challenge ISMRM 2024 - T2* mapping

# %%
# Imports
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import zenodo_get
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore [import-untyped]
from mrpro.algorithms.optimizers import adam
from mrpro.data import IData
from mrpro.operators.functionals import MSEDataDiscrepancy
from mrpro.operators.models import MonoExponentialDecay

# %% [markdown]
# ### Overview
# The dataset consists of gradient echo images obtained at 11 different echo times each saved in a separate dicom file.
# In order to obtain a T2* map we are going to:
# - download the data from zenodo
# - read in the dicom files (one for each echo time) and combine them in a IData object
# - define a signal model (mon-exponential decay) and data loss (mean-squared error) function
# - carry out a fit using ADAM from pytorch
#
# Everything is based on pytorch and therefore we can run the code either on the CPU or GPU. Simply set the flag below
# to True, to run the parameter estimation on the GPU.

# %%
flag_use_cuda = False


# %% [markdown]
# ### Get data from zenodo

# %%
data_folder = Path(tempfile.mkdtemp())
dataset = '10868361'
zenodo_get.zenodo_get([dataset, '-r', 5, '-o', data_folder])  # r: retries
with zipfile.ZipFile(data_folder / Path('T2star.zip'), 'r') as zip_ref:
    zip_ref.extractall(data_folder)

# %% [markdown]
# ### Create IData object with different echo times
# %%
te_dicom_files = data_folder.glob('**/*.dcm')
idata_multi_te = IData.from_dicom_files(te_dicom_files)

# Move the data to the GPU
if flag_use_cuda:
    idata_multi_te = idata_multi_te.cuda()

if idata_multi_te.header.te is None:
    raise ValueError('Echo times need to be defined in the dicom files.')

# %%
# Let's have a look at some of the images
fig, axes = plt.subplots(1, 3)
for idx, ax in enumerate(axes.flatten()):
    ax.imshow(torch.abs(idata_multi_te.data[idx, 0, 0, :, :]).cpu())
    ax.set_title(f'TE = {idata_multi_te.header.te[idx]:.0f}ms')


# %% [markdown]
# ### Signal model and loss function
# We use the model $q$
#
# $q(TE) = M_0 e^{-TE/T2^*}$
#
# with the euqilibrium magnetisation $M_0$, the echo time $TE$ and $T2^*$

# %%
model = MonoExponentialDecay(decay_time=idata_multi_te.header.te)


# %% [markdown]
# As a loss function for the optimizer we calculate the mean-squared error between the image data $x$ and our signal
# model $q$.
# %%
mse = MSEDataDiscrepancy(idata_multi_te.data)

# %% [markdown]
# No we can simply combine the two into a functional which will then solve
#
# $ \min_{M_0, T2^*} ||q(M_0, T2^*, TE) - x||_2^2$
# %%
functional = mse @ model


# %% [markdown]
# ### Carry out fit

# %%
# The shortest echo time is a good approximation of the equilibrium magnetisation
m0_start = torch.abs(idata_multi_te.data[torch.argmin(idata_multi_te.header.te), ...])
# 20 ms as a staring value for T2*
t2star_start = torch.ones(m0_start.shape, dtype=torch.float32, device=m0_start.device) * 20

# Hyperparameters for optimizer
max_iter = 20000
lr = 1e0

if flag_use_cuda:
    functional.cuda()

# Run optimisation
start_time = time.time()
params_result = adam(functional, [m0_start, t2star_start], max_iter=max_iter, lr=lr)
print(f'Optimisation took {time.time()-start_time}s')
m0, t2star = (p.detach() for p in params_result)
m0[torch.isnan(t2star)] = 0
t2star[torch.isnan(t2star)] = 0


# %% [markdown]
# ### Visualise the final results
# To get an impression of how well the fit as worked, we are going to calculate the relative error between
#
# $E_{relative} = \sum_{TE}\frac{|(q(M_0, T2^*, TE) - x)|}{|x|}$
#
# on a voxel-by-voxel basis.
# %%
img_mult_te_abs_sum = torch.sum(torch.abs(idata_multi_te.data), dim=0)
relative_absolute_error = torch.sum(torch.abs(model(m0, t2star)[0] - idata_multi_te.data), dim=0) / (
    img_mult_te_abs_sum + 1e-9
)
fig, axes = plt.subplots(1, 3, figsize=(10, 2))
colorbar_ax = [make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05) for ax in axes]

im = axes[0].imshow(m0[0, 0, ...].cpu())
axes[0].set_title('M0')
fig.colorbar(im, cax=colorbar_ax[0])

im = axes[1].imshow(t2star[0, 0, ...].cpu(), vmin=0, vmax=500)
axes[1].set_title('T2*')
fig.colorbar(im, cax=colorbar_ax[1])

im = axes[2].imshow(relative_absolute_error[0, 0, ...].cpu(), vmin=0, vmax=0.1)
axes[2].set_title('Relative error')
fig.colorbar(im, cax=colorbar_ax[2])

# %%
# Clean-up by removing temporary directory
shutil.rmtree(data_folder)

# %% [markdown]
# Copyright 2024 Physikalisch-Technische Bundesanstalt
# Apache License 2.0. See LICENSE file for details.
