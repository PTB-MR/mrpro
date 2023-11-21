# %% [markdown]
# # T1 mapping from a continuous golden radial acquisition

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# %% [markdown]
# If you want to run this notebook in binder you need to still install the MRpro package.
# This only needs to be done once in a binder session. Open a terminal (File -> New -> Terminal) and run:
# ```
# pip install -e ".[notebook]"
# ```
# This will install the MRpro package. Any other required python packages should already be present in this
# docker image.

# %%
# Imports
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# TODO remove once Fourier operator is available
import torchkbnufft as tkbn
from einops import rearrange

from mrpro.data import CsmData
from mrpro.data import DcfData
from mrpro.data import IData
from mrpro.data import KData
from mrpro.data.traj_calculators import KTrajectoryIsmrmrd
from mrpro.operators import SensitivityOp


# %%
# Fourier operator
# TODO remove once Fourier operator is available
class E:
    def __init__(self, im_size, ktraj):
        self.kbnufft_adj = tkbn.KbNufftAdjoint(im_size=im_size)
        self.ktraj = rearrange(ktraj, 'dim other k2 k1 k0->other dim (k2 k1 k0)')

    def bwd(self, y):
        y = rearrange(y, 'other coils k2 k1 k0->other coils (k2 k1 k0)')
        return rearrange(self.kbnufft_adj(y, self.ktraj), 'other (coils z) y x->other coils z y x', z=1)


# %%
# Download raw data in ISMRMRD format from zenodo into a temporary directory
data_folder = Path(tempfile.mkdtemp())
zenodo_cmd = f'zenodo_get 10.5281/zenodo.7903232 -o {str(data_folder)}'
out = subprocess.call(zenodo_cmd, shell=True)

# %% [markdown]
# ## Image reconstruction
# Image reconstruction involves the following steps:
# - Reading in the raw data and the trajectory from the ismrmrd raw data file
# - Calculating the density compensation function (dcf)
# - Reconstructing one image averaging over the entire relaxation period

# %%
# Read raw data and trajectory
kdata = KData.from_file(data_folder / '2D_Dyn_GRad.h5', KTrajectoryIsmrmrd())

# Calculate dcf
kdcf = DcfData.from_traj_voronoi(kdata.traj)

# Reconstruct average image for coil map estimation
im_size = (kdata.header.recon_matrix.y, kdata.header.recon_matrix.x)
ERad = E(im_size, kdata.traj.as_tensor()[1:3, ...])
im = ERad.bwd(kdata.data * kdcf.data[:, None, ...])

# %%
# Calculate coilmaps
idat = IData.from_tensor_and_kheader(im, kdata.header)
csm = CsmData.from_idata_walsh(idat)
csm_op = SensitivityOp(csm)

# %%
# Coil combination
im = csm_op.adjoint(im)

# %%
# Visualize results
plt.figure()
plt.imshow(torch.abs(im[0, 0, 0, :, :]))

# %%
# Clean-up by removing temporary directory
shutil.rmtree(data_folder)
