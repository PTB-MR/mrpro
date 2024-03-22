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
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from mrpro.algorithms._cg import cg
from mrpro.data import CsmData
from mrpro.data import DcfData
from mrpro.data import IData
from mrpro.data import KData
from mrpro.data.traj_calculators import KTrajectoryIsmrmrd
from mrpro.operators import EinsumOp
from mrpro.operators import FourierOp
from mrpro.operators import SensitivityOp

# %%
# Download raw data in ISMRMRD format from zenodo into a temporary directory
data_folder = Path(
    '/echo/_allgemein/projects/8_13/MRPro/2024_03_19/20240319_spiral_2D_256mm_252k0_256interleaves_golden_angle'
)
dataset_traj = 'spiral_2D_256mm_252k0_256interleaves_golden_angle_with_traj.h5'
seq_folder = Path(
    '/echo/_allgemein/projects/8_13/MRPro/2024_03_19/20240319_spiral_2D_256mm_252k0_256interleaves_golden_angle/'
)
seqfile = 'spiral_2D_256mm_252k0_256interleaves_golden_angle.seq'

# %% [markdown]
# ## Image reconstruction
# Image reconstruction involves the following steps:
# - Reading in the raw data and the trajectory from the ismrmrd raw data file
# - Calculating the density compensation function (dcf)
# - Reconstructing one image averaging over the entire relaxation period

# %%
# Read raw data and trajectory
kdata = KData.from_file(filename=data_folder / dataset_traj, ktrajectory=KTrajectoryIsmrmrd())
# Calculate dcf
dcf = DcfData.from_traj_voronoi(kdata.traj)

# Reconstruct average image for coil map estimation
fourier_op = FourierOp(
    recon_matrix=kdata.header.recon_matrix,
    encoding_matrix=kdata.header.encoding_matrix,
    traj=kdata.traj,
)
(img,) = fourier_op.adjoint(kdata.data * dcf.data[:, None, ...])

# %%
# Calculate coilmaps
idata = IData.from_tensor_and_kheader(img, kdata.header)
csm = CsmData.from_idata_walsh(idata)
csm_op = SensitivityOp(csm)

# %%
# Coil combination
(img,) = csm_op.adjoint(img)

# %%
# Visualize results
plt.figure()
plt.imshow(torch.abs(img[0, 0, 0, :, :]))

# %%
# Construct operators
A = fourier_op @ csm_op
AH = A.H
W = EinsumOp((dcf.data / dcf.data.shape[2]).to(torch.complex64), '...ij,...j->...j')
AHW = AH @ W
H = (AH @ W) @ A

# %%
# Visualize results
x0 = AHW(kdata.data)[0]
plt.imshow(torch.abs(x0[0, 0, 0, :, :]))
plt.colorbar()
plt.title('x0')
plt.show()

# %%
# Apply Conjugate Gradient with N steps and visualize results
N = 2
b = x0
with torch.no_grad():
    x_cg = cg(H, b, x0, N)

plt.imshow(torch.abs(x_cg[0, 0, 0, :, :]))
plt.colorbar()
plt.title('xCG')
plt.show()
