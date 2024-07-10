# %% [markdown]
# # T1 mapping from a continuous golden radial acquisition

# %%
# Imports
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import zenodo_get
from mrpro.data import CsmData, DcfData, IData, KData
from mrpro.data.traj_calculators import KTrajectoryIsmrmrd
from mrpro.operators import FourierOp, SensitivityOp

# %%
# Download raw data in ISMRMRD format from zenodo into a temporary directory
data_folder = Path(tempfile.mkdtemp())
dataset = '10671597'
zenodo_get.zenodo_get([dataset, '-r', 5, '-o', data_folder])  # r: retries

# %% [markdown]
# ## Image reconstruction
# Image reconstruction involves the following steps:
# - Reading in the raw data and the trajectory from the ismrmrd raw data file
# - Calculating the density compensation function (dcf)
# - Reconstructing one image averaging over the entire relaxation period

# %%
# Read raw data and trajectory
kdata = KData.from_file(data_folder / '2D_GRad_map_t1.h5', KTrajectoryIsmrmrd())

# Calculate dcf
dcf = DcfData.from_traj_voronoi(kdata.traj)

# Reconstruct average image for coil map estimation
fourier_op = FourierOp(
    recon_matrix=kdata.header.recon_matrix, encoding_matrix=kdata.header.encoding_matrix, traj=kdata.traj
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
# Clean-up by removing temporary directory
shutil.rmtree(data_folder)

# %% [markdown]
# Copyright 2024 Physikalisch-Technische Bundesanstalt
# Apache License 2.0. See LICENSE file for details.
