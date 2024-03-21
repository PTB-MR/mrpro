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
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import zenodo_get
from mrpro.data import CsmData
from mrpro.data import DcfData
from mrpro.data import IData
from mrpro.data import KData
from mrpro.data.traj_calculators import KTrajectoryIsmrmrd
from mrpro.data.traj_calculators._KTrajectoryPulseq import KTrajectoryPulseq
from mrpro.operators import FourierOp
from mrpro.operators import SensitivityOp
import torch.nn as nn
from mrpro.algorithms._cg import cg
from mr_ops import EncObj_Reco

# %%
# Download raw data in ISMRMRD format from zenodo into a temporary directory
data_folder = Path("/echo/_allgemein/projects/8_13/MRPro/2024_03_19/20240319_spiral_2D_256mm_252k0_256interleaves_golden_angle")#Path(tempfile.mkdtemp())
dataset_traj = 'spiral_2D_256mm_252k0_256interleaves_golden_angle_with_traj.h5'
seq_folder = Path("/echo/_allgemein/projects/8_13/MRPro/2024_03_19/20240319_spiral_2D_256mm_252k0_256interleaves_golden_angle/")
seqfile = "spiral_2D_256mm_252k0_256interleaves_golden_angle.seq"
# seqheader = "20240319_spiral_2D_256mm_252k0_256interleaves_golden_angle_header.h5"
# dataset = 'spiral_2D_256mm_252k0_256interleaves_golden_angle.h5'
#zenodo_get.zenodo_get([dataset, '-r', 5, '-o', data_folder])  # r: retries

# %% [markdown]
# ## Image reconstruction
# Image reconstruction involves the following steps:
# - Reading in the raw data and the trajectory from the ismrmrd raw data file
# - Calculating the density compensation function (dcf)
# - Reconstructing one image averaging over the entire relaxation period

# %%
# Read raw data and trajectory
kdata = KData.from_file(filename=data_folder/dataset_traj, ktrajectory=KTrajectoryIsmrmrd())
#kdata = KData.from_file(filename="/echo/martin13/data/Measurements/meas_MID00080_FID02053_20230926_spiral_256px_fov256_8mm_75shots/meas_MID00080_FID02053_20230926_spiral_256px_fov256_8mm_75shots.mrd", ktrajectory=KTrajectoryPulseq("/echo/martin13/data/Measurements/meas_MID00080_FID02053_20230926_spiral_256px_fov256_8mm_75shots/20230926_spiral_256px_fov256_8mm_75shots.seq"))
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

#%%
class EncObj_Reco(nn.Module):

    def __init__(self, kdata, csm):
        super(EncObj_Reco, self).__init__()
    
        self.F = FourierOp(
            recon_matrix=kdata.header.recon_matrix, encoding_matrix=kdata.header.encoding_matrix, traj=kdata.traj
        )
        self.C = SensitivityOp(csm)

    def apply_A(self, x):
        k = (self.F @ self.C)(x) 
        return k[0]

    def apply_AH(self, k):
        x = (self.F @ self.C).adjoint(k)
        return x[0]

    def apply_AHA(self, x):
        k = self.apply_A(x)
        x = self.apply_AH(k)
        return x

    def apply_dcomp(self, k, dcomp):
        return dcomp * k

    def apply_Adag(self, k, dcomp):
        dcomp_k = self.apply_dcomp(k, dcomp)
        x = self.apply_AH(dcomp_k)
        return x

    def apply_AdagA(self, x, dcomp):
        k = self.apply_A(x)
        x = self.apply_Adag(k, dcomp)
        return x

#%%
EncObj = EncObj_Reco(kdata=kdata,csm=csm)

xu = EncObj.apply_Adag(kdata.data,dcf.data)
H = lambda xu: EncObj.apply_AdagA(x=xu,dcomp=dcf.data)
b = xu

with torch.no_grad():
	xCG = cg(H,b,xu,40)

# %%
# Clean-up by removing temporary directory
#shutil.rmtree(data_folder)
