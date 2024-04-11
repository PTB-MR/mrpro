# %% [markdown]
# # Reconstruction of respiratory resolved images from a GRPE acquisition

# %%
# Imports
import matplotlib.pyplot as plt
import torch
from einops import rearrange
from mrpro.data import CsmData
from mrpro.data import DcfData
from mrpro.data import IData
from mrpro.data import KData
from mrpro.data.traj_calculators import KTrajectoryRpe
from mrpro.operators import FourierOp
from mrpro.operators import FastFourierOp
from mrpro.operators import SensitivityOp
from mrpro.operators import CartesianSamplingOp
from mrpro.operators import EinsumOp
from mrpro.algorithms.optimizers import cg
from mrpro.utils import split_idx

# %%
fname = '/echo/_allgemein/projects/Christoph/TSE_T2/2024_04_10_Charite/meas_MID00022_FID84486_pulseq.mrd'

# %% [markdown]
# ### Motion-averaged image reconstruction
# We are going to reconstruct an image using all the acquired data.

# %%
# Read the raw data, calculate the trajectory and dcf
kdata = KData.from_file(fname, KTrajectoryRpe(angle=torch.pi * 0.618034))
kdcf = DcfData.from_traj_voronoi(kdata.traj)
 
# Set the matrix sizes which are not encoded correctly in the pulseq sequence
kdata.header.encoding_matrix.z = kdata.header.encoding_matrix.y
kdata.header.recon_matrix.x = 192
kdata.header.recon_matrix.y = 192
kdata.header.recon_matrix.z = 192

# Direct image reconstruction of individual coil images
cart_sampling_op = CartesianSamplingOp(kdata.header.encoding_matrix, kdata.traj)
fourier_op = FourierOp(kdata.header.recon_matrix, kdata.header.encoding_matrix, kdata.traj)
(img,) = fourier_op.adjoint(cart_sampling_op.adjoint(kdata.data * kdcf.data[:,None,...])[0])

# Calculate coilmaps
idata = IData.from_tensor_and_kheader(img, kdata.header)
csm_data = CsmData.from_idata_walsh(idata)
csm_op = SensitivityOp(csm_data)

# Get coil-combined image
(img,) = csm_op.adjoint(img)

# Visualise 
fig, ax = plt.subplots(1,3)
ax[0].imshow(torch.abs(img[0,0,img.shape[-3]//2,:,:]))
ax[1].imshow(torch.abs(img[0,0,:,img.shape[-2]//2,:]))
ax[2].imshow(torch.abs(img[0,0,:,:,img.shape[-1]//2]))

# %% [markdown]
# ### Respiratory self-navigator
# We are going to obtain a self-navigator from the k-space centre line ky = kz = 0

# Find readouts at ky=kz=0
fft_op_1d = FastFourierOp(dim=(-1,))
ky0_kz0_idx = torch.where(((kdata.traj.ky == 0) & (kdata.traj.kz == 0)))
nav_data = kdata.data[ky0_kz0_idx[0],:,ky0_kz0_idx[1],ky0_kz0_idx[2],:]
nav_data = torch.abs(fft_op_1d(nav_data)[0])
nav_signal_time_stamp = kdata.header.acq_info.acquisition_time_stamp[ky0_kz0_idx[0],ky0_kz0_idx[1],ky0_kz0_idx[2]]

fig, ax = plt.subplots(5,1)
for ind in range(5):
    ax[ind].imshow(torch.abs(nav_data[:,ind,:]))
    
# Carry out SVD along coil and readout dimension
nav_data = rearrange(nav_data, 'k1k2 coil k0 -> k1k2 (coil k0)')
u, _, _ = torch.linalg.svd(nav_data - nav_data.mean())

resp_nav = u[:,0]
resp_nav[0] = resp_nav[1]

plt.figure() 
plt.plot(resp_nav, ':k')   

# %% [markdown]
# ### Motion-resolved image reconstruction
# Now we can reconstruct the respirator motion-resolved images. 
# Here we have to deal with undersampled data and so we use an iterative reconstruction (iterative SENSE).

resp_idx = split_idx(torch.argsort(resp_nav), 56, 16)
kdata_resp_resolved = kdata.split_k2_into_other(resp_idx, other_label='repetition')
kdcf = DcfData.from_traj_voronoi(kdata_resp_resolved.traj)

cart_sampling_op = CartesianSamplingOp(kdata_resp_resolved.header.encoding_matrix, kdata_resp_resolved.traj)
fourier_op = FourierOp(kdata_resp_resolved.header.recon_matrix, kdata_resp_resolved.header.encoding_matrix, kdata_resp_resolved.traj)
A = cart_sampling_op @ fourier_op @ csm_op
AH = A.H
#W = EinsumOp((kdcf.data[:,None,...] / kdcf.data.shape[1]).to(torch.complex64), '...ij,...j->...j')
#AHW = AH @ W
#H = (AH @ W) @ A
#x0 = AHW(kdata_resp_resolved.data)[0]
#b = x0
H = AH @ A

# Direkt reconstruction for comparison
x0 = AH(kdata_resp_resolved.data * kdcf.data[:,None,...])[0]

plot_resp_idx = [0, x0.shape[0]-1]
fig, ax = plt.subplots(len(plot_resp_idx),3)
for nnd in range(len(plot_resp_idx)):
    ax[nnd,0].imshow(torch.abs(x0[plot_resp_idx[nnd],0,img.shape[-3]//2,:,:]))
    ax[nnd,1].imshow(torch.abs(x0[plot_resp_idx[nnd],0,:,img.shape[-2]//2,:]))
    ax[nnd,2].imshow(torch.abs(x0[plot_resp_idx[nnd],0,:,:,img.shape[-1]//2]))


# Iterative reconstruction
N = 10
b = AH(kdata_resp_resolved.data)[0]
with torch.no_grad():
    img_resp_resolved = cg(H, b, x0, N)

fig, ax = plt.subplots(len(plot_resp_idx),3)
for nnd in range(len(plot_resp_idx)):
    ax[nnd,0].imshow(torch.abs(img_resp_resolved[plot_resp_idx[nnd],0,img.shape[-3]//2,:,:]))
    ax[nnd,1].imshow(torch.abs(img_resp_resolved[plot_resp_idx[nnd],0,:,img.shape[-2]//2,:]))
    ax[nnd,2].imshow(torch.abs(img_resp_resolved[plot_resp_idx[nnd],0,:,:,img.shape[-1]//2]))

# We can also use an iterative reconstruction

# %% [markdown]
# Copyright 2024 Physikalisch-Technische Bundesanstalt
# Apache License 2.0. See LICENSE file for details.
