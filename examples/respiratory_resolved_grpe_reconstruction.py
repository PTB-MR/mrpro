# %% [markdown]
# # Reconstruction of respiratory resolved images from a GRPE acquisition

# %%
# Imports
import matplotlib.pyplot as plt
import torch
from einops import rearrange
from mrpro.data import CsmData
from mrpro.data import KTrajectory
from mrpro.data import DcfData
from mrpro.data import IData
from mrpro.data import KData
from mrpro.data.traj_calculators import KTrajectoryRpe
from mrpro.data.traj_calculators import KTrajectoryPulseq
from mrpro.operators import FourierOp
from mrpro.operators import FastFourierOp
from mrpro.operators import SensitivityOp
from mrpro.operators import CartesianSamplingOp
from mrpro.operators import EinsumOp
from mrpro.algorithms.optimizers import cg
from mrpro.utils import split_idx

# %%
fseq = '/echo/kolbit01/data/GRPE_Charite/2024_05_30/20240520_fov288_288_200mm_192_192_640_3d_Charite_grpe_centric_itl_pf0.6.seq'
fname = '/echo/kolbit01/data/GRPE_Charite/2024_05_30/meas_MID00291_FID96729_20240520_fov288_288_200mm_192_192_640_3D_saggital.mrd'
# %% [markdown]
# ### Motion-averaged image reconstruction
# We are going to reconstruct an image using all the acquired data.

#%%
if False:
    import ismrmrd
    
    # Get info and acquisitons from original data
    with ismrmrd.File(fname, 'r') as file:
        ds = file[list(file.keys())[-1]]
        ismrmrd_header = ds.header
        acquisitions = ds.acquisitions[:]

    # Create new file
    fname_out = fname.replace('.mrd', '_pf.h5')
    ds = ismrmrd.Dataset(fname_out)
    ismrmrd_header.encoding[0].encodingLimits.kspace_encoding_step_1.maximum=111
    ismrmrd_header.encoding[0].encodingLimits.kspace_encoding_step_1.center=96
    ds.write_xml_header(ismrmrd_header.toXML())
    
    k1 = []
    k2 = []
    for acq in acquisitions:
        k1.append(acq.idx.kspace_encode_step_1)
        k2.append(acq.idx.kspace_encode_step_2)
        ds.append_acquisition(acq)
    ds.close()
    
    plt.figure()
    plt.plot(k1)
    plt.plot(k2)

# %%
# Read the raw data, calculate the trajectory and dcf
kdata = KData.from_file(fname.replace('.mrd', '_pf.h5'), KTrajectoryRpe(angle=torch.pi * 0.618034))

plt.figure()
plt.plot(kdata.traj.ky[0,:10,:,0], kdata.traj.kz[0,:10,:,0], 'ob')

kdcf = DcfData.from_traj_voronoi(kdata.traj)
 
# Set the matrix sizes which are not encoded correctly in the pulseq sequence
kdata.header.encoding_matrix.z = kdata.header.encoding_matrix.y
kdata.header.recon_matrix.x = 192
kdata.header.recon_matrix.y = 192
kdata.header.recon_matrix.z = 192

# Direct image reconstruction of individual coil images
fourier_op = FourierOp(kdata.header.recon_matrix, kdata.header.encoding_matrix, kdata.traj)
(img,) = fourier_op.adjoint(kdata.data * kdcf.data[:,None,...])

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
    ax[ind].imshow(rearrange(torch.abs(nav_data[:,ind,:]), 'angle x->x angle'))
    
# Carry out SVD along coil and readout dimension
nav_data = rearrange(nav_data, 'k1k2 coil k0 -> k1k2 (coil k0)')
u, _, _ = torch.linalg.svd(nav_data - nav_data.mean())

resp_nav = u[:,0]
resp_nav[0] = resp_nav[3]
resp_nav[1] = resp_nav[3]
resp_nav[2] = resp_nav[3]
plt.figure() 
plt.plot(resp_nav, ':k')   

# %% [markdown]
# ### Motion-resolved image reconstruction
# Now we can reconstruct the respirator motion-resolved images. 
# Here we have to deal with undersampled data and so we use an iterative reconstruction (iterative SENSE).

resp_idx = split_idx(torch.argsort(resp_nav), 120, 20)
kdata_resp_resolved = kdata.split_k2_into_other(resp_idx, other_label='repetition')
kdcf = DcfData.from_traj_voronoi(kdata_resp_resolved.traj)

cart_sampling_op = CartesianSamplingOp(kdata_resp_resolved.header.encoding_matrix, kdata_resp_resolved.traj)
fourier_op = FourierOp(kdata_resp_resolved.header.recon_matrix, kdata_resp_resolved.header.encoding_matrix, kdata_resp_resolved.traj)
A = cart_sampling_op @ fourier_op @ csm_op

#AW = kdcf.data[:,None,...] * cart_sampling_op @ fourier_op @ csm_op
#x0 = AW.H(kdata_resp_resolved.data)[0]

# Direkt reconstruction for comparison
x0 = A.H(kdata_resp_resolved.data * kdcf.data[:,None,...])[0]

plot_resp_idx = [0, x0.shape[0]-1]
fig, ax = plt.subplots(len(plot_resp_idx),3)
for nnd in range(len(plot_resp_idx)):
    ax[nnd,0].imshow(torch.abs(x0[plot_resp_idx[nnd],0,img.shape[-3]//2,:,:]))
    ax[nnd,1].imshow(torch.abs(x0[plot_resp_idx[nnd],0,:,img.shape[-2]//2,:]))
    ax[nnd,2].imshow(torch.abs(x0[plot_resp_idx[nnd],0,:,:,img.shape[-1]//2]))

# Iterative reconstruction
H = A.H @ A
N = 2
b = A.H(kdata_resp_resolved.data)[0]
with torch.no_grad():
    img_resp_resolved = cg(H, b, x0, N)

fig, ax = plt.subplots(len(plot_resp_idx),3)
for nnd in range(len(plot_resp_idx)):
    ax[nnd,0].imshow(torch.abs(img_resp_resolved[plot_resp_idx[nnd],0,img.shape[-3]//2,:,:]))
    ax[nnd,1].imshow(torch.abs(img_resp_resolved[plot_resp_idx[nnd],0,:,img.shape[-2]//2,:]))
    ax[nnd,2].imshow(torch.abs(img_resp_resolved[plot_resp_idx[nnd],0,:,:,img.shape[-1]//2]))


# %% [markdown]
# Copyright 2024 Physikalisch-Technische Bundesanstalt
# Apache License 2.0. See LICENSE file for details.
