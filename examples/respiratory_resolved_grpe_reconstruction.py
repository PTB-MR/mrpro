# %%
# # Reconstruction of respiratory resolved images from a GRPE acquisition
# ### File name
fname = r'/sc-projects/sc-proj-cc06-agsack/noja11/ImageReconstruction/Data/meas_MID00291_FID96729_20240520_fov288_288_200mm_192_192_640_3D_saggital.mrd'
path_out = r'/sc-projects/sc-proj-cc06-agsack/noja11/ImageReconstruction/'
fname = '/echo/kolbit01/data/GRPE_Charite/2024_05_30/meas_MID00291_FID96729_20240520_fov288_288_200mm_192_192_640_3D_saggital.mrd'
path_out = '/echo/kolbit01/data/GRPE_Charite/2024_05_30/'
# ### Imports
import matplotlib.pyplot as plt
import torch
from einops import rearrange
from mrpro.data import CsmData, KTrajectory, DcfData, IData, KData
from mrpro.data.traj_calculators import KTrajectoryRpe, KTrajectoryPulseq
from mrpro.operators import FourierOp, FastFourierOp, SensitivityOp, CartesianSamplingOp, EinsumOp
from mrpro.algorithms.optimizers import cg
from mrpro.utils import split_idx
import os
import nibabel as nib


# Correct for partial fourier sampling, as Siemens scanner does not save trajectory correctly
if not os.path.exists(fname.replace('.mrd', '_pf.h5')):
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

# ### Motion-averaged image reconstruction
# We are going to reconstruct an image using all the acquired data.
# Read the raw data, calculate the trajectory and dcf
kdata = KData.from_file(fname.replace('.mrd', '_pf.h5'), KTrajectoryRpe(angle=torch.pi * 0.618034))

kdcf = DcfData.from_traj_voronoi(kdata.traj)
 
# Set the matrix sizes which are not encoded correctly in the pulseq sequence
kdata.header.encoding_matrix.z = kdata.header.encoding_matrix.y
kdata.header.recon_matrix.x = 192
kdata.header.recon_matrix.y = 192
kdata.header.recon_matrix.z = 192

# Direct image reconstruction of individual coil images
fourier_op = FourierOp(kdata.header.recon_matrix, kdata.header.encoding_matrix, kdata.traj)
(img,) = fourier_op.adjoint(kdata.data * kdcf.data[:,None,...])

plt.figure()
plt.plot(kdata.traj.ky[0,:10,:,0], kdata.traj.kz[0,:10,:,0], 'ob')

# Calculate coilmaps and get coil combined image
idata = IData.from_tensor_and_kheader(img, kdata.header)
csm_data = CsmData.from_idata_walsh(idata)
csm_op = SensitivityOp(csm_data)
(img,) = csm_op.adjoint(img)

# Visualize the image
img = torch.abs(torch.sqrt(torch.sum(img**2, dim=1, keepdim=True)))

fig, ax = plt.subplots(1,3);
ax[0].imshow(torch.abs(img[0,0,img.shape[-3]//2,:,:]))
ax[1].imshow(torch.abs(img[0,0,:,img.shape[-2]//2,:]))
ax[2].imshow(torch.abs(img[0,0,:,:,img.shape[-1]//2]))

# ### Respiratory self-navigator
# We are going to obtain a self-navigator from the k-space centre line ky = kz = 0
# Find readouts at ky=kz=0
fft_op_1d = FastFourierOp(dim=(-1,))
ky0_kz0_idx = torch.where(((kdata.traj.ky == 0) & (kdata.traj.kz == 0)))
nav_data = kdata.data[ky0_kz0_idx[0],:,ky0_kz0_idx[1],ky0_kz0_idx[2],:]
nav_data = torch.abs(fft_op_1d(nav_data)[0])
nav_signal_time_stamp = kdata.header.acq_info.acquisition_time_stamp[ky0_kz0_idx[0],ky0_kz0_idx[1],ky0_kz0_idx[2]]

fig, ax = plt.subplots(3,1);
for ind in range(3):
    ax[ind].imshow(rearrange(torch.abs(nav_data[:,ind,:]), 'angle x->x angle'),aspect='auto');
    ax[ind].set_xlim([0, 200]);

# Carry out SVD along coil and readout dimension
nav_data_pre_rearrange = nav_data
nav_data = rearrange(nav_data, 'k1k2 coil k0 -> k1k2 (coil k0)')
u, _, _ = torch.linalg.svd(nav_data - nav_data.mean(dim=0, keepdim=True))

# Find SVD component that is closest to expected respitory frequency
t_sec = 2.5*kdata.header.acq_info.acquisition_time_stamp[ky0_kz0_idx[0],ky0_kz0_idx[1],ky0_kz0_idx[2],:]/1000
resp_freq_window_Hz = [0.2, 0.3]
dt = torch.mean(torch.diff(t_sec, dim=0))
U_freq = torch.abs(torch.fft.fft(u, dim=0)) 

fmax_Hz = 1/dt 
f_Hz = torch.linspace(0, fmax_Hz, u.shape[0]) 

condition_below = f_Hz <=resp_freq_window_Hz[1] 
condition_above = f_Hz >= resp_freq_window_Hz[0] 
condition_window = condition_below * condition_above 

U_freq_window = U_freq[condition_window, :] 
peak_in_window = torch.max(U_freq_window, dim=0) 
peak_component = torch.argmax(peak_in_window.values) 

resp_nav = u[:,peak_component] 

plt.figure() 
plt.plot(resp_nav, ':k')   
plt.xlim(0,200)

rescaled_resp_nav = 80 * (resp_nav - resp_nav.min()) / (resp_nav.max() - resp_nav.min()) +80

plt.figure()
plt.imshow(rearrange(torch.abs(nav_data_pre_rearrange[:,0,:]), 'angle x->x angle'), aspect='auto')
plt.plot(rescaled_resp_nav, ':w')
plt.xlim(0,200)

# Non iterative reconstruction
resp_idx = split_idx(torch.argsort(resp_nav), 160, 20)
kdata_resp_resolved = kdata.split_k2_into_other(resp_idx, other_label='repetition')
kdcf = DcfData.from_traj_voronoi(kdata_resp_resolved.traj)

cart_sampling_op = CartesianSamplingOp(kdata_resp_resolved.header.encoding_matrix, kdata_resp_resolved.traj)
fourier_op = FourierOp(kdata_resp_resolved.header.recon_matrix, kdata_resp_resolved.header.encoding_matrix, kdata_resp_resolved.traj)
(img,) = fourier_op.adjoint(kdata_resp_resolved.data * kdcf.data[:,None,...])
img = torch.abs(torch.sqrt(torch.sum(img**2, dim=1, keepdim=True)))

plot_resp_idx = [0, img.shape[0]-1]
fig, ax = plt.subplots(len(plot_resp_idx),3)
for nnd in range(len(plot_resp_idx)):
    ax[nnd,0].imshow(torch.abs(img[plot_resp_idx[nnd],0,50,:,:]))
    ax[nnd,1].imshow(torch.abs(img[plot_resp_idx[nnd],0,:,75,:]))
    ax[nnd,2].imshow(torch.abs(img[plot_resp_idx[nnd],0,:,:,110]))

for i in range(img.shape[0]):
    nii_resp = nib.Nifti1Image(img[i,0].numpy(), affine=torch.eye(4))
    nib.save(nii_resp,os.path.join(path_out,"resp_state_"+str(i)))


# ### Motion-resolved image reconstruction
# Now we can reconstruct the respirator motion-resolved images. 
# Here we have to deal with undersampled data and so we use an iterative reconstruction (iterative SENSE).

resp_idx = split_idx(torch.argsort(resp_nav), 160, 20)
kdata_resp_resolved = kdata.split_k2_into_other(resp_idx, other_label='repetition')
kdcf = DcfData.from_traj_voronoi(kdata_resp_resolved.traj)


#%%
cart_sampling_op = CartesianSamplingOp(kdata_resp_resolved.header.encoding_matrix, kdata_resp_resolved.traj)
fourier_op = FourierOp(kdata_resp_resolved.header.recon_matrix, kdata_resp_resolved.header.encoding_matrix, kdata_resp_resolved.traj)

# Create A^H W
AH_W = csm_op.H @ fourier_op.H @ cart_sampling_op.H @ kdcf.as_operator()

# Calculate x0 = A^H W y
(x0,) = AH_W(kdata_resp_resolved.data)

plot_resp_idx = [0, x0.shape[0]-1]
fig, ax = plt.subplots(len(plot_resp_idx),3)
for nnd in range(len(plot_resp_idx)):
    ax[nnd,0].imshow(torch.abs(x0[plot_resp_idx[nnd],0,img.shape[-3]//2,:,:]))
    ax[nnd,1].imshow(torch.abs(x0[plot_resp_idx[nnd],0,:,img.shape[-2]//2,:]))
    ax[nnd,2].imshow(torch.abs(x0[plot_resp_idx[nnd],0,:,:,img.shape[-1]//2]))

# Create H = A^H W A
H = AH_W @ cart_sampling_op @ fourier_op @ csm_op

# Iterative reconstruction
number_of_iterations = 8
with torch.no_grad():
    img_resp_resolved = cg(H, x0, x0, number_of_iterations)

fig, ax = plt.subplots(len(plot_resp_idx),3)
for nnd in range(len(plot_resp_idx)):
    ax[nnd,0].imshow(torch.abs(img_resp_resolved[plot_resp_idx[nnd],0,img.shape[-3]//2,:,:]))
    ax[nnd,1].imshow(torch.abs(img_resp_resolved[plot_resp_idx[nnd],0,:,img.shape[-2]//2,:]))
    ax[nnd,2].imshow(torch.abs(img_resp_resolved[plot_resp_idx[nnd],0,:,:,img.shape[-1]//2]))

for i in range(img_resp_resolved.shape[0]):
    nii_resp = nib.Nifti1Image(img_resp_resolved[i,0].numpy(), affine=torch.eye(4))
    nib.save(nii_resp,os.path.join(path_out,"resp_state_iterative_"+str(i)))
