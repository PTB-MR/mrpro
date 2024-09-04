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
import numpy as np
from einops import rearrange
from mrpro.data import CsmData, KTrajectory, DcfData, IData, KData
from mrpro.data.traj_calculators import KTrajectoryRpe, KTrajectoryPulseq
from mrpro.operators import FourierOp, FastFourierOp, SensitivityOp, CartesianSamplingOp, EinsumOp
from mrpro.algorithms.optimizers import cg
from mrpro.algorithms.reconstruction import DirectReconstruction, IterativeSENSEReconstruction
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
    
    
def get_coronal_sagittal_axial_view(img, other=None, z=None, y=None, x=None):
    other = img.shape[-4]//2 if other is None else other
    z = img.shape[-3]//2 if z is None else z
    y = img.shape[-2]//2 if y is None else y
    x = img.shape[-1]//2 if x is None else x
    
    return np.concatenate((np.rot90(img[other,:,y,:]),
                        np.rot90(img[other,z,:,:]),
                        np.rot90(img[other,:,:,x])), axis=1)
    
    
# Display slices
showz, showy, showx = [50, 96, 115]
    

# %% Read data
# We are going to reconstruct an image using all the acquired data.
# Read the raw data, calculate the trajectory and dcf
kdata = KData.from_file(fname.replace('.mrd', '_pf.h5'), KTrajectoryRpe(angle=torch.pi * 0.618034))

# Set the matrix sizes which are not encoded correctly in the pulseq sequence
kdata.header.encoding_matrix.z = kdata.header.encoding_matrix.y
kdata.header.recon_matrix.x = 192
kdata.header.recon_matrix.y = 192
kdata.header.recon_matrix.z = 220

plt.figure()
plt.plot(kdata.traj.ky[0,:10,:,0], kdata.traj.kz[0,:10,:,0], 'ob')

# %% Iterative reconstruction of average
itSENSE_recon = IterativeSENSEReconstruction(kdata)
img_itSENSE = itSENSE_recon.forward(kdata)

img_coronal_sagittal_axial = get_coronal_sagittal_axial_view(img_itSENSE.rss(), z=showz, y=showy, x=showx)
img_coronal_sagittal_axial /= img_coronal_sagittal_axial.max()
plt.figure()
plt.imshow(img_coronal_sagittal_axial, cmap='grey', vmin=0, vmax=0.35)
plt.imsave(os.path.join(path_out,f'average.png'), img_coronal_sagittal_axial, dpi=300, cmap='grey', vmin=0, vmax=0.35)


# %% Respiratory self-navigator
# We are going to obtain a self-navigator from the k-space centre line ky = kz = 0
# Find readouts at ky=kz=0
fft_op_1d = FastFourierOp(dim=(-1,))
ky0_kz0_idx = torch.where(((kdata.traj.ky == 0) & (kdata.traj.kz == 0)))
nav_data = kdata.data[ky0_kz0_idx[0],:,ky0_kz0_idx[1],ky0_kz0_idx[2],:]
nav_data = torch.abs(fft_op_1d(nav_data)[0])
nav_signal_time_in_s = 2.5*kdata.header.acq_info.acquisition_time_stamp[ky0_kz0_idx[0],ky0_kz0_idx[1],ky0_kz0_idx[2]]/1000

fig, ax = plt.subplots(3,1);
for ind in range(3):
    ax[ind].imshow(rearrange(torch.abs(nav_data[:,ind,:]), 'angle x->x angle'),aspect='auto');
    ax[ind].set_xlim([0, 200]);

# Carry out SVD along coil and readout dimension
nav_data_pre_rearrange = nav_data
nav_data = rearrange(nav_data, 'k1k2 coil k0 -> k1k2 (coil k0)')
u, _, _ = torch.linalg.svd(nav_data - nav_data.mean(dim=0, keepdim=True))

# Find SVD component that is closest to expected respitory frequency
resp_freq_window_Hz = [0.2, 0.3]
dt = torch.mean(torch.diff(nav_signal_time_in_s, dim=0))
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

# Combine k2 and k1
kdata = kdata.rearrange_k2_k1_into_k1()

# Interpolate navigator from k-space center ky=kz=0 to all phase encoding points
resp_nav_interpolated = np.interp(2.5*kdata.header.acq_info.acquisition_time_stamp[0,0,:,0]/1000, nav_signal_time_in_s[:,0], resp_nav)

# %%  Split data into different motion phases
resp_idx = split_idx(torch.argsort(torch.as_tensor(resp_nav_interpolated)), 160*112, 160*50)
kdata_resp_resolved = kdata.split_k1_into_other(resp_idx, other_label='repetition')

# %% Motion-resolved reconstruction
direct_recon_resp_resolved = DirectReconstruction(kdata_resp_resolved, csm=itSENSE_recon.csm)
img_direct_resp_resolved = direct_recon_resp_resolved.forward(kdata_resp_resolved)

plot_resp_idx = [0, resp_idx.shape[0]-1]
fig, ax = plt.subplots(len(plot_resp_idx),1)
for nnd in range(len(plot_resp_idx)):
    ax[nnd].imshow(get_coronal_sagittal_axial_view(img_direct_resp_resolved.rss(), other=plot_resp_idx[nnd], z=50, y=96, x=115))


itSENSE_recon_resp_resolved = IterativeSENSEReconstruction(kdata_resp_resolved, csm=itSENSE_recon.csm, n_iterations=10)
img_itSENSE_resp_resolved = itSENSE_recon_resp_resolved.forward(kdata_resp_resolved)

img_resp_resolved_abs = img_itSENSE_resp_resolved.rss().numpy()
img_resp_resolved_abs /= img_resp_resolved_abs.max()

plot_resp_idx = [0, resp_idx.shape[0]-1]
fig, ax = plt.subplots(len(plot_resp_idx),1)
for nnd in range(len(plot_resp_idx)):
    img_coronal_sagittal_axial = get_coronal_sagittal_axial_view(img_resp_resolved_abs, other=plot_resp_idx[nnd], z=50, y=96, x=115)
    ax[nnd].imshow(img_coronal_sagittal_axial, cmap='grey', vmin=0, vmax=0.25)


for i in range(img_resp_resolved_abs.shape[0]):
    nii_resp = nib.Nifti1Image(img_resp_resolved_abs[i], affine=torch.eye(4))
    nib.save(nii_resp,os.path.join(path_out,"resp_state_iterative_"+str(i))) 
    
    img_coronal_sagittal_axial = get_coronal_sagittal_axial_view(img_resp_resolved_abs, other=nnd, z=showz, y=showy, x=showx)
    plt.imsave(os.path.join(path_out,f'resp_state_iterative_{nnd}.png'), img_coronal_sagittal_axial, dpi=300, cmap='grey', vmin=0, vmax=0.25)
    
    