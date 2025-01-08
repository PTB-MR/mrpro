# %%
# # Reconstruction of respiratory resolved images from a GRPE acquisition
# ### File name
fname = r'/sc-projects/sc-proj-cc06-agsack/noja11/ImageReconstruction/Data/meas_MID00291_FID96729_20240520_fov288_288_200mm_192_192_640_3D_saggital.mrd'
path_out = r'/sc-projects/sc-proj-cc06-agsack/noja11/ImageReconstruction/'
fname = '/echo/kolbit01/data/GRPE_Charite/2024_05_30/meas_MID00291_FID96729_20240520_fov288_288_200mm_192_192_640_3D_saggital.mrd'
path_out = '/echo/kolbit01/data/GRPE_Charite/2024_05_30/'

fname = '/echo/kolbit01/data/GRPE_Charite/2024_10_23/meas_MID00044_FID09487_20241021_fov288_288_160mm_154_92_200_3d_grpe_itl_pf0_6.h5'
path_out = '/echo/kolbit01/data/GRPE_Charite/2024_10_23/'

fname = '/home/kolbit01/data/GRPE/meas_MID00291_FID96729_20240520_fov288_288_200mm_192_192_640_3D_saggital.h5'
#fname = '/home/kolbit01/data/GRPE/meas_MID00044_FID09487_20241021_fov288_288_160mm_154_92_200_3d_grpe_itl_pf0_6.h5'
path_out = '/home/kolbit01/data/GRPE'

fname = '/echo/kolbit01/data/GRPE_Charite/2024_12_09/meas_MID00058_FID17527_grpe_seq_sag_short.mrd'
fname = '/echo/kolbit01/data/GRPE_Charite/2024_12_09/meas_MID00060_FID17529_grpe_seq_sag_long.mrd'

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

import time

flag_plot = False


# Correct for partial fourier sampling, as Siemens scanner does not save trajectory correctly
if not os.path.exists(fname.replace('.mrd', '_pf.mrd')):
    import ismrmrd
    
    # Get info and acquisitons from original data
    with ismrmrd.File(fname, 'r') as file:
        ds = file[list(file.keys())[-1]]
        ismrmrd_header = ds.header
        acquisitions = ds.acquisitions[:]

    # 192 ky
    if ismrmrd_header.encoding[0].encodedSpace.matrixSize.y == 192 or ismrmrd_header.encoding[0].encodedSpace.matrixSize.y == 222:
        ismrmrd_header.encoding[0].encodingLimits.kspace_encoding_step_1.maximum=111
        ismrmrd_header.encoding[0].encodingLimits.kspace_encoding_step_1.center=96
    # 92 ky
    elif ismrmrd_header.encoding[0].encodedSpace.matrixSize.y == 92:
        ismrmrd_header.encoding[0].encodingLimits.kspace_encoding_step_1.maximum=54
        ismrmrd_header.encoding[0].encodingLimits.kspace_encoding_step_1.center=46
    # 64 ky
    elif ismrmrd_header.encoding[0].encodedSpace.matrixSize.y == 128:
        ismrmrd_header.encoding[0].encodingLimits.kspace_encoding_step_1.maximum=127
        ismrmrd_header.encoding[0].encodingLimits.kspace_encoding_step_1.center=64
    else:
        raise NotImplementedError(f'{ismrmrd_header.encoding[0].encodedSpace.matrixSize.y} k1 points.')
    
    # Create new file
    fname_out = fname.replace('.mrd', '_pf.mrd')
    if fname_out == fname:
        raise ValueError('filennames cannot be identical')
    ds = ismrmrd.Dataset(fname_out)
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
    
    img = img.numpy()
    img = [np.rot90(img[other,:,y,:]), np.rot90(img[other,z,:,:]), np.rot90(img[other,:,::-1,x])]
    img_shape = np.max(np.asarray([i.shape for i in img]), axis=0)
    img = [np.pad(i, img_shape-np.asarray(i.shape)) for i in img]
    
    return np.concatenate(img, axis=1)


def save_image_data(img, suffix):
    if img.ndim == 5:
        img = img[:,0,...]
    img = img.abs()
    img= img.max()
    
    for i in range(img.shape[0]):
        nii_resp = nib.Nifti1Image(img[i].numpy(), affine=torch.eye(4))
        nib.save(nii_resp,os.path.join(path_out,suffix+str(i))) 
        
        img_coronal_sagittal_axial = get_coronal_sagittal_axial_view(img_resp_resolved_abs, other=i, z=showz, y=showy, x=showx)
        plt.imsave(os.path.join(path_out,suffix+f'{i}.png'), img_coronal_sagittal_axial, dpi=300, cmap='grey', vmin=0, vmax=0.25)
    
    
    
# Display slices
showz, showy, showx = [50, 96, 115]
#showz, showy, showx = [50, 96, 28]   
#showz, showy, showx = [50, 50, 70] 

tstart = time.time()

# %% Read data
# We are going to reconstruct an image using all the acquired data.
# Read the raw data, calculate the trajectory and dcf
kdata = KData.from_file(fname.replace('.mrd', '_pf.mrd'), KTrajectoryRpe(angle=torch.pi * 0.618034))

# Set the matrix sizes which are not encoded correctly in the pulseq sequence
kdata.header.encoding_matrix.z = kdata.header.encoding_matrix.y
kdata.header.recon_matrix.x = 192 #100 or 192
kdata.header.recon_matrix.y = 192 #100 or 192
kdata.header.recon_matrix.z = 192 #100 or 192

kdata.header.encoding_matrix.y = 192 #92 or 192
kdata.header.encoding_matrix.z = 192 #92 or 192

# Speed up things
kdata = kdata.compress_coils(n_compressed_coils=6)
kdata.header.recon_matrix.x = 192
kdata = kdata.remove_readout_os()


if flag_plot:
    plt.figure()
    plt.plot(kdata.traj.ky[0,:10,:,0], kdata.traj.kz[0,:10,:,0], 'ob')

#idx_rpe_lines = torch.as_tensor(torch.arange(0,50))[None,:]
#kdata_dynamic = kdata.split_k2_into_other(idx_rpe_lines, other_label='repetition')

# %% Coil maps
# Calculate coil maps
avg_recon = DirectReconstruction(kdata, csm = None)
avg_im = avg_recon(kdata)
csm_maps = CsmData.from_idata_walsh(avg_im, smoothing_width = 9)

if flag_plot:
    fig, ax = plt.subplots(5,1)
    for idx, cax in enumerate(ax.flatten()):
        img_coronal_sagittal_axial = get_coronal_sagittal_axial_view(csm_maps.data[:,idx], z=showz, y=showy, x=showx)
        cax.imshow(np.abs(img_coronal_sagittal_axial))


# %% Iterative reconstruction of average
itSENSE_recon = IterativeSENSEReconstruction(kdata, csm = csm_maps)
img_itSENSE = itSENSE_recon.forward(kdata)

if flag_plot:
    img_coronal_sagittal_axial = get_coronal_sagittal_axial_view(img_itSENSE.rss(), z=showz, y=showy, x=showx)
    img_coronal_sagittal_axial /= img_coronal_sagittal_axial.max()
    plt.figure()
    plt.imshow(img_coronal_sagittal_axial, cmap='grey', vmin=0, vmax=0.35)
    plt.imsave(os.path.join(path_out,f'average.png'), img_coronal_sagittal_axial, dpi=300, cmap='grey', vmin=0, vmax=0.35)

save_image_data(img_itSENSE, 'static_itSENSE_')
# %%
# Create the entire acquisition operator A
acquisition_operator = itSENSE_recon.fourier_op @ SensitivityOp(csm_maps)

# Define the gradient operator \nabla to be used in the operator K=[A, \nabla]^T for PDHG
from mrpro.operators import FiniteDifferenceOp

# The operator computes the directional derivatives along the the last two dimensions (x,y)
nabla_operator = FiniteDifferenceOp(dim=(-3, -2, -1), mode='forward')

from mrpro.algorithms.optimizers import pdhg
from mrpro.algorithms.optimizers.pdhg import PDHGStatus
from mrpro.operators import LinearOperatorMatrix, ProximableFunctionalSeparableSum
from mrpro.operators.functionals import L1NormViewAsReal, L2NormSquared, ZeroFunctional

# Regularization parameter for the $\ell_1$-norm
regularization_parameter = 1.0

# Set up the problem by using the previously described identification
l2 = 0.5 * L2NormSquared(target=kdata.data, divide_by_n=True)
l1 = regularization_parameter * L1NormViewAsReal(divide_by_n=True)

f = ProximableFunctionalSeparableSum(l2, l1)
g = ZeroFunctional()
K = LinearOperatorMatrix(((acquisition_operator,), (nabla_operator,)))

# initialize PDHG with iterative SENSE solution for warm start
initial_values = (img_itSENSE.data,)

# Run PDHG for a certain number of iterations
max_iterations = 100


# call backfunction to track the value of the objective functional f(K(x)) + g(x)
def callback(optimizer_status: PDHGStatus) -> None:
    """Print the value of the objective functional every 8th iteration."""
    iteration = optimizer_status['iteration_number']
    solution = optimizer_status['solution']
    if iteration % 1 == 0:
        print(optimizer_status['objective'](*solution).item())


(img_pdhg,) = pdhg(
    f=f, g=g, operator=K, initial_values=initial_values, max_iterations=max_iterations, callback=callback
)

if flag_plot:
    img_coronal_sagittal_axial = get_coronal_sagittal_axial_view(img_pdhg.data[0].abs(), z=showz, y=showy, x=showx)
    img_coronal_sagittal_axial /= img_coronal_sagittal_axial.max()
    plt.figure()
    plt.imshow(img_coronal_sagittal_axial, cmap='grey', vmin=0, vmax=0.35)
    plt.imsave(os.path.join(path_out,f'average.png'), img_coronal_sagittal_axial, dpi=300, cmap='grey', vmin=0, vmax=0.35)

save_image_data(img_pdhg, 'static_pdhg_')

# %% Respiratory self-navigator
# We are going to obtain a self-navigator from the k-space centre line ky = kz = 0
# Find readouts at ky=kz=0
fft_op_1d = FastFourierOp(dim=(-1,))
ky0_kz0_idx = torch.where(((kdata.traj.ky == 0) & (kdata.traj.kz == 0)))
nav_data = kdata.data[ky0_kz0_idx[0],:,ky0_kz0_idx[1],ky0_kz0_idx[2],:]
nav_data = torch.abs(fft_op_1d(nav_data)[0])
nav_signal_time_in_s = 2.5*kdata.header.acq_info.acquisition_time_stamp[ky0_kz0_idx[0],ky0_kz0_idx[1],ky0_kz0_idx[2]]/1000

if flag_plot:
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

if flag_plot:
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
total_npe = kdata.data.shape[-2]*kdata.data.shape[-3]
resp_idx = split_idx(torch.argsort(torch.as_tensor(resp_nav_interpolated)), int(total_npe*0.3), int(total_npe*0.15)) # 160*112, 160*50
kdata_resp_resolved = kdata.split_k1_into_other(resp_idx, other_label='repetition')

# %% Motion-resolved reconstruction
direct_recon_resp_resolved = DirectReconstruction(kdata_resp_resolved, csm=itSENSE_recon.csm)
img_direct_resp_resolved = direct_recon_resp_resolved.forward(kdata_resp_resolved)

save_image_data(img_direct_resp_resolved, 'dynamic_direct_')

if flag_plot:
    plot_resp_idx = [0, resp_idx.shape[0]-1]
    fig, ax = plt.subplots(len(plot_resp_idx),1)
    for nnd in range(len(plot_resp_idx)):
        ax[nnd].imshow(get_coronal_sagittal_axial_view(img_direct_resp_resolved.rss(), other=plot_resp_idx[nnd], z=showz, y=showy, x=showx))


itSENSE_recon_resp_resolved = IterativeSENSEReconstruction(kdata_resp_resolved, csm=itSENSE_recon.csm, n_iterations=10)
img_itSENSE_resp_resolved = itSENSE_recon_resp_resolved.forward(kdata_resp_resolved)

if flag_plot:
    img_resp_resolved_abs = img_itSENSE_resp_resolved.rss()
    img_resp_resolved_abs /= img_resp_resolved_abs.max()

    plot_resp_idx = [0, resp_idx.shape[0]-1]
    fig, ax = plt.subplots(len(plot_resp_idx),1)
    for nnd in range(len(plot_resp_idx)):
        img_coronal_sagittal_axial = get_coronal_sagittal_axial_view(img_resp_resolved_abs, other=plot_resp_idx[nnd], z=showz, y=showy, x=showx)
        ax[nnd].imshow(img_coronal_sagittal_axial, cmap='grey', vmin=0, vmax=0.25)


save_image_data(img_itSENSE_resp_resolved, 'dynamic_itSENSE_')
print(f'Total time {(time.time()-tstart)/60}min')

# %%
tstart = time.time()

# Create the entire acquisition operator A
acquisition_operator = itSENSE_recon_resp_resolved.fourier_op @ SensitivityOp(itSENSE_recon_resp_resolved.csm)

# Define the gradient operator \nabla to be used in the operator K=[A, \nabla]^T for PDHG
from mrpro.operators import FiniteDifferenceOp

# The operator computes the directional derivatives along the the last two dimensions (x,y)
nabla_operator = FiniteDifferenceOp(dim=(0,), mode='forward')

from mrpro.algorithms.optimizers import pdhg
from mrpro.algorithms.optimizers.pdhg import PDHGStatus
from mrpro.operators import LinearOperatorMatrix, ProximableFunctionalSeparableSum
from mrpro.operators.functionals import L1NormViewAsReal, L2NormSquared, ZeroFunctional

# Regularization parameter for the $\ell_1$-norm
regularization_parameter = 1.0

# Set up the problem by using the previously described identification
l2 = 0.5 * L2NormSquared(target=kdata_resp_resolved.data, divide_by_n=True)
l1 = regularization_parameter * L1NormViewAsReal(divide_by_n=True)

f = ProximableFunctionalSeparableSum(l2, l1)
g = ZeroFunctional()
K = LinearOperatorMatrix(((acquisition_operator,), (nabla_operator,)))

# initialize PDHG with iterative SENSE solution for warm start
initial_values = (img_itSENSE_resp_resolved.data,)

max_iterations = 100
(img_pdhg,) = pdhg(
    f=f, g=g, operator=K, initial_values=initial_values, max_iterations=max_iterations, callback=callback
)

if flag_plot:
    img_resp_resolved_abs = img_pdhg.abs()[:,0,...]
    img_resp_resolved_abs /= img_resp_resolved_abs.max()

    plot_resp_idx = [0, resp_idx.shape[0]-1]
    fig, ax = plt.subplots(len(plot_resp_idx),1)
    for nnd in range(len(plot_resp_idx)):
        img_coronal_sagittal_axial = get_coronal_sagittal_axial_view(img_resp_resolved_abs, other=plot_resp_idx[nnd], z=showz, y=showy, x=showx)
        ax[nnd].imshow(img_coronal_sagittal_axial, cmap='grey', vmin=0, vmax=0.25)

save_image_data(img_pdhg, 'dynamic_pdhg_')
print(f'Total time TV {(time.time()-tstart)/60}min')
# %%

tstart = time.time()

#### IMAGE DENOISING
# Regularization parameter for the $\ell_1$-norm
regularization_parameter = 1.0

# Set up the problem by using the previously described identification
l2 = 0.5 * L2NormSquared(target=img_itSENSE_resp_resolved.data, divide_by_n=True)
l1 = regularization_parameter * L1NormViewAsReal(divide_by_n=True)

f = ProximableFunctionalSeparableSum(l2, l1)
g = ZeroFunctional()
K = LinearOperatorMatrix(((acquisition_operator,), (nabla_operator,)))

# initialize PDHG with iterative SENSE solution for warm start
initial_values = (img_itSENSE_resp_resolved.data,)

max_iterations = 100
(img_admm_denoised,) = pdhg(
    f=f, g=g, operator=K, initial_values=initial_values, max_iterations=max_iterations, callback=callback
)

save_image_data(img_admm_denoised, 'dynamic_admm_denoised_')

#### DATA CONSISTENCY

#### FINAL ITERATION

print(f'Total time ADMM {(time.time()-tstart)/60}min')
# %%
