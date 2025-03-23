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

path_data = '/echo/kolbit01/data/GRPE_Charite/2024_12_09/'
fname = path_data + 'meas_MID00058_FID17527_grpe_seq_sag_short.mrd'
fname = path_data + 'meas_MID00060_FID17529_grpe_seq_sag_long.mrd'

path_data = '/echo/kolbit01/data/GRPE_Charite/2025_01_14/'
fname = path_data + 'meas_MID00277_FID22434_Test_long_libbalanceCheckOn_phantom.mrd'
#fname = path_data + 'meas_MID00274_FID22431_Test_short_libbalanceCheckOn_phantom.mrd'

# ### Imports
import matplotlib.pyplot as plt
import torch
import numpy as np
from einops import rearrange, repeat
from mrpro.data import CsmData, KTrajectory, DcfData, IData, KData, Rotation
from mrpro.data.traj_calculators import KTrajectoryRpe, KTrajectoryPulseq
from mrpro.operators import FourierOp, FastFourierOp, SensitivityOp, IdentityOp, DensityCompensationOp, FiniteDifferenceOp, LinearOperatorMatrix, ProximableFunctionalSeparableSum
from mrpro.algorithms.optimizers import cg, pdhg
from mrpro.algorithms.reconstruction import DirectReconstruction, IterativeSENSEReconstruction
from mrpro.operators.functionals import L1NormViewAsReal, L2NormSquared, ZeroFunctional
from mrpro.utils import split_idx
from mrpro.utils import unsqueeze_right
import os
import nibabel as nib

import time

flag_plot = True


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
    elif ismrmrd_header.encoding[0].encodedSpace.matrixSize.y == 92 or ismrmrd_header.encoding[0].encodedSpace.matrixSize.y == 154:
        ismrmrd_header.encoding[0].encodingLimits.kspace_encoding_step_1.maximum=54
        ismrmrd_header.encoding[0].encodingLimits.kspace_encoding_step_1.center=46
    # 64 ky
    elif ismrmrd_header.encoding[0].encodedSpace.matrixSize.y == 128:
        ismrmrd_header.encoding[0].encodingLimits.kspace_encoding_step_1.maximum=127
        ismrmrd_header.encoding[0].encodingLimits.kspace_encoding_step_1.center=64
    else:
        raise NotImplementedError(f'{ismrmrd_header.encoding[0].encodedSpace.matrixSize.y} k1 points.')
    
    ismrmrd_header.encoding[0].encodedSpace.matrixSize.z = ismrmrd_header.encoding[0].encodedSpace.matrixSize.y
    ismrmrd_header.encoding[0].reconSpace.matrixSize.z = ismrmrd_header.encoding[0].reconSpace.matrixSize.y
    
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
        nib.save(nii_resp,os.path.join(path_data,suffix+str(i))) 
        
        img_coronal_sagittal_axial = get_coronal_sagittal_axial_view(img, other=i, z=showz, y=showy, x=showx)
        plt.imsave(os.path.join(path_data,suffix+f'{i}.png'), img_coronal_sagittal_axial, dpi=300, cmap='grey', vmin=0, vmax=0.25)
    

def tv_reg_reco(kdata, csm, img_initial, reg_weight=0.1, reg_weight_t=0.1, n_iterations=100):
    fourier_operator = FourierOp.from_kdata(kdata)
    csm_operator = SensitivityOp(csm)
    acquisition_operator = fourier_operator @ csm_operator
    
    if img_initial.data.shape[0] == 1:
        tv_dim = (-3,-2,-1)
        regularisation_weight = torch.tensor([reg_weight, reg_weight, reg_weight])
    else:
        tv_dim = (-5, -3,-2,-1)
        regularisation_weight = torch.tensor([reg_weight_t, reg_weight, reg_weight, reg_weight])
        
    nabla_operator = FiniteDifferenceOp(dim=tv_dim, mode='forward')
    l2 = 0.5 * L2NormSquared(target=kdata.data)
    l1 = L1NormViewAsReal(
            weight=unsqueeze_right(regularisation_weight, kdata.data.ndim)
        )

    f = ProximableFunctionalSeparableSum(l2, l1)
    g = ZeroFunctional()
    K = LinearOperatorMatrix(((acquisition_operator,), (nabla_operator,)))

    initial_values = (img_initial.data.clone(),)

    (img_pdhg,) = pdhg(
        f=f, g=g, operator=K, initial_values=initial_values, max_iterations=n_iterations)
    return IData(data=img_pdhg, header=img_initial.header)


def rearrange_k2_k1_into_k1(kdata: KData) -> KData:
    """Rearrange kdata from (... k2 k1 ...) to (... 1 (k2 k1) ...).

    Parameters
    ----------
    kdata
        K-space data (other coils k2 k1 k0)

    Returns
    -------
        K-space data (other coils 1 (k2 k1) k0)
    """
    # Rearrange data
    kdat = rearrange(kdata.data, '... coils k2 k1 k0->... coils 1 (k2 k1) k0')

    # Rearrange trajectory
    ktraj = rearrange(kdata.traj.as_tensor(), 'dim ... k2 k1 k0-> dim ... 1 (k2 k1) k0')
    
    header = kdata.header.apply(
            lambda field: rearrange(field, 'dim ... k2 k1 k0-> dim ... 1 (k2 k1) k0') 
            if isinstance(field, Rotation | torch.Tensor)
            else field
        )

    return KData(header, kdat, KTrajectory.from_tensor(ktraj))
    

# %% Read data
# We are going to reconstruct an image using all the acquired data.
# Read the raw data, calculate the trajectory and dcf
# kdata = KData.from_file(fname.replace('.mrd', '_pf.mrd'), KTrajectoryRpe(angle=torch.pi * 0.618034, shift_between_rpe_lines=[0,]))
kdata = KData.from_file(fname.replace('.mrd', '_pf.mrd'), KTrajectoryRpe(angle=torch.pi * 0.618034))

# Display slices
showz, showy, showx = [50, 96, 120] if kdata.header.recon_matrix.y == 192 else [50, 96, 28] 

# Set the matrix sizes which are not encoded correctly in the pulseq sequence
kdata.header.encoding_matrix.z = kdata.header.encoding_matrix.y
kdata.header.recon_matrix.z = kdata.header.recon_matrix.y

# Temp Fix
kdata.header.encoding_matrix.y = 192
kdata.header.encoding_matrix.z = 192
kdata.header.recon_matrix.x = 192
kdata.header.recon_matrix.y = 192
kdata.header.recon_matrix.z = 192

# Speed up things
kdata = kdata.compress_coils(n_compressed_coils=6)
kdata = kdata.remove_readout_os()

if flag_plot:
    plt.figure()
    plt.plot(kdata.traj.ky[0,0,:10,:,0], kdata.traj.kz[0,0,:10,:,0], 'ob')

# %% Coil maps
# Calculate coil maps
tstart = time.time()
avg_recon = DirectReconstruction(kdata, csm = None)
avg_im = avg_recon(kdata)
csm_maps = CsmData.from_idata_inati(avg_im, smoothing_width = 3)
print(f'Csm {(time.time()-tstart)/60}min')

if flag_plot:
    fig, ax = plt.subplots(5,1)
    for idx, cax in enumerate(ax.flatten()):
        img_coronal_sagittal_axial = get_coronal_sagittal_axial_view(csm_maps.data[:,idx], z=showz, y=showy, x=showx)
        cax.imshow(np.abs(img_coronal_sagittal_axial))


# %% Average reconstruction
tstart = time.time()
direct_recon = DirectReconstruction(kdata, csm=csm_maps)
img_direct = direct_recon.forward(kdata)
print(f'Average reco direct {(time.time()-tstart)/60}min')

tstart = time.time()
itSENSE_recon = IterativeSENSEReconstruction(kdata, csm = csm_maps)
img_itSENSE = itSENSE_recon.forward(kdata)
print(f'Average reco itSENSE {(time.time()-tstart)/60}min')

tstart_pdhg = time.time()
img_pdhg = tv_reg_reco(kdata, csm_maps, img_initial=img_itSENSE, reg_weight=1e-6, n_iterations=100)
print(f'Average reco PDHG {(time.time()-tstart_pdhg)/60}min')

if flag_plot:
    img_save = []
    for idx, img in enumerate([img_direct.rss(), img_itSENSE.rss(), img_pdhg.rss()]):
        img_coronal_sagittal_axial = get_coronal_sagittal_axial_view(img, z=showz, y=showy, x=showx)
        img_coronal_sagittal_axial /= img_coronal_sagittal_axial.max()
        img_save.append(img_coronal_sagittal_axial)
    img_save = np.concatenate(img_save)
    plt.figure()
    plt.imshow(img_save, cmap='grey', vmin=0, vmax=0.35)
    plt.imsave(os.path.join(path_data, fname.replace('.mrd', '_average.png')), img_save, dpi=300, cmap='grey', vmin=0, vmax=0.35)



# %% Respiratory self-navigator
# We are going to obtain a self-navigator from the k-space centre line ky = kz = 0
# Find readouts at ky=kz=0
fft_op_1d = FastFourierOp(dim=(-1,))
ky0_kz0_idx = torch.where(((kdata.traj.ky == 0) & (kdata.traj.kz == 0)))
nav_data = kdata.data[ky0_kz0_idx[0],:,ky0_kz0_idx[2],ky0_kz0_idx[3],:]
nav_data = torch.abs(fft_op_1d(nav_data)[0])
nav_signal_time_in_s = kdata.header.acq_info.acquisition_time_stamp[ky0_kz0_idx[0],0,ky0_kz0_idx[2],ky0_kz0_idx[3],0]

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
resp_freq_window_Hz = [0.2, 0.5]
dt = torch.mean(torch.diff(nav_signal_time_in_s, dim=0))
U_freq = torch.abs(torch.fft.fft(u, dim=0)) 

fmax_Hz = 1/dt 
f_Hz = torch.linspace(0, fmax_Hz, u.shape[0]) 

condition_below = f_Hz <=resp_freq_window_Hz[1] 
condition_above = f_Hz >= resp_freq_window_Hz[0] 
condition_window = condition_below * condition_above 

if flag_plot:
    plt.figure() 
    plt.plot(f_Hz, U_freq) 
    plt.plot([resp_freq_window_Hz[0], resp_freq_window_Hz[0]], [0, U_freq.max()], '-k')
    plt.plot([resp_freq_window_Hz[1], resp_freq_window_Hz[1]], [0, U_freq.max()], '-k')

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
    plt.imshow(rearrange(torch.abs(nav_data_pre_rearrange[:,2,:]), 'angle x->x angle'), aspect='auto')
    plt.plot(rescaled_resp_nav, ':w')
    plt.xlim(0,200)

# Combine k2 and k1
kdata = rearrange_k2_k1_into_k1(kdata)

# Interpolate navigator from k-space center ky=kz=0 to all phase encoding points
resp_nav_interpolated = np.interp(kdata.header.acq_info.acquisition_time_stamp[0,0,0,:,0], nav_signal_time_in_s, resp_nav)

# %%  Split data into different motion phases
total_npe = kdata.data.shape[-2]*kdata.data.shape[-3]
resp_idx = split_idx(torch.argsort(torch.as_tensor(resp_nav_interpolated)), int(total_npe*0.22), int(total_npe*0.11)) # 160*112, 160*50
kdata_resp_resolved = kdata.split_k1_into_other(resp_idx, other_label='repetition')

# %% Motion-resolved reconstruction
tstart = time.time()
direct_recon_resp_resolved = DirectReconstruction(kdata_resp_resolved, csm=csm_maps)
img_direct_resp_resolved = direct_recon_resp_resolved.forward(kdata_resp_resolved)
print(f'Dynamic reco direct {(time.time()-tstart)/60}min')

tstart = time.time()
itSENSE_recon_resp_resolved = IterativeSENSEReconstruction(kdata_resp_resolved, csm=csm_maps, n_iterations=10)
img_itSENSE_resp_resolved = itSENSE_recon_resp_resolved.forward(kdata_resp_resolved)
print(f'Dynamic reco itSENSE {(time.time()-tstart)/60}min')

tstart_pdhg = time.time()
img_pdhg_resp_resolved = tv_reg_reco(kdata_resp_resolved, csm_maps, img_initial=img_itSENSE_resp_resolved, reg_weight=2e-7, reg_weight_t=2e-5, n_iterations=100)
print(f'Dynamic reco PDHG {(time.time()-tstart_pdhg)/60}min')


if flag_plot:
    img_dynamic = [img_direct_resp_resolved.rss(), img_itSENSE_resp_resolved.rss(), img_pdhg_resp_resolved.rss()]
    for nnd in range(img_dynamic[0].shape[0]):
        img_save = []
        for idx, img in enumerate(img_dynamic):
            img /= img.max()
            img_save.append(get_coronal_sagittal_axial_view(img, other=nnd, z=showz, y=showy, x=showx))
        img_save = np.concatenate(img_save)
        plt.figure()
        plt.imshow(img_save, cmap='grey', vmin=0, vmax=0.35)
        plt.imsave(os.path.join(path_data,fname.replace('.mrd', f'_dynamic_{nnd}.png')), img_save, dpi=300, cmap='grey', vmin=0, vmax=0.35)
  
        
        
# %%
from PIL import Image, GifImagePlugin, ImageEnhance
GifImagePlugin.LOADING_STRATEGY = GifImagePlugin.LoadingStrategy.RGB_AFTER_DIFFERENT_PALETTE_ONLY
img_pil = []
for nnd in range(img_dynamic[0].shape[0]):
    img_pil.append(ImageEnhance.Brightness(Image.open(os.path.join(path_data,fname.replace('.mrd', f'_dynamic_{nnd}.png'))).convert('L')).enhance(1.5))
img_pil[0].save(fp=os.path.join(path_data,fname.replace('.mrd', '_dynamic.gif')), format='GIF', append_images=img_pil[1:], save_all=True, 
                duration=200, loop=0, optimize=False, lossless=True)
# %%
