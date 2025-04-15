# %% [markdown]
# # Motion-corrected image reconstruction
# Physiological motion (e.g. due to breathing or the beating of the heart) is a common challenge for MRI. Motion during 
# the data acquisition can lead to motion artifacts in the reconstructed image. Mathematically speaking, the motion
# occuring during data acquisition can be described by linear operators $M_m$ for each motion state $m$:
#
# $ y = \sum_m A_m M_m x_{\mathrm{true}} + n, $
#
# where $y$ is the acquired k-space data, $A_m$ is the acquisition operator describing which data was acquired in motion 
# state $m$ and $n$ describes complex Gaussian noise.

# %% [markdown]
# There are different approaches of how to minimize the impact of motion. The simplest approach is to acquire data only
# in a single motion state by e.g. asking the subject to hold their breath or synchronize the acquisition with the heart
# beat. This reduces the above problem to
# 
# $ y = A x_{\mathrm{true}} + n, $
#
# but this is not always possible and can also lead to long acquisition times.
#
# A more efficient approach is to acquire data in different motion states, estimate $M_m$ and solve the above problem. 
# This is often referred to motion-corrected image reconstruction (MCIR). For some examples have a look at 
# [Kolbitsch et al., JNM 2017](http://doi.org/10.2967/jnumed.115.171728), 
# [Ippoliti et al., MRM 2019](http://doi.wiley.com/10.1002/mrm.27867) or 
# [Mayer et al., JNM 2021](http://doi.org/10.1007/s00259-020-05180-4)
#
# Here we will show how to do a MCIR of a free-breathing acquisition of the thorax following these steps:
# 1. Estimate a respiratory self-navigator from the acquired k-space data
# 2. Use the self-navigator to separate the data into differnet breathing states
# 3. Reconstruct dynamic images of different breathing states
# 4. Estimate non-rigid motion fields from the dynamic images
# 5. Use the motion fields to solve the above equation and obtain a motion-corrected image
#
# To achieve high image quality a TV-regularised image reconstruction is used here. To safe time we will use only a few 
# iterations. Increase 'n_iterations_tv' in the code to 100 to get a better image quality.

# %%
n_iterations_tv = 100

# %% [markdown]
# ### Data acquisition
# The data was acquired with a Golden Radial Phase Encoding (GRPE) sampling scheme 
# [[Prieto et al., MRM 2010](http://doi.org/10.1002/mrm.22446)]. This sampling scheme combines a Cartesian readout with 
# radial phase encoding in the 2D ky-kz plane. The central k-space line (i.e. 1D projection of the object along the 
# foot-head direction) is acquired repeatedly and can be used as a respiratory self-navigator. This sequence was 
# implemented in pulseq and also available. 

# %% tags=["hide-cell"]  mystnb={"code_prompt_show": "Show download details"}
# Download raw data from Zenodo

# %%
# ### File name
path_data = '/echo/kolbit01/data/GRPE_Charite/2025_01_14/'
fname = path_data + 'meas_MID00277_FID22434_Test_long_libbalanceCheckOn_phantom_pf_corr.mrd'
n_pe_ms = 0.22
fname = path_data + 'meas_MID00274_FID22431_Test_short_libbalanceCheckOn_phantom_pf_corr.mrd'
n_pe_ms = 0.36

# ### Imports
import matplotlib.pyplot as plt
import torch
import numpy as np
from einops import rearrange, repeat
from mrpro.data import CsmData, KTrajectory, IData, KData, Rotation
from mrpro.data.traj_calculators import KTrajectoryRpe
from mrpro.operators import FourierOp, FastFourierOp, SensitivityOp, AveragingOp, GridSamplingOp, FiniteDifferenceOp, LinearOperatorMatrix, ProximableFunctionalSeparableSum
from mrpro.algorithms.optimizers import cg, pdhg
from mrpro.algorithms.reconstruction import DirectReconstruction, IterativeSENSEReconstruction
from mrpro.operators.functionals import L1NormViewAsReal, L2NormSquared, ZeroFunctional
from mrpro.utils import split_idx
from mrpro.utils import unsqueeze_right


import time


flag_plot = True

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
    
# %% [markdown]
# ### Motion-corrupted image reconstruction
# As a first step we will reconstruct the image using all the acquired data which will lead to an image corrupted by 
# respiratory motion.

# %%
kdata = KData.from_file(fname, KTrajectoryRpe(angle=torch.pi * 0.618034))

# Speed up things
kdata = kdata.compress_coils(n_compressed_coils=6)
kdata = kdata.remove_readout_os()

# Calculate coil maps
tstart = time.time()
avg_recon = DirectReconstruction(kdata, csm = None)
avg_im = avg_recon(kdata)
csm_maps = CsmData.from_idata_inati(avg_im, smoothing_width = 5)
print(f'Csm {(time.time()-tstart)/60}min')

#  SENSE reconstruction
iterative_sense = IterativeSENSEReconstruction(kdata, csm=csm_maps)
img_sense = iterative_sense.forward(kdata)
img_sense = torch.squeeze(img_sense.rss())
img_sense /= img_sense.max()


fig, ax = plt.subplots(1,3, squeeze=False, figsize=(8, 4))
[a.set_xticks([]) for a in ax.flatten()]
[a.set_yticks([]) for a in ax.flatten()]
ax[0,0].imshow(torch.rot90(img_sense[:,80,:]), vmin=0, vmax=0.2, cmap='grey')
ax[0,0].set_title('Coronal', fontsize=18)
ax[0,1].imshow(torch.flipud(torch.rot90(img_sense[:,:,80])), vmin=0, vmax=0.2, cmap='grey')
ax[0,1].set_title('Transversal', fontsize=18)
ax[0,2].imshow(torch.rot90(img_sense[100,:,:]), vmin=0, vmax=0.2, cmap='grey')
ax[0,2].set_title('Saggital', fontsize=18)



# %% [markdown]
# ### 1. Respiratory self-navigator
# We are going to obtain a self-navigator from the k-space centre line ky = kz = 0
# %%
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

# %% [markdown]
# ### 2. Split data into different breathing states
# The self-navigator is used to split the data into different motion phases.

# %%
# Combine k2 and k1
kdata = rearrange_k2_k1_into_k1(kdata)

# Interpolate navigator from k-space center ky=kz=0 to all phase encoding points
resp_nav_interpolated = np.interp(kdata.header.acq_info.acquisition_time_stamp[0,0,0,:,0], nav_signal_time_in_s, resp_nav)

# %%  Split data into different motion phases
total_npe = kdata.data.shape[-2]*kdata.data.shape[-3]
resp_idx = split_idx(torch.argsort(torch.as_tensor(resp_nav_interpolated)), int(total_npe*n_pe_ms), int(total_npe*n_pe_ms/2))
kdata_resp_resolved = kdata.split_k1_into_other(resp_idx, other_label='repetition')

# %% [markdown]
# ### 3. Reconstruct dynamic images of different breathing states

# %% 
tstart = time.time()
direct_recon_resp_resolved = IterativeSENSEReconstruction(kdata_resp_resolved, csm=csm_maps)
img_direct_resp_resolved = direct_recon_resp_resolved.forward(kdata_resp_resolved)
print(f'Dynamic reco direct {(time.time()-tstart)/60}min')

tstart_pdhg = time.time()
img_pdhg_resp_resolved = tv_reg_reco(kdata_resp_resolved, csm_maps, img_initial=img_direct_resp_resolved, reg_weight=2e-7, reg_weight_t=2e-6, n_iterations=n_iterations_tv)
print(f'Dynamic reco PDHG {(time.time()-tstart_pdhg)/60}min')


# %%
img_dynamic = [img_direct_resp_resolved.rss(), img_pdhg_resp_resolved.rss()]
slice_idx = 84
fig, ax = plt.subplots(2,3, squeeze=False, figsize=(12, 8))
[a.set_xticks([]) for a in ax.flatten()]
[a.set_yticks([]) for a in ax.flatten()]
for img_idx, img in enumerate(img_dynamic):
    img /= img.max()
    ax[img_idx,0].imshow(torch.rot90(img[0,:,slice_idx,:]), vmin=0, vmax=0.2, cmap='grey')
    ax[img_idx,0].set_title('MS 1', fontsize=18)
    ax[img_idx,1].imshow(torch.rot90(img[-1,:,slice_idx,:]), vmin=0, vmax=0.2, cmap='grey')
    ax[img_idx,1].set_title(f'MS {img.shape[0]}', fontsize=18)
    ax[img_idx,2].imshow(torch.rot90(torch.abs(img[0,:,slice_idx,:]-img[-1,:,slice_idx,:])), vmin=0, vmax=0.2, cmap='grey')
    ax[img_idx,2].set_title(f'|MS 1 - MS {img.shape[0]}|', fontsize=18)
ax[0,0].set_ylabel('Iterative SENSE', fontsize=18)
ax[1,0].set_ylabel('TV-regularization', fontsize=18)


# %% [markdown]
# ### 4. Estimate the motion fields from the dynamic images
# 

# %% [markdown]
# ### 5. Use the motion fields to solve the above equation and obtain a motion-corrected image


# %%
mf = torch.as_tensor(np.load('/echo/kolbit01/data/GRPE_Charite/2025_01_14/mf_mirtk_sp3.npy'), dtype=torch.float32)
# The motion fields describe the motion transformation in voxel units. We need to convert them to the convention of
# the grid sampling operator, which is in [-1, 1] for each dimension. 
for dim in range(3):
    mf[...,dim] /= kdata_resp_resolved.data.shape[-1] / 2
n_motion_states = mf.shape[0]
unity_matrix = repeat(torch.cat((torch.eye(3), torch.zeros(3,1)), dim=1), 'h w->n_ms h w', n_ms=n_motion_states)
grid = torch.nn.functional.affine_grid(unity_matrix, (n_motion_states, 1, *mf.shape[1:-1]))
grid += mf
grid[grid > 1] = 1
grid[grid < -1] = -1
moco_op = GridSamplingOp(grid, input_shape=kdata.header.recon_matrix)

fourier_op = direct_recon_resp_resolved.fourier_op
dcf_op = direct_recon_resp_resolved.dcf.as_operator()
csm_op = SensitivityOp(csm_maps)
averaging_op = AveragingOp(dim=0)
acquisition_operator = fourier_op @ moco_op.H @ csm_op @ averaging_op.H
(initial_value,) = acquisition_operator.H(dcf_op(kdata_resp_resolved.data)[0])
(right_hand_side,) = acquisition_operator.H(kdata_resp_resolved.data)
operator = acquisition_operator.H @ acquisition_operator

tstart = time.time()
img_mcir = cg(
    operator, right_hand_side, initial_value=initial_value, max_iterations=30, tolerance=0.0
)
print(f'MCIR {(time.time()-tstart)/60}min')
img_mcir = torch.squeeze(img_mcir.abs())
img_mcir = img_mcir/img_mcir.max()


nabla_operator = FiniteDifferenceOp(dim=(-3,-2,-1), mode='forward')
l2 = 0.5 * L2NormSquared(target=kdata_resp_resolved.data)
l1 = L1NormViewAsReal(weight=(unsqueeze_right(torch.as_tensor((1e-9, 1e-9, 1e-9)), kdata.data.ndim)))
f = ProximableFunctionalSeparableSum(l2, l1)
g = ZeroFunctional()
K = LinearOperatorMatrix(((acquisition_operator,), (nabla_operator,)))

tstart = time.time()
# Calculate operator norm
operator_norm = K.operator_norm(*[torch.randn_like(v) for v in (initial_value,)], relative_tolerance=0.1)
primal_stepsize = dual_stepsize = 1.0 / operator_norm
(img_pdhg_mcir,) = pdhg(
    f=f, g=g, operator=K, initial_values=(initial_value,), max_iterations=n_iterations_tv, primal_stepsize=primal_stepsize, dual_stepsize=dual_stepsize)
print(f'MCIR + TV {(time.time()-tstart)/60}min')
img_pdhg_mcir = torch.squeeze(img_pdhg_mcir.abs())
img_pdhg_mcir = img_pdhg_mcir/img_pdhg_mcir.max()

slicei = 84
for shift in np.arange(-10,10):
    slice_idx = slicei + shift
    fig, ax = plt.subplots(1,3, squeeze=False, figsize=(8, 4))
    [a.set_xticks([]) for a in ax.flatten()]
    [a.set_yticks([]) for a in ax.flatten()]
    ax[0,0].imshow(torch.rot90(img_sense[:,slice_idx,:]), vmin=0, vmax=0.2, cmap='grey')
    ax[0,0].set_title('Uncorrected', fontsize=18)
    ax[0,1].imshow(torch.rot90(img_mcir[:,slice_idx,:]), vmin=0, vmax=0.15, cmap='grey')
    ax[0,1].set_title(f'MCIR', fontsize=18)
    ax[0,2].imshow(torch.rot90(img_pdhg_mcir[:,slice_idx,:]), vmin=0, vmax=0.2, cmap='grey')
    ax[0,2].set_title(f'MCIR + TV', fontsize=18)


# %%
