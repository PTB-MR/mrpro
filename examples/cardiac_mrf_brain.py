# %% [markdown]
# # Cardiac Magnetic Resonance Fingerprinting

# %%
# Imports
from pathlib import Path
import pydicom
import numpy as np

import matplotlib.pyplot as plt
import torch
import scipy as sp
from einops import rearrange
from mrpro.operators import FourierOp, SensitivityOp
from mrpro.utils import split_idx
from mrpro.data import KData, DcfData, IData, CsmData, KTrajectory
from mrpro.data.traj_calculators import KTrajectoryIsmrmrd, KTrajectoryPulseq, KTrajectoryRadial2D
from mrpro.algorithms.reconstruction import DirectReconstruction, IterativeSENSEReconstruction
from mrpro.operators.models import EpgMrfFispWithPreparation


recon = 'iterative' # direct, iterative
used_trajectory = 'spiral' # radial, spiral

fname_t1_ref = Path('/echo/_allgemein/projects/pulseq/measurements/2024-06-24_t1mes_cMRF_pulseq/meas_MID00033_FID05475_t1map_T1MES/dicom/170734_t1map_T1MES_MOCO_T1_0005/SCHUENKE_PHANTOM-0005-0001.dcm')
fname_t2_ref = Path('/Users/kolbit01/Documents/PTB/Data/mrpro/brain_cmrf/t2map_brain_256_MOCO_T2_10_MR/1.dcm')


if used_trajectory == 'spiral':
    pname = Path('/Users/kolbit01/Documents/PTB/Data/mrpro/brain_cmrf/meas_MID00027_FID08310_20240911_spiral_cMRF_705rep_trig_delay800ms/')
    scan_name = Path('meas_MID00027_FID08310_20240911_spiral_cMRF_705rep_trig_delay800ms_with_traj.h5')

    n_lines_per_img = 5
    n_lines_overlap= 4
    n_cg_it = 90

elif used_trajectory == 'radial':
    pname = Path('//Users/kolbit01/Documents/PTB/Data/mrpro/brain_cmrf/meas_MID00025_FID08308_20240911_radial_cMRF_705rep_trig_delay800ms/')
    scan_name = Path('meas_MID00025_FID08308_20240911_radial_cMRF_705rep_trig_delay800ms_with_traj.h5')

    n_lines_per_img = 20
    n_lines_overlap= 10
    n_cg_it = 30

else:
    raise ValueError(f'method {used_trajectory} not recognised')

rr_duration = 1200
fname_angle = Path('/Users/kolbit01/Documents/PTB/Data/mrpro/rabbit/cMRF_fa_705rep.txt')

with open(fname_angle, "r") as file:
    fa = torch.as_tensor([float(line) for line in file.readlines()])/180 * torch.pi

# %%
# Function to calculate mask
def calc_2d_mask(idat, mask_thresh=0.01):
    # Calculate mask
    idat_abs = torch.abs(torch.squeeze(idat))
    idat_norm = idat_abs / idat_abs.max()
    mask = torch.zeros(idat_norm.shape)
    mask[idat_norm > mask_thresh] = 1

    mask = torch.as_tensor(sp.ndimage.binary_opening(mask.to(dtype=torch.int64), torch.ones((3, 3), dtype=torch.int64)))
    mask = torch.as_tensor(sp.ndimage.binary_closing(mask.to(dtype=torch.int64), torch.ones((9, 9), dtype=torch.int64)))
    mask = torch.as_tensor(sp.ndimage.binary_fill_holes(mask.to(dtype=torch.int64)))

    return(mask)


# %%
# Image reconstruction of average image
kdata = KData.from_file(pname / scan_name, KTrajectoryIsmrmrd())
kdata.header.recon_matrix.x = 256
kdata.header.recon_matrix.y = 256

avg_recon = DirectReconstruction.from_kdata(kdata)
avg_im = avg_recon(kdata)
mask = calc_2d_mask(avg_im.rss(), mask_thresh=0.1)

fig, ax = plt.subplots(1,2)
ax[0].imshow(torch.squeeze(avg_im.rss()), cmap='grey')
ax[1].imshow(mask, cmap='grey')

# %%
# Split data into dynamics and reconstruct
dyn_idx = split_idx(torch.arange(0,47), n_lines_per_img, n_lines_overlap)
dyn_idx = torch.cat([dyn_idx + ind*47 for ind in range(15)], dim=0)

kdata_dyn = kdata.split_k1_into_other(dyn_idx, other_label='repetition')

if recon == 'direct':
    dyn_recon = DirectReconstruction.from_kdata(kdata_dyn, coil_combine=False)
elif recon == 'iterative':
    dyn_recon = IterativeSENSEReconstruction(FourierOp.from_kdata(kdata_dyn), n_iterations=n_cg_it)
else:
    raise ValueError(f'recon {recon} not recognised.')
dyn_recon.csm = avg_recon.csm

dcf_data_dyn = rearrange(avg_recon.dcf.data, 'k2 k1 other k0->other k2 k1 k0')
dcf_data_dyn = rearrange(dcf_data_dyn[dyn_idx.flatten(),...], '(other k1) 1 k2 k0->other k2 k1 k0', k1=dyn_idx.shape[-1])
dyn_recon.dcf = DcfData(dcf_data_dyn)

img = dyn_recon(kdata_dyn).rss()[:,0,:,:]

fig, ax = plt.subplots(1,5, figsize=(12,4))
for ind in range(4):
    ax[ind].imshow(img[ind*8,:,:].abs(), cmap='grey')
    ax[ind].set_title(f'Image # {ind*6}')
ax[-1].imshow(torch.abs(torch.mean(img[:,:,:],dim=0)))
ax[-1].set_title('Average over all images')

# %%
# Dictionary settings
t1 = torch.linspace(100, 2000, 120)[:,None]
t2 = torch.linspace(10, 200, 10)[None,:]
t1, t2 = torch.broadcast_tensors(t1, t2)
t1 = t1.flatten()
t2 = t2.flatten()
m0 = torch.ones_like(t1)

# %%
# Dictionary calculationg
flip_angles = fa
rf_phases = 0
te = kdata.header.te * 1000
tr = kdata.header.tr * 1000
inv_prep_ti = [21,None,None,None,None]*3 # 20 ms delay after inversion pulse in block 0
t2_prep_te = [None,None,30,50,100]*3 # T2-preparation pulse with TE = 30, 50, 100
n_rf_pulses_per_block = 47 # 47 RF pulses in each block
delay_after_block = [0, 30, 50, 100, 21]*3
delay_after_block = [rr_duration-delay-n_rf_pulses_per_block*tr for delay in delay_after_block]
epg_mrf_fisp = EpgMrfFispWithPreparation(flip_angles, rf_phases, te, tr, inv_prep_ti, t2_prep_te, n_rf_pulses_per_block, delay_after_block)
(signal_dictionary,) = epg_mrf_fisp.forward(m0, t1, t2)


signal_dictionary = rearrange(signal_dictionary[dyn_idx.flatten(),...], '(other k1) t->other t k1', k1=dyn_idx.shape[-1])
signal_dictionary = torch.mean(signal_dictionary, dim=-1)
signal_dictionary = signal_dictionary.abs()

# %%
# Normalise dictionary entries
vector_norm = torch.linalg.vector_norm(signal_dictionary, dim=0)
signal_dictionary /= vector_norm

# %%
# Dictionary matching
n_y, n_x = img.shape[-2:]
dot_product = torch.mm(rearrange(img.abs(), 'other y x->(y x) other'), signal_dictionary)
idx_best_match = torch.argmax(torch.abs(dot_product), dim=1)
t1_match = rearrange(t1[idx_best_match], '(y x)->y x', y=n_y, x=n_x)
t2_match = rearrange(t2[idx_best_match], '(y x)->y x', y=n_y, x=n_x)


# %%
fig, ax = plt.subplots(1,3, figsize=(14,4))
ax[0].imshow(torch.squeeze(avg_im.rss()), cmap='grey')
ax[0].set_title('Average image')

im = ax[1].imshow(t1_match * mask, vmin=0, vmax=1800)
ax[1].set_title('MRF T1 (ms)')
fig.colorbar(im, extend='both', shrink=0.9, ax=ax[1])

im = ax[2].imshow(t2_match * mask, vmin=0, vmax=200)
ax[2].set_title('MRF T2 (ms)')
fig.colorbar(im, extend='both', shrink=0.9, ax=ax[2])
plt.tight_layout()




# %%
# Read in reference scans
#dcm = pydicom.dcmread(fname_t1_ref)
#t1_ref = np.asarray(dcm.pixel_array.astype(np.float32))[:,::-1]

dcm = pydicom.dcmread(fname_t2_ref)
t2_ref = np.asarray(dcm.pixel_array.astype(np.float32))[:,::-1]/10 # t2 maps are always scaled

# %%
fig, ax = plt.subplots(2,2, figsize=(6,4))
#im = ax[0,0].imshow(t1_ref, vmin=0, vmax=2000)
#ax[0,0].set_title('MOLLI (ms)')
#fig.colorbar(im, extend='both', shrink=0.9, ax=ax[0,0])

im = ax[1,0].imshow(t2_ref, vmin=0, vmax=200)
ax[1,0].set_title('T2prep (ms)')
fig.colorbar(im, extend='both', shrink=0.9, ax=ax[1,0])

im = ax[0,1].imshow(t1_match, vmin=0, vmax=2000)
ax[0,1].set_title('MRF T1 (ms)')
fig.colorbar(im, extend='both', shrink=0.9, ax=ax[0,1])

im = ax[1,1].imshow(t2_match, vmin=0, vmax=200)
ax[1,1].set_title('MRF T2 (ms)')
fig.colorbar(im, extend='both', shrink=0.9, ax=ax[1,1])
plt.tight_layout()

# %%
