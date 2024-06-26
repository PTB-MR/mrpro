# %% [markdown]
# # Cardiac Magnetic Resonance Fingerprinting

# %%
# Imports
import shutil
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import numpy as np
import pydicom
from einops import rearrange
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore [import-untyped]
from mrpro.algorithms.optimizers import adam
from mrpro.data import IData
from mrpro.operators import MagnitudeOp
from mrpro.operators.functionals import MSEDataDiscrepancy
from mrpro.operators.models import InversionRecovery
from mrpro.data import KData
from mrpro.data import KTrajectory
from mrpro.data.traj_calculators import KTrajectoryIsmrmrd
from mrpro.data.traj_calculators import KTrajectoryPulseq
from mrpro.algorithms.reconstruction import DirectReconstruction
from mrpro.operators.models import EpgMrfFispWithPreparation

fname_t1_ref = Path('/echo/_allgemein/projects/pulseq/measurements/2024-06-24_t1mes_cMRF_pulseq/meas_MID00033_FID05475_t1map_T1MES/dicom/170734_t1map_T1MES_MOCO_T1_0005/SCHUENKE_PHANTOM-0005-0001.dcm')
fname_t2_ref = Path('/echo/_allgemein/projects/pulseq/measurements/2024-06-24_t1mes_cMRF_pulseq/meas_MID00039_FID05481_t2map_flash_T1MES/dicom/170854_t2map_flash_T1MES_MOCO_T2_0009/SCHUENKE_PHANTOM-0009-0001.dcm')

pname = Path('/echo/_allgemein/projects/pulseq/measurements/2024-06-24_t1mes_cMRF_pulseq/')

scan_name = Path('meas_MID00027_FID05469_20240624_spiral_cMRF_705rep_trig_delay800ms_fov/meas_MID00027_FID05469_20240624_spiral_cMRF_705rep_trig_delay800ms_fov_with_traj.h5')
scan_name = Path('meas_MID00026_FID05468_20240624_spiral_cMRF_705rep_trig_delay800ms/meas_MID00026_FID05468_20240624_spiral_cMRF_705rep_trig_delay800ms_with_traj.h5')
scan_name = Path('meas_MID00019_FID05461_20240624_spiral_cMRF_705rep_trig_delay800ms/meas_MID00019_FID05461_20240624_spiral_cMRF_705rep_trig_delay800ms_with_traj.h5')


# %%
# Image reconstruction
trajectory = KTrajectoryIsmrmrd()
kdata = KData.from_file(pname / scan_name, trajectory)
kdata.header.recon_matrix.x = 192
kdata.header.recon_matrix.y = 192

reconstruction = DirectReconstruction.from_kdata(kdata)
img = torch.squeeze(reconstruction(kdata).data)

fig, ax = plt.subplots(1,3, figsize=(12,4))
ax[0].imshow(img[0,:,:].abs())
ax[0].set_title('First image')
ax[1].imshow(torch.mean(img[:44,:,:].abs(),dim=0))
ax[1].set_title('Average over first 44 images')
ax[2].imshow(torch.mean(img[:,:,:].abs(),dim=0))
ax[2].set_title('Average over all images')

# %%
# Dictionary settings
t1 = torch.linspace(200, 3000, 40)[:,None]/1000
t2 = torch.linspace(50, 300, 40)[None,:]/1000
t1, t2 = torch.broadcast_tensors(t1, t2)
t1 = t1.flatten()
t2 = t2.flatten()
m0 = torch.ones_like(t1)

# %%
# Dictionary calculationg
flip_angles = 
rf_phases = 0
te = 
tr = 
inv_prep_ti = [20,None,None,None,None]*3 # 20 ms delay after inversion pulse in block 0
te_prep_te = [None,None,30,50,100]*3 # T2-preparation pulse with TE = 30, 50, 100
n_rf_pulses_per_block = 47 # 47 RF pulses in each block
delay_after_block = [1000, 970, 950, 900, 980]*3
epg_mrf_fisp = EpgMrfFispWithPreparation(flip_angles=flip_angles, rf_phases=rf_phases, te=te, tr=tr)
(signal_dictionary,) = epg_mrf_fisp.forward(m0, t1, t2)

# %%
# Normalise dictionary entries
vector_norm = torch.linalg.vector_norm(signal_dictionary, dim=0)
signal_dictionary /= vector_norm

# %%
# Dictionary matching
n_y, n_x = img.shape[-2:]
dot_product = torch.mm(rearrange(img, 'other 1 z y x->(z y x) other'), signal_dictionary)
idx_best_match = torch.argmax(torch.abs(dot_product), dim=1)
t1_match = rearrange(t1[idx_best_match], '(y x)->y x', y=n_y, x=n_x)
t2_match = rearrange(t2[idx_best_match], '(y x)->y x', y=n_y, x=n_x)

# %%
# Read in reference scans
dcm = pydicom.dcmread(fname_t1_ref)
t1_ref = np.asarray(dcm.pixel_array.astype(np.float32))

dcm = pydicom.dcmread(fname_t2_ref)
t2_ref = np.asarray(dcm.pixel_array.astype(np.float32))/10 # t2 maps are always scaled

# %%
fig, ax = plt.subplots(2,2, figsize=(8,4))
ax[0,0].imshow(t1_ref, vmin=0, vmax=2000)
ax[1,0].imshow(t2_ref, vmin=0, vmax=200)
ax[0,1].imshow(t1_match, vmin=0, vmax=2000)
ax[1,1].imshow(t2_match, vmin=0, vmax=200)
# %%
