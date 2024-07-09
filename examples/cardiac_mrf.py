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
import scipy as sp
import pydicom
from einops import rearrange
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore [import-untyped]
from mrpro.algorithms.optimizers import adam
from mrpro.data import DcfData
from mrpro.operators import MagnitudeOp
from mrpro.operators import FourierOp
from mrpro.operators.functionals import MSEDataDiscrepancy
from mrpro.operators.models import InversionRecovery
from mrpro.data import KData
from mrpro.data import KTrajectory
from mrpro.data.traj_calculators import KTrajectoryIsmrmrd
from mrpro.data.traj_calculators import KTrajectoryRadial2D
from mrpro.data.traj_calculators import KTrajectoryPulseq
from mrpro.algorithms.reconstruction import DirectReconstruction
from mrpro.operators.models import EpgMrfFispWithPreparation

fname_t1_ref = Path('/echo/_allgemein/projects/pulseq/measurements/2024-06-24_t1mes_cMRF_pulseq/meas_MID00033_FID05475_t1map_T1MES/dicom/170734_t1map_T1MES_MOCO_T1_0005/SCHUENKE_PHANTOM-0005-0001.dcm')
fname_t2_ref = Path('/echo/_allgemein/projects/pulseq/measurements/2024-06-24_t1mes_cMRF_pulseq/meas_MID00039_FID05481_t2map_flash_T1MES/dicom/170854_t2map_flash_T1MES_MOCO_T2_0009/SCHUENKE_PHANTOM-0009-0001.dcm')

pname = Path('/echo/_allgemein/projects/pulseq/measurements/2024-06-24_t1mes_cMRF_pulseq/')
scan_name = Path('meas_MID00027_FID05469_20240624_spiral_cMRF_705rep_trig_delay800ms_fov/meas_MID00027_FID05469_20240624_spiral_cMRF_705rep_trig_delay800ms_fov_with_traj.h5')
#scan_name = Path('meas_MID00026_FID05468_20240624_spiral_cMRF_705rep_trig_delay800ms/meas_MID00026_FID05468_20240624_spiral_cMRF_705rep_trig_delay800ms_with_traj.h5')
scan_name = Path('meas_MID00019_FID05461_20240624_spiral_cMRF_705rep_trig_delay800ms/meas_MID00019_FID05461_20240624_spiral_cMRF_705rep_trig_delay800ms_with_traj.h5')


pname = Path('/echo/_allgemein/projects/pulseq/measurements/2024-06-27_cMRF_tests/')
scan_name = Path('meas_MID00143_FID05777_20240627_spiral_cMRF_705rep_only_t1prep/meas_MID00143_FID05777_20240627_spiral_cMRF_705rep_only_t1prep_with_traj.h5')


pname_new = Path('/echo/_allgemein/projects/pulseq/measurements/2024-06-27_cMRF_tests/')
scan_name_new = Path('meas_MID00143_FID05777_20240627_spiral_cMRF_705rep_only_t1prep/meas_MID00143_FID05777_20240627_spiral_cMRF_705rep_only_t1prep_with_traj.h5')

fname_angle = Path('/echo/_allgemein/projects/pulseq/mrf/cMRF_fa_705rep.txt')
with open(fname_angle, "r") as file:
    fa = torch.as_tensor([float(line) for line in file.readlines()])/180 * torch.pi

# %%
# Image reconstruction
trajectory = KTrajectoryIsmrmrd()
kdata = KData.from_file(pname / scan_name, trajectory)
kdata.header.recon_matrix.x = 192
kdata.header.recon_matrix.y = 192

kdata_new = KData.from_file(pname_new / scan_name_new, KTrajectoryRadial2D())
kdata = KData(header=kdata_new.header, data=kdata_new.data, traj=kdata.traj)

# Voronoi dcf
ktraj_all_rep = KTrajectory(kz=kdata.traj.kz, ky=rearrange(kdata.traj.ky, 'other k2 k1 k0->k2 k1 other k0'), kx=rearrange(kdata.traj.kx, 'other k2 k1 k0->k2 k1 other k0'))
dcf = DcfData.from_traj_voronoi(ktraj_all_rep)
dcf_data_voronoi = rearrange(dcf.data, 'k2 k1 other k0->other k2 k1 k0')
dcf_data_voronoi[dcf_data_voronoi < 1e-2] = 1e-2

# Analytical dcf calculation following 
# Hoge, R.D., Kwan, R.K.S. and Bruce Pike, G. (1997), Density compensation functions for spiral MRI. Magn. Reson. Med., 
# 38: 117-128. https://doi.org/10.1002/mrm.1910380117
ktraj_tensor = kdata.traj.as_tensor()
ktraj_tensor_norm = torch.linalg.norm(ktraj_tensor[-2:,...], dim=0)
ktraj_tensor_angle = torch.atan2(ktraj_tensor[-2,...], ktraj_tensor[-1,...])
gradient_tensor = torch.diff(ktraj_tensor, dim=-1)
gradient_tensor = torch.cat([gradient_tensor, gradient_tensor[...,-1,None]], dim=-1)
gradient_tensor_norm = torch.linalg.norm(gradient_tensor[-2:,...], dim=0)
gradient_tensor_angle = torch.atan2(gradient_tensor[-2,...], gradient_tensor[-1,...])
dcf_data = gradient_tensor_norm * ktraj_tensor_norm * torch.abs(torch.cos(gradient_tensor_angle - ktraj_tensor_angle))
dcf_data /= dcf_data.max()
dcf_data[dcf_data < 1e-2] = 1e-2

dcf_data2 = gradient_tensor_norm * torch.abs(torch.sin(gradient_tensor_angle - ktraj_tensor_angle))

plt.figure()
plt.plot(dcf_data_voronoi[0,0,0,:], '-r')
plt.plot(dcf_data_voronoi[100,0,0,:], '-b')
plt.plot(dcf_data[0,0,0,:], '--r')
plt.plot(dcf_data[100,0,0,:], '--b')

fourier_op = FourierOp.from_kdata(kdata)
(img,) = fourier_op.adjoint(kdata.data * dcf_data[:,None,...])
img = torch.sum(img**2,dim=1)
img = torch.squeeze(img)

fig, ax = plt.subplots(1,3, figsize=(12,4))
ax[0].imshow(img[0,:,:].abs())
ax[0].set_title('First image')
ax[1].imshow(torch.mean(img[:47,:,:].abs(),dim=0))
ax[1].set_title('Average over first 44 images')
ax[2].imshow(torch.mean(img[:,:,:].abs(),dim=0))
ax[2].set_title('Average over all images')

# %%
# Dictionary settings
t1 = torch.linspace(100, 3000, 40)[:,None]
t2 = torch.linspace(10, 300, 40)[None,:]
t1, t2 = torch.broadcast_tensors(t1, t2)
t1 = t1.flatten()
t2 = t2.flatten()
m0 = torch.ones_like(t1)

# %%
# Dictionary calculationg
flip_angles = fa
rf_phases = 0
te = 0.845
tr = kdata.header.tr * 1000
inv_prep_ti = [20,None,None,None,None]*3 # 20 ms delay after inversion pulse in block 0
t2_prep_te = [None,None,30,50,100]*3 # T2-preparation pulse with TE = 30, 50, 100
t2_prep_te = None
n_rf_pulses_per_block = 47 # 47 RF pulses in each block
delay_after_block = [1000, 970, 950, 900, 980]*3
epg_mrf_fisp = EpgMrfFispWithPreparation(flip_angles, rf_phases, te, tr, inv_prep_ti, t2_prep_te, n_rf_pulses_per_block, delay_after_block)
(signal_dictionary,) = epg_mrf_fisp.forward(m0, t1, t2)

signal_dictionary = signal_dictionary.abs()

# %%
# Normalise dictionary entries
vector_norm = torch.linalg.vector_norm(signal_dictionary, dim=0)
signal_dictionary /= vector_norm

# %%
t1_eval = torch.as_tensor([100, 500, 1000, 100, 500, 1000])
t2_eval = torch.as_tensor([40, 40, 40, 100, 100, 100])
m0_eval = torch.ones_like(t1_eval)
(signal_eval,) = epg_mrf_fisp.forward(m0_eval, t1_eval, t2_eval)

# median filtered signal
img_filt = sp.signal.medfilt(img.abs(), kernel_size = (11,1,1))

fig, ax = plt.subplots(6,1, figsize=(12,12))
ax[0].plot(fa / torch.pi * 180)
ax[1].plot(signal_eval.abs(), label=['100|40', '500|40', '1000|40', '100|100', '500|100', '1000|100'])
ax[1].legend()
ax[2].plot(img[:,105,::12].abs())
ax[3].plot(img_filt[:,105,::12])
ax[4].imshow(rearrange(img[:,105,:].abs(), 't x->x t'))
ax[5].imshow(rearrange(img_filt[:,105,:], 't x->x t'))

# %%
# Dictionary matching
n_y, n_x = img.shape[-2:]
dot_product = torch.mm(rearrange(img.abs(), 'other y x->(y x) other'), signal_dictionary)
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
