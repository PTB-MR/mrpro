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
from mrpro.data import DcfData
from mrpro.operators import FourierOp, SensitivityOp
from mrpro.utils import split_idx
from mrpro.data import KData, DcfData, IData, CsmData
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


pname = Path('/echo/_allgemein/projects/pulseq/measurements/2024-07-10_t2prep_tests/meas_MID00018_FID06326_20240710_spiral_cMRF_705rep_trig_delay800ms/')
scan_name = Path('meas_MID00018_FID06326_20240710_spiral_cMRF_705rep_trig_delay800ms_with_traj.h5')


pname = Path('/echo/_allgemein/projects/pulseq/measurements/2024-10-16_T1MES_cMRF/meas_MID00069_FID06447_20240716_spiral_cMRF_no_vds_705rep_trig_delay800ms/')
scan_name = Path('meas_MID00069_FID06447_20240716_spiral_cMRF_no_vds_705rep_trig_delay800ms_with_traj.h5')
rr_duration = 1200 

#pname = Path('/echo/_allgemein/projects/pulseq/measurements/2024-10-16_T1MES_cMRF/meas_MID00070_FID06448_20240716_spiral_cMRF_no_vds_705rep_trig_delay800ms/')
#scan_name = Path('meas_MID00070_FID06448_20240716_spiral_cMRF_no_vds_705rep_trig_delay800ms_with_traj.h5')
#rr_duration = 1200 
   
#pname = Path('/echo/_allgemein/projects/pulseq/measurements/2024-10-16_T1MES_cMRF/meas_MID00021_FID06399_20240710_spiral_cMRF_705rep_trig_delay800ms/')
#scan_name = Path('meas_MID00021_FID06399_20240710_spiral_cMRF_705rep_trig_delay800ms_with_traj.h5')
#rr_duration = 1000  

pname = Path('/echo/_allgemein/projects/pulseq/measurements/2024-10-16_T1MES_cMRF/meas_MID00074_FID06452_20240716_radial_cMRF_with_prep_705rep_trig_delay800ms/')
scan_name = Path('meas_MID00074_FID06452_20240716_radial_cMRF_with_prep_705rep_trig_delay800ms_with_traj.h5')
rr_duration = 1200 


pname = Path('/echo/_allgemein/projects/pulseq/measurements/2024-10-17_T1MES_cMRF/meas_MID00049_FID06498_20240717_spiral_cMRF_705rep_trig_delay800ms/')
scan_name = Path('meas_MID00049_FID06498_20240717_spiral_cMRF_705rep_trig_delay800ms_with_traj.h5')
rr_duration = 1200 

pname = Path('/echo/_allgemein/projects/pulseq/measurements/2024-10-17_T1MES_cMRF/meas_MID00048_FID06497_20240717_radial_cMRF_705rep_trig_delay800ms/')
scan_name = Path('meas_MID00048_FID06497_20240717_radial_cMRF_705rep_trig_delay800ms_with_traj.h5')
rr_duration = 1200 
     
fname_angle = Path('/echo/_allgemein/projects/pulseq/mrf/cMRF_fa_705rep.txt')
with open(fname_angle, "r") as file:
    fa = torch.as_tensor([float(line) for line in file.readlines()])/180 * torch.pi

# %%
# Image reconstruction
trajectory = KTrajectoryIsmrmrd()
#trajectory = KTrajectoryPulseq(pname / '20240716_radial_cMRF_lin_label_705rep_trig-delay800ms.seq')
kdata = KData.from_file(pname / scan_name, trajectory)
kdata.header.recon_matrix.x = 192
kdata.header.recon_matrix.y = 192

#traj_minus_first_point = KTrajectory(kz=kdata.traj.kz, ky=kdata.traj.ky-kdata.traj.ky[:,:,:,0,None], kx=kdata.traj.kx-kdata.traj.kx[:,:,:,0,None])
#kdata = KData(header=kdata.header, data=kdata.data, traj=traj_minus_first_point)

# Voronoi dcf
#ktraj_all_rep = KTrajectory(kz=kdata.traj.kz, ky=rearrange(kdata.traj.ky, 'other k2 k1 k0->k2 k1 other k0'), kx=rearrange(kdata.traj.kx, 'other k2 k1 k0->k2 k1 other k0'))
#dcf = DcfData.from_traj_voronoi(ktraj_all_rep)
#dcf_data_voronoi = rearrange(dcf.data, 'k2 k1 other k0->other k2 k1 k0')
#dcf_data_voronoi[dcf_data_voronoi < 1e-2] = 1e-2

dcf = DcfData.from_traj_voronoi(kdata.traj)
dcf_data_voronoi = dcf.data

fourier_op = FourierOp.from_kdata(kdata)
(img,) = fourier_op.adjoint(kdata.data * dcf_data_voronoi[:,None,...])
idata = IData.from_tensor_and_kheader(img, kdata.header)
csm_data = CsmData.from_idata_walsh(idata)
csm_op = SensitivityOp(csm_data)

dyn_idx = split_idx(torch.arange(0,47), 10, 8)
dyn_idx = torch.cat([dyn_idx + ind*47 for ind in range(15)], dim=0)

kdata = kdata.split_k1_into_other(dyn_idx, other_label='repetition')
dcf_data_voronoi = rearrange(dcf_data_voronoi, 'k2 k1 other k0->other k2 k1 k0')
dcf_data_voronoi = rearrange(dcf_data_voronoi[dyn_idx.flatten(),...], '(other k1) 1 k2 k0->other k2 k1 k0', k1=dyn_idx.shape[-1])
if False:
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


    kdata_all_rep = rearrange(kdata.data, 'other coils k2 k1 k0->k2 coils k1 other k0')
    kdata_pipe = KData(header=kdata.header, data=kdata_all_rep, traj=ktraj_all_rep)
    dcf_data_pipe=torch.ones(*kdata_pipe.data.shape[:-4],1,*kdata_pipe.data.shape[-3:])
    F=FourierOp.from_kdata(kdata_pipe)
    for i in range(10):
        new = (F@F.H)(dcf_data_pipe+0j)[0]
        print("loss",(new-1).abs().mean())
        dcf_data_pipe/=new.abs().sqrt()
    #dcf_data_pipe = DcfData(dcf_pipe.squeeze(-4))
    dcf_data_pipe = dcf_data_pipe[:,0,...]
    dcf_data_pipe = rearrange(dcf_data_pipe, 'k2 k1 other k0->other k2 k1 k0')

    plt.figure()
    plt.plot(dcf_data_voronoi[0,0,0,:], '-r')
    plt.plot(dcf_data_voronoi[100,0,0,:], '-b')
    plt.plot(dcf_data[0,0,0,:], '--r')
    plt.plot(dcf_data[100,0,0,:], '--b')
    plt.plot(dcf_data_pipe[0,0,0,:], ':r')
    plt.plot(dcf_data_pipe[100,0,0,:], ':b')

fourier_op = FourierOp.from_kdata(kdata)
(img,) = fourier_op.adjoint(kdata.data * dcf_data_voronoi[:,None,...])
#img = torch.sum(img,dim=1)
(img,) = csm_op.adjoint(img)
img = torch.squeeze(img)

fig, ax = plt.subplots(1,3, figsize=(12,4))
ax[0].imshow(img[0,:,:].abs())
ax[0].set_title('First image')
ax[1].imshow(torch.abs(torch.mean(img[:47,:,:],dim=0)))
ax[1].set_title('Average over first 47 images')
ax[2].imshow(torch.abs(torch.mean(img[:,:,:],dim=0)))
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
inv_prep_ti = [21,None,None,None,None]*3 # 20 ms delay after inversion pulse in block 0
t2_prep_te = [None,None,30,50,100]*3 # T2-preparation pulse with TE = 30, 50, 100
n_rf_pulses_per_block = 47 # 47 RF pulses in each block
delay_after_block = [0, 30, 50, 100, 21]*3
delay_after_block = [rr_duration-delay for delay in delay_after_block]
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
t1_ref = np.asarray(dcm.pixel_array.astype(np.float32))[:,::-1]

dcm = pydicom.dcmread(fname_t2_ref)
t2_ref = np.asarray(dcm.pixel_array.astype(np.float32))[:,::-1]/10 # t2 maps are always scaled

# %%
fig, ax = plt.subplots(2,2, figsize=(6,4))
im = ax[0,0].imshow(t1_ref, vmin=0, vmax=2000)
ax[0,0].set_title('MOLLI (ms)')
fig.colorbar(im, extend='both', shrink=0.9, ax=ax[0,0])

im = ax[1,0].imshow(t2_ref, vmin=0, vmax=200)
ax[1,0].set_title('T2prep (ms)')
fig.colorbar(im, extend='both', shrink=0.9, ax=ax[1,0])

im = ax[0,1].imshow(t1_match[40:160,40:160], vmin=0, vmax=2000)
ax[0,1].set_title('MRF T1 (ms)')
fig.colorbar(im, extend='both', shrink=0.9, ax=ax[0,1])

im = ax[1,1].imshow(t2_match[40:160,40:160], vmin=0, vmax=200)
ax[1,1].set_title('MRF T2 (ms)')
fig.colorbar(im, extend='both', shrink=0.9, ax=ax[1,1])
plt.tight_layout()
# %%
