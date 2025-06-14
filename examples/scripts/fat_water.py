#%%
from mrpro.data import KData, CsmData
from mrpro.data.traj_calculators import KTrajectoryCartesian
from mrpro.algorithms.reconstruction import DirectReconstruction, IterativeSENSEReconstruction
from mrpro.operators import FourierOp, AveragingOp, GridSamplingOp, FastFourierOp
import torch
import ismrmrd
import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from einops import rearrange
from mrpro.algorithms.optimizers import cg
import time

pname = '/echo/kolbit01/data/FatWater/Robert_3_6_25/mf_expreg_mirtk15_0.001_0.001_20240815-2_58816_4_0.0_6_echo2_under1.0_false2_middle_new4/'
fraw = 'meas_MID00058_FID01550_5min_3accel20_1_16seg.h5'
fmf = 'mf_expreg_mirtk15_0.001_0.001_20240815-2_58816_4_0.0_6_echo2_under1.0_false2_middle_new4.npy'
fidx = 'splits4_0.0_20240815-2_58816_1.0.npy'
fidx_ms = 'SplitDynSelIdx_20240815-2_58816_4_0.0.npy'
fcsm = 'csm_largeDims.npy'

puncorr_robert = pname + 'CG SENSE without moco/cg5_20240815-2_58816_under1.0_false2_CSMinati.npy'

flag_cuda = False

def get_slice(img):
    idim = img.shape
    return np.rot90(np.squeeze(img)[idim[-3]//2, ::-1, :], 1)

def _norm_squared(value: torch.Tensor) -> torch.Tensor:
    return torch.vdot(value.flatten(), value.flatten()).real

# %%
from mrpro.algorithms.optimizers.pdhg import PDHGStatus
global prev_solution_tv
prev_solution_tv = torch.zeros(1,2,1,96,360,304, dtype=torch.complex64)

def tv_callback(optimizer_status: PDHGStatus) -> None:
    """Print the value of the objective functional every 16th iteration."""
    iteration = optimizer_status['iteration_number']
    solution = optimizer_status['solution']
    print(f'Iteration {iteration: >3}: Objective = {optimizer_status["objective"](*solution).item():.3e}')
    change = torch.sqrt(_norm_squared(solution[0].cpu() - prev_solution_tv))/ torch.sqrt(_norm_squared(prev_solution_tv))
    if change < 2e-3:
        print(f'Convergence reached at iteration {iteration} with change {change:.3e}')
        return False
    else:
        print(f'Change {change:.3e}')
        prev_solution_tv.copy_(solution[0].cpu())
        return True

from mrpro.data import IData
from mrpro.algorithms.optimizers import pdhg
from mrpro.operators import (
    FiniteDifferenceOp,
    FourierOp,
    LinearOperatorMatrix,
    ProximableFunctionalSeparableSum,
    SensitivityOp,
)
from mrpro.operators.functionals import L1NormViewAsReal, L2NormSquared, ZeroFunctional
from mrpro.utils import unsqueeze_right
def tv_reg_reco(kdata, csm, img_initial=None, motion_op=None, reg_weight=0.1, reg_weight_t=0.1, n_iterations=100):
    fourier_operator = FourierOp.from_kdata(kdata)
    csm_operator = SensitivityOp(csm)
    if motion_op is not None:
        averaging_op = AveragingOp(dim=0, domain_size=kdata.shape[0])
        acquisition_operator = fourier_operator @ motion_op @ csm_operator @ averaging_op.H
    else:
        acquisition_operator = fourier_operator @ csm_operator
        
    if img_initial is None:
        (img_initial,) = acquisition_operator.H(kdata.data)
        

    if img_initial.data.shape[0] == 1:
        tv_dim = (-3, -2, -1)
        regularization_weight = torch.tensor([reg_weight, reg_weight, reg_weight])
    else:
        tv_dim = (-5, -3, -2, -1)
        regularization_weight = torch.tensor([reg_weight_t, reg_weight, reg_weight, reg_weight])

    nabla_operator = FiniteDifferenceOp(dim=tv_dim, mode='forward')
    l2 = 0.5 * L2NormSquared(target=kdata.data)
    l1 = L1NormViewAsReal(weight=unsqueeze_right(regularization_weight, kdata.data.ndim))

    f = ProximableFunctionalSeparableSum(l2, l1)
    g = ZeroFunctional()
    K = LinearOperatorMatrix(((acquisition_operator,), (nabla_operator,)))

    initial_values = (img_initial.data.clone(),)

    (img_pdhg,) = pdhg(f=f, g=g, operator=K, initial_values=initial_values, 
                       max_iterations=n_iterations, callback=tv_callback)
    return IData(data=img_pdhg, header=img_initial.header)


# %%
from mrpro.algorithms.optimizers.pgd import PGDStatus
global prev_solution_wavelet 
prev_solution_wavelet = torch.zeros(1,2,2,12011526, dtype=torch.complex64)

def wavelet_status(optimizer_status: PGDStatus) -> bool:
        iteration = optimizer_status['iteration_number']
        solution = optimizer_status['solution']
        print(f'{iteration}: {optimizer_status["objective"](*solution).item()}, stepsize: {optimizer_status["stepsize"]}')
        change = torch.sqrt(_norm_squared(solution[0].cpu() - prev_solution_wavelet))/ torch.sqrt(_norm_squared(prev_solution_wavelet))
        if change < 2e-3:
            print(f'Convergence reached at iteration {iteration} with change {change:.3e}')
            return False
        else:
            print(f'Change {change:.3e}')
            prev_solution_wavelet.copy_(solution[0].cpu())
            return True
        
       
from mrpro.algorithms.optimizers import pgd     
def wavelet_reg_reco(kdata, csm, img_initial=None, motion_op=None, reg_weight=0.1, n_iterations=100):
    fourier_operator = FourierOp.from_kdata(kdata)
    csm_operator = SensitivityOp(csm)
    if motion_op is not None:
        averaging_op = AveragingOp(dim=0, domain_size=kdata.shape[0])
        acquisition_operator = fourier_operator @ motion_op @ csm_operator @ averaging_op.H
    else:
        acquisition_operator = fourier_operator @ csm_operator
        
    if img_initial is None:
        (img_initial,) = acquisition_operator.H(kdata.data)
        

    from mrpro.operators import WaveletOp
    wavelet_operator = WaveletOp(
        domain_shape=img_initial.data.shape[-3:], dim=(-3, -2, -1), wavelet_name='db4', level=None
    )
    initial_values = wavelet_operator(img_initial.data)
    acquisition_operator = acquisition_operator @ wavelet_operator.H
    
    op_norm = acquisition_operator.operator_norm(
        initial_value=torch.randn_like(initial_values[0]), dim=(-3, -2, -1), max_iterations=12
    ).item()
    stepsize = 0.9 * (1 / op_norm**2)

    l2 = 0.5 * L2NormSquared(target=kdata.data)
    l1 = L1NormViewAsReal()
    f = l2 @ acquisition_operator
    g = reg_weight * l1

    (img_wave_pgd,) = pgd(
        f=f,
        g=g,
        initial_value=initial_values,
        stepsize=stepsize,
        max_iterations=n_iterations,
        backtrack_factor=1.0,
        convergent_iterates_variant=True,
        callback=wavelet_status,
    )

    # map the solution back to image domain
    (img_pgd,) = wavelet_operator.H(img_wave_pgd)
    return IData(data=img_pgd, header=img_initial.header)

# %%
kdata = KData.from_file(pname+fraw, KTrajectoryCartesian())
kdata.header.recon_matrix.x = 304
kdata.header.recon_matrix.y = 360
kdata.header.recon_matrix.z = 96
kdata = kdata.remove_readout_os()

# %% Reuse CSM
rec = DirectReconstruction(kdata, csm=None)
img_avg = rec(kdata)

csm_robert = np.load(pname + fcsm, allow_pickle=True)
csm_robert = rearrange(np.squeeze(csm_robert), 'x y z coil -> coil z y x')
csm = CsmData(data=torch.as_tensor(csm_robert, dtype=torch.complex64), header=img_avg.header[0,...])


# %%
rec = IterativeSENSEReconstruction(kdata, csm=csm, n_iterations=6)
img_avg = rec(kdata)
idat = img_avg.data.cpu().numpy()
idat /= np.abs(idat).max()

# %%
idat_robert = np.load(puncorr_robert, allow_pickle=True)
idat_robert /= np.abs(idat_robert).max()
idat_robert = rearrange(np.squeeze(idat_robert), 'x y z echo -> echo z y x')

#%%
fig, ax = plt.subplots(2,4, figsize=(12,6))
idim = idat.shape
for jnd in range(2):
    ax[0,2*jnd].imshow(np.abs(get_slice(idat[jnd,...])), cmap='gray', vmin=0, vmax=0.4)
    ax[1,2*jnd].imshow(np.angle(get_slice(idat[jnd,...])), cmap='bwr', vmin=-3, vmax=3)
    ax[0,2*jnd+1].imshow(np.abs(get_slice(idat_robert[jnd,...])), cmap='gray', vmin=0, vmax=0.4)
    ax[1,2*jnd+1].imshow(np.angle(get_slice(idat_robert[jnd,...])), cmap='bwr', vmin=-3, vmax=3)
    
#%%
time_idx = torch.argsort(torch.squeeze(kdata.header.acq_info.acquisition_time_stamp[0,...]))
kdata = kdata[..., time_idx, :, :]

    
# %%
motion_idx = np.load(pname + fidx, allow_pickle=True)
n_idx = min([len(idx) for idx in motion_idx])
motion_idx = [torch.as_tensor(idx[:n_idx]) for idx in motion_idx]
motion_idx = torch.stack(motion_idx, dim=0)
kdata_resp_resolved = kdata[..., motion_idx, :, :]
recon_resp_resolved = IterativeSENSEReconstruction(kdata_resp_resolved[:,0,...], csm=csm, n_iterations=10)
img_resp_resolved = recon_resp_resolved(kdata_resp_resolved[:,0,...])


#%%
idat = img_resp_resolved.data.cpu().numpy()
idat /= np.abs(idat).max()
fig, ax = plt.subplots(2,4, figsize=(12,9))
idim = idat.shape
for jnd in range(4):
    ax[0,jnd].imshow(np.abs(get_slice(idat[jnd,...])), cmap='gray', vmin=0, vmax=0.5)
    ax[1,jnd].imshow(np.abs(np.abs(get_slice(idat[jnd,...]))-np.abs(get_slice(idat[0,...]))), cmap='gray', vmin=0, vmax=0.5)
    
    
# %%
mf = torch.as_tensor(np.load(pname + fmf), dtype=torch.float32)
mf = rearrange(mf, 'x y z dim ms bwd_fwd -> ms bwd_fwd z y x dim')[:,1,None,...]
motion_op = GridSamplingOp.from_displacement(mf[..., 2], mf[..., 1], mf[..., 0])


# %%
fourier_op = FourierOp.from_kdata(kdata_resp_resolved)
dcf_op = recon_resp_resolved.dcf.as_operator()
csm_op = SensitivityOp(csm)
averaging_op = AveragingOp(dim=0)
acquisition_operator = fourier_op @ motion_op @ csm_op @ averaging_op.H

(initial_value,) = acquisition_operator.H(dcf_op(kdata_resp_resolved.data)[0])
(right_hand_side,) = acquisition_operator.H(kdata_resp_resolved.data)
operator = acquisition_operator.H @ acquisition_operator

if flag_cuda:
    kdata_resp_resolved = kdata_resp_resolved.cuda()
    initial_value = initial_value.cuda()
    right_hand_side = right_hand_side.cuda()
    operator = operator.cuda()
    motion_op = motion_op.cuda()
    csm = csm.cuda()


# %% MCIR iterative SENSE
tstart = time.time()
(img_mcir,) = cg(operator, right_hand_side, initial_value=right_hand_side, max_iterations=5, tolerance=0.0)
print(f'Time for MCIR: {(time.time() - tstart)/60:.2f} min')

# %% MCIR TV regularization
tstart = time.time()
img_mcir_tv = tv_reg_reco(kdata_resp_resolved, csm, img_initial=IData.from_tensor_and_kheader(img_mcir, kdata_resp_resolved.header[0,...]), 
                         motion_op=motion_op, reg_weight=4e-8, n_iterations=100)
print(f'Time for MCIR TV regularization: {(time.time() - tstart)/60:.2f} min')

# %% MCIR wavelet regularization
tstart = time.time()
img_mcir_wavelet = wavelet_reg_reco(kdata_resp_resolved, csm, img_initial=IData.from_tensor_and_kheader(img_mcir, kdata_resp_resolved.header[0,...]),
                                   motion_op=motion_op, reg_weight=2e-7, n_iterations=100)
print(f'Time for MCIR wavelet regularization: {(time.time() - tstart)/60:.2f} min')

# %%
#idat = [img_avg.data.cpu().numpy(), img_mcir.cpu().numpy(), img_mcir_tv.data.cpu().numpy(), img_mcir_wavelet.data.cpu().numpy()]
idat = [img_avg.data.cpu().numpy(), img_mcir.cpu().numpy(), img_mcir_tv.data.cpu().numpy(), img_mcir_wavelet.data.cpu().numpy()]
idat  = [np.squeeze(i)/np.abs(i).max() for i in idat]
fig, ax = plt.subplots(4,len(idat), figsize=(9*len(idat),36))
idim = idat[0].shape
for jnd in range(2):
    for knd in range(len(idat)):
        ax[0+2*jnd,knd].imshow(np.abs(get_slice(idat[knd][jnd,...])), cmap='gray', vmin=0, vmax=0.5)
        ax[1+2*jnd,knd].imshow(np.angle(get_slice(idat[knd][jnd,...])), cmap='bwr', vmin=-3, vmax=3)
    
# %%
np.save(pname + 'mrpro_mcir_tv.npy', img_mcir_tv.data.cpu().numpy())
np.save(pname + 'mrpro_mcir_wavelet.npy', img_mcir_wavelet.data.cpu().numpy())