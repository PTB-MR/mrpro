# %%

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from einops import rearrange
from mrpro.data import KData
from mrpro.data.traj_calculators import KTrajectoryCartesian
from mrpro.operators import FastFourierOp
from mrpro.operators.models import EpgMrfFispWithPreparation

# %%
pname = Path(
    '/echo/_allgemein/projects/pulseq/measurements/2024-10-16_T1MES_cMRF/meas_MID00043_FID06421_20240716_cMRF_1D_705rep_trig_delay800ms/'
)
scan_name = Path('meas_MID00043_FID06421_20240716_cMRF_1D_705rep_trig_delay800ms.h5')
rr_duration = 1200

fname_angle = Path('/echo/_allgemein/projects/pulseq/mrf/cMRF_fa_705rep.txt')
with open(fname_angle) as file:
    fa = torch.as_tensor([float(line) for line in file.readlines()]) / 180 * torch.pi


trajectory = KTrajectoryCartesian()
kdata = KData.from_file(pname / scan_name, trajectory)

fft_1d_op = FastFourierOp(dim=(-1,))
(idat,) = fft_1d_op(kdata.data)
idat = torch.abs(torch.squeeze(idat[:, 0, ...]))

# %%
plt.figure(figsize=(12, 16))
plt.imshow(idat)

# %%

t1_ref = [850, 1718, 1100, 500]
t2_ref = [155, 270, 240, 80]

# Select tubes
tube1 = torch.mean(idat[:, 70:75], dim=1)
tube2 = torch.mean(idat[:, 85:90], dim=1)
tube3 = torch.mean(idat[:, 105:110], dim=1)
tube4 = torch.mean(idat[:, 120:125], dim=1)
signal_curves = torch.stack((tube1, tube2, tube3, tube4), dim=0)
signal_curves = rearrange(signal_curves, 'x t->t x')

plt.figure()
plt.plot(signal_curves)


# %%
# Dictionary settings
t1 = torch.linspace(100, 3000, 100)[:, None]
t2 = torch.linspace(10, 400, 100)[None, :]
t1, t2 = torch.broadcast_tensors(t1, t2)
t1 = t1.flatten()
t2 = t2.flatten()
m0 = torch.ones_like(t1)

# %%
flip_angles = fa
rf_phases = 0
te = 2.2
tr = 5.4  # kdata.header.tr * 1000
inv_prep_ti = [21, None, None, None, None] * 3  # 20 ms delay after inversion pulse in block 0
t2_prep_te = [None, None, 30, 50, 100] * 3  # T2-preparation pulse with TE = 30, 50, 100
n_rf_pulses_per_block = 47  # 47 RF pulses in each block
delay_after_block = [0, 30, 50, 100, 21] * 3
delay_after_block = [rr_duration - delay for delay in delay_after_block]
epg_mrf_fisp = EpgMrfFispWithPreparation(
    flip_angles, rf_phases, te, tr, inv_prep_ti, t2_prep_te, n_rf_pulses_per_block, delay_after_block
)
(signal_dictionary,) = epg_mrf_fisp.forward(m0, t1, t2)

signal_dictionary = signal_dictionary.abs()

# Normalize dictionary entries
vector_norm = torch.linalg.vector_norm(signal_dictionary, dim=0)
signal_dictionary /= vector_norm

# %%
t1_eval = torch.as_tensor([100, 500, 1000, 100, 500, 1000])
t2_eval = torch.as_tensor([40, 40, 40, 100, 100, 100])
m0_eval = torch.ones_like(t1_eval)
(signal_eval,) = epg_mrf_fisp.forward(m0_eval, t1_eval, t2_eval)

fig, ax = plt.subplots(2, 1, figsize=(12, 6))
ax[0].plot(fa / torch.pi * 180)
ax[1].plot(signal_eval.abs(), label=['100|40', '500|40', '1000|40', '100|100', '500|100', '1000|100'])
ax[1].legend()

# %%
# Dictionary matching
dot_product = torch.mm(rearrange(signal_curves.abs(), 'other x->x other'), signal_dictionary)
idx_best_match = torch.argmax(torch.abs(dot_product), dim=1)
t1_match = t1[idx_best_match]
t2_match = t2[idx_best_match]
signal_best_match = signal_dictionary[:, idx_best_match]

# %%
# Calc ref signals
if t1_ref is not None and t2_ref is not None:
    m0_ref = [1] * len(t1_ref)
    (signal_ref,) = epg_mrf_fisp.forward(torch.as_tensor(m0_ref), torch.as_tensor(t1_ref), torch.as_tensor(t2_ref))
    signal_ref = torch.abs(signal_ref)


# %%
sig_max = signal_curves.max() * 1.2
import matplotlib.pyplot as plt

fig, ax = plt.subplots(len(t2_match), 1, figsize=(12, 12))
for ind in range(len(t2_match)):
    ax[ind].set_title(
        f'T2: {int(t2_match[ind])} | T1: {int(t1_match[ind])} (GT: T2: {int(t2_ref[ind])} | T1: {int(t1_ref[ind])})'
    )
    p = ax[ind].plot(signal_curves[:, ind] / signal_curves[:, ind].max(), '-o')
    ax[ind].plot(signal_best_match[:, ind] / signal_best_match[:, ind].max(), '--', color=p[0].get_color())
    ax[ind].plot(signal_ref[:, ind] / signal_ref[:, ind].max(), '--k')
    ax[ind].set_ylabel('Signal')
# %%
