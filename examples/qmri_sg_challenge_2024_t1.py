# %%
# Imports
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from mrpro.algorithms import adam
from mrpro.data import IData
from mrpro.operators.functionals import MSEDataDiscrepancy
from mrpro.operators.models import InversionRecovery

# %% Get dicom files and create IData
pathname = Path('/echo/_allgemein/projects/8_13/MRPro/qMR_Study_Group_Challenge_2024/T1 IR/')
dicom_files = glob(str(pathname / '**/*.dcm'), recursive=True)

idata_list = [IData.from_single_dicom(dicom_file) for dicom_file in dicom_files]
ti = torch.as_tensor([idata.header.ti[0] for idata in idata_list])
idata = IData(header=idata_list[0].header, data=torch.cat([idata.data for idata in idata_list]))
idata.header.ti = ti

# %% TI images
fig, axes = plt.subplots(2, 5)
for idx, ax in enumerate(axes.flatten()):
    ax.imshow(torch.abs(idata.data[idx, 0, 0, :, :]))
    ax.set_title(f'{idata.header.ti[idx]:.0f}ms')


# %% Estimate T1 times
class InversionRecoveryAmplitude(InversionRecovery):
    def forward(self, m0: torch.Tensor, t1: torch.Tensor) -> tuple[torch.Tensor,]:
        return (torch.abs(super().forward(m0, t1)[0]),)


model = InversionRecoveryAmplitude(ti=idata.header.ti)

mse = MSEDataDiscrepancy(idata.data)

functional = mse @ model


ti_shortest_idx = torch.argmin(idata.header.ti)
m0_start = torch.abs(idata.data[ti_shortest_idx, ...])


# enable gradient calculation
m0_start.requires_grad = True

# hyperparams for optimizer
max_iter = 10
outer_iter = 200
lr = 1e0

mse_voxel_basis = []
m0_end = []
t1_end = []
plt.figure()
for t1_start_value in torch.linspace(100, 500, 10):
    t1_start = torch.ones(idata.data[0, ...].shape, dtype=torch.float32) * t1_start_value
    t1_start.requires_grad = True
    params_init = [m0_start, t1_start]

    fobj = []
    for oit in range(outer_iter):
        # estimate minimizer
        params_result = adam(
            functional,
            params_init,
            max_iter=max_iter,
            lr=lr,
        )
        fobj.append(functional(*params_result)[0].detach())
        params_init = params_result
        #
    m0_end.append(params_result[0].detach())
    t1_end.append(params_result[1].detach())
    m0_end[-1][torch.isnan(t1_end[-1])] = 0
    t1_end[-1][torch.isnan(t1_end[-1])] = 1000

    # MSE on pixel-basis
    signal_diff = model.forward(*params_result)[0].detach() - idata.data
    mse_voxel_basis.append(torch.mean(signal_diff**2, dim=0))

    plt.plot(fobj, label=f'{t1_start_value}')
    plt.ylim([0, 100000])
plt.legend()

# %%
fig, ax = plt.subplots(1, len(mse_voxel_basis))
for ind in range(len(mse_voxel_basis)):
    ax[ind].imshow(torch.abs(mse_voxel_basis[ind][0, 0, ...]), vmin=0, vmax=20000)

# %% Select best result based on MSE
mse_voxel_basis = torch.abs(torch.cat(mse_voxel_basis))
mse_min_idx = torch.argmin(mse_voxel_basis, dim=0)
m0_end.append(torch.take_along_dim(torch.cat(m0_end), dim=0, indices=mse_min_idx[None, ...]))
t1_end.append(torch.take_along_dim(torch.cat(t1_end), dim=0, indices=mse_min_idx[None, ...]))

fobj_final = torch.as_tensor([functional(m0, t1)[0].detach() for m0, t1 in zip(m0_end, t1_end, strict=False)])
print(f'Min obj of optimisation {torch.min(fobj_final[:-1]):.2f} Hybrid obj {fobj_final[-1]:.2f}')

m0_end = m0_end[-1]
t1_end = t1_end[-1]

# %% Parameters
fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(m0_start[0, 0, ...].detach())
ax[1, 0].imshow(m0_end[0, 0, ...].detach())
ax[0, 1].imshow(t1_start[0, 0, ...].detach(), vmin=0, vmax=2000)
ax[1, 1].imshow(t1_end[0, 0, ...].detach(), vmin=0, vmax=2000)

# %% TI Images
max_intensity = torch.max(torch.abs(idata.data))
(synthetic_data,) = model.forward(m0_end, t1_end)
fig, ax = plt.subplots(3, 10)
for idx in range(idata.data.shape[0]):
    ax[0, idx].imshow(torch.abs(idata.data[idx, 0, 0, :, :]), vmin=0, vmax=max_intensity)
    ax[1, idx].imshow(torch.abs(synthetic_data[idx, 0, 0, :, :].detach()), vmin=0, vmax=max_intensity)
    ax[2, idx].imshow(
        torch.abs(synthetic_data[idx, 0, 0, :, :].detach() - idata.data[idx, 0, 0, :, :]),
        vmin=0,
        vmax=max_intensity * 0.1,
    )
    ax[0, idx].set_title(f'{idata.header.ti[idx]:.0f}')

plt.show()


# %% Single voxel
voxel_positions = [
    [45, 30],
    [70, 30],
    [30, 50],
    [60, 50],
    [95, 40],
    [30, 70],
    [50, 70],
    [75, 70],
    [100, 65],
    [45, 95],
    [65, 100],
    [90, 90],
]
plt.figure()
plt.imshow(torch.abs(idata.data[idx, 0, 0, :, :]))
for pidx, pos in enumerate(voxel_positions):
    plt.plot(pos[0], pos[1], 'wx')
    plt.annotate(str(pidx), xy=pos, color='w')


model_for_plotting = InversionRecoveryAmplitude(ti=torch.linspace(idata.header.ti.min(), idata.header.ti.max(), 100))
(signal_for_plotting,) = model_for_plotting(m0_end, t1_end)

fig, ax = plt.subplots(2, 2)
tidx = torch.argsort(ti)
for aidx, ax in enumerate(ax.flatten()):
    for jnd in range(3):
        pos = voxel_positions[jnd + aidx * 3]
        p = ax.plot(ti[tidx], idata.data[tidx, 0, 0, pos[1], pos[0]], '-o', label=f'{jnd + aidx * 3}')
        ax.plot(
            model_for_plotting.ti, signal_for_plotting[:, 0, 0, pos[1], pos[0]].detach(), '--', color=p[0].get_color()
        )
    ax.set_xlabel('Time (s)')
    if aidx == 0 or aidx == 2:
        ax.set_ylabel('Signal')
    if aidx == 0:
        ax.set_title('Original (solid) and fitted (dashed) signals.', {'ha': 'left'})
    ax.legend()
# %%
