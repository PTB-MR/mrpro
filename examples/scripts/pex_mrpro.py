# %%
import matplotlib.pyplot as plt
import mrpro
import torch
from mrpro.utils.unit_conversion import GYROMAGNETIC_RATIO_PROTON, sqrt_kwatt_to_volt
from pathlib import Path

data_folder = Path('RAW')
# %%  Reco
# Read raw data and trajectory

kdata = mrpro.data.KData.from_file(
    data_folder / 'meas_MID59_PEX_slow_B1_500us_RECT_60V_FID18835.h5',
    mrpro.data.traj_calculators.KTrajectoryCartesian(),
).remove_readout_os()

csm = mrpro.data.CsmData.from_kdata_inati(kdata[12])
reco = mrpro.algorithms.reconstruction.DirectReconstruction(kdata, csm=csm)
img = reco(kdata).data
img = img.flip(dims=(0,))


def get_pex_special_tab(kdata):
    voltages = []
    for i in range(kdata.shape[0]):
        voltages.append(kdata.header._misc['userParameters']['userParameterDouble'][68 + i]['value'])

    pulse_duration = kdata.header._misc['userParameters']['userParameterDouble'][18]['value'] * 1e-6
    prep_delay = kdata.header._misc['userParameters']['userParameterDouble'][11]['value'] * 1e-6
    return voltages[::-1], pulse_duration, prep_delay


voltages, pulse_duration, prep_delay = get_pex_special_tab(kdata)
print(f'voltages: {voltages} V')
print(f'pulse_duration: {pulse_duration} s')
print(f'prep_delay: {prep_delay} s')

# %%

img_fit = img / img[0, ...]

img_abs = img_fit.abs().squeeze()
img_phase = img_fit.angle().squeeze()
img_sign = torch.ones_like(img_abs)
img_sign[img_phase.abs() > torch.pi / 2] = -1
img_abs_sign = img_abs * img_sign

# Plot coil-combined images for PEX
fig, axs = plt.subplots(2, img.shape[0], figsize=(10, 3))
for i in range(img.shape[0]):
    im1 = axs[0, i].imshow(img_abs[i, ...], cmap='gray', vmin=0, vmax=1)
    im2 = axs[1, i].imshow(img_phase[i, ...], cmap='turbo', vmin=-torch.pi, vmax=torch.pi)

fig.colorbar(im1, ax=axs[0, -1])
fig.colorbar(im2, ax=axs[1, -1])
plt.show()

# Plot coil-combined images for PEX
fig, axs = plt.subplots(1, img.shape[0], figsize=(10, 3))
for i in range(img.shape[0]):
    im1 = axs[i].imshow(img_abs_sign[i, ...], cmap='seismic', vmin=-1, vmax=1)
fig.colorbar(im1, ax=axs[-1])
plt.show()

# plot signal of one pixel
plt.figure()
plt.plot(voltages, img_abs_sign[:, 32, 32], marker='o', linestyle='None')
plt.show()
# %% define signal model
model_op = mrpro.operators.models.PexSimple(voltages, prep_delay, t1=1, pulse_duration=pulse_duration, n_tx=8)

# test signal model
# zero crossing should be close to 90/a
a = torch.tensor([2, 4, 10])
(signal,) = model_op.forward(a)

plt.figure()
plt.plot(voltages, signal[..., 2], marker='o', linestyle='None')
plt.show()

# %%
dictionary = mrpro.operators.DictionaryMatchOp(model_op).append(torch.linspace(1, 100, 10000))
(a_start,) = dictionary(img_abs_sign)

# mask out data from approx 135°
voltage_mask = sqrt_kwatt_to_volt(
    (3 / 4 * torch.pi) / (GYROMAGNETIC_RATIO_PROTON * 2 * torch.pi * a_start * 1e-6 * pulse_duration)
) * 8 ** (-0.5)

weight = torch.ones(13, 64, 64)
for i in range(voltage_mask.shape[0]):
    for j in range(voltage_mask.shape[1]):
        voltages_tensor = torch.tensor(voltages) if isinstance(voltages, list) else voltages
        weight[:, i, j] = voltages_tensor < voltage_mask[i, j]

mse_loss = mrpro.operators.functionals.MSE(img_abs_sign, weight=weight)

constraints_op = mrpro.operators.ConstraintsOp(
    bounds=(
        (0, 100),  # a is constrained between 1 and 100 µT/sqrt(kW)
    )
)
functional = mse_loss @ model_op @ constraints_op
initial_parameters = constraints_op.inverse(a_start)


(result,) = constraints_op(*mrpro.algorithms.optimizers.lbfgs(functional, initial_parameters=initial_parameters))
result = result.detach().cpu().squeeze()
# %%
plt.figure()
plt.imshow(result, vmin=0, vmax=60)
plt.colorbar()

plt_idx = [20, 20]

plt.figure()
plt.plot(voltages, img_abs_sign[:, plt_idx[0], plt_idx[1]], marker='o', linestyle='None', label='data')
plt.plot(voltages, model_op.forward(result[plt_idx[0], plt_idx[1]])[0], marker='o', linestyle='None', label='fit')
plt.plot(voltages, model_op.forward(a_start[plt_idx[0], plt_idx[1]])[0], marker='o', linestyle='None', label='initial')
plt.legend()
plt.show()

# %%
