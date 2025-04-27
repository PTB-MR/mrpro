# %% [markdown]
# # Cardiac MRF reconstructions
#
# This notebook provides the image reconstruction and parameter estimation methods required to reconstruct cardiac MR
# Fingerprinting (cMRF) data.


# %% [markdown]
#
# # Overview
# %%
# In this notebook the cardiac MR Fingerprinting (cMRF) data acquired at one scanner and the corresponding spin-echo
# reference sequence are reconstructed and $T_1$ and $T_2$ maps are estimated. This example is based on the same
# data as for one of the scanners in the scanner comparison example in [SCHUE2024].  Average $T_1$ and $T_2$ are
# calculated in circular ROIs for different tissue types represented in the phantom.


# For the reconstruction of the acquired data the cMRF signal model is used. This model simulates a cardiac MR
# fingerprinting sequence as described in [HAMI2017]_ using the extended phase graph
# (`~mrpro.operators.models.EPG`) formalism.

# It is a four-fold repetition of

#             Block 0                   Block 1                   Block 2                     Block 3
#    R-peak                   R-peak                    R-peak                    R-peak                    R-peak
# ---|-------------------------|-------------------------|-------------------------|-------------------------|-----

#         [INV TI=30ms][ACQ]                     [ACQ]     [T2-prep TE=50ms][ACQ]    [T2-prep TE=100ms][ACQ]

# In order not to reconstruct all acquired images, the acquired data is split into windows of 20 acquisitions. The
# windows have an overap of 10 acquisitins among each other.  As a result the acquired data is averaged over these
# windows, so that less images have to be reconstructed.

# We carry out dictionary matching to estimate the quantitative parameters from a series of qualitative images. For
# this we employ `~mrpro.operators.DictionaryMatchOp`. It performs absolute normalized dot product matching between
# a dictionary of signals, i.e. find the entry :math:`d^*` in the dictionary maximizing
# :math:`\left|\frac{d}{\|d\|} \cdot \frac{y}{\|y\|}\right|` and returns the associated signal model parameters
# :math:`x` generating the matching signal :math:`d^*=d(x)`.

# At initialization, the cMRF signal model from before needs to be provided. In addition to the extended phase graph
# formalism the simulated signal is averaged over 45 acquisition windows consisting of 20 acquisitions with an overlap
# of ten acquisitions. Then parameters like $m_0$, $T_1$ and $T_2$ are provided as entries for the dictionary and the
# cMRF signal model from before is applied on them. Given the reconstructed images from above, the tuple of
# ($m_0$, $T_1$, $T_2$)-values in the dictionary that result in  a signal with the highest dot-product similarity
# will be returned.

# References
# ----------
# .. [SCHUE2024] Schuenke, P. et al. (2024) Open-Source Cardiac MR Fingerprinting
# .. [HAMI2017] Hamilton, J. I. et al. (2017) MR fingerprinting for rapid quantification of myocardial T1, T2, and
#         proton spin density. Magn. Reson. Med. 77 http://doi.wiley.com/10.1002/mrm.26668


# %% [markdown]
#
# In this example, we are going to:
# - Download data
# - Define image reconstruction and parameter estimation methods for cMRF and reference sequences
# - Run through all datasets and calculate $T_1$ and $T_2$ maps
# - Visualize and evaluate results
# - Assertion of cMRF results
# %%

# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show download details"}
# Download data from zenodo
import tempfile
from pathlib import Path

import zenodo_get

dataset = '14617082'  # FIXME: Change to real dataset!

tmp = tempfile.TemporaryDirectory()  # RAII, automatically cleaned up
data_folder = Path(tmp.name)
zenodo_get.zenodo_get([dataset, '-r', 5, '-o', data_folder])  # r: retries

# %%
import mrpro

kdata = mrpro.data.KData.from_file(data_folder / 'cMRF.h5', mrpro.data.traj_calculators.KTrajectoryIsmrmrd())
avg_recon = mrpro.algorithms.reconstruction.DirectReconstruction(kdata)

# %% [markdown]
# We want to perform a sliding window reconstruction respecting the block structure of the acquisition.
# We construct a split index that splits the data into windows of 10 acquisitions with an overlap of 5 acquisitions.

import torch

n_acq_per_image = 10
n_overlap = 5
n_acq_per_block = 47
n_blocks = 15

idx_in_block = torch.arange(n_acq_per_block).unfold(0, n_acq_per_image, n_acq_per_image - n_overlap)
split_indices = (n_acq_per_block * torch.arange(n_blocks)[:, None, None] + idx_in_block).flatten(end_dim=1)
kdata_split = kdata[..., split_indices, :]

# %% [markdown]
# We can now perform the reconstruction for each window.
avg_recon = mrpro.algorithms.reconstruction.DirectReconstruction(kdata)
recon = mrpro.algorithms.reconstruction.DirectReconstruction(kdata_split, csm=avg_recon.csm)
img = recon(kdata_split).data[:, 0, :, :]


# Dictionary settings
t1_keys = torch.arange(0.05, 5, 0.01)[:, None]
t2_keys = torch.arange(0.006, 0.5, 0.002)[None, :]
m0_keys = torch.tensor(1.0)

model = mrpro.operators.AveragingOp(dim=0, idx=split_indices) @ mrpro.operators.models.CardiacFingerprinting(
    kdata.header.acq_info.acquisition_time_stamp.squeeze(), echo_time=0.00155
)
dictionary = mrpro.operators.DictionaryMatchOp(model, index_of_scaling_parameter=0).append(m0_keys, t1_keys, t2_keys)
m0_match, t1_match, t2_match = dictionary(img)

# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show statistics helper functions"}
import numpy as np


def image_statistics(idat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate mean value and standard deviation in the ROIs."""
    mask = np.squeeze(np.load(data_folder / 'mask.npy'))
    n_tubes = 9
    mean = torch.cat([torch.mean(idat[mask == idx]) for idx in range(1, n_tubes + 1)])
    std_deviation = torch.cat([torch.std(idat[mask == idx]) for idx in range(1, n_tubes + 1)])
    return mean, std_deviation


def r_squared(true: torch.Tensor, predicted: torch.Tensor) -> float:
    """Calculate the coefficient of determination (R-squared)."""
    total = ((true - true.mean()) ** 2).sum()
    residual = (true - predicted).sum() ** 2
    r2 = 1 - residual / total
    return r2.item()


# %% [markdown]
# ## Visualize and evaluate results
# Now we visualize and compare all the results.
# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show plotting code"}
import matplotlib.pyplot as plt
from cmap import Colormap


def show_image(t1: torch.Tensor, t2: torch.Tensor) -> None:
    """Show the cMRF $T_1$ and $T_2$ maps."""
    cmap_t1 = Colormap('lipari')
    cmap_t2 = Colormap('navia')
    fig, ax = plt.subplots(2, 1)

    im = ax[0].imshow(t1.numpy(force=True), vmin=0, vmax=2, cmap=cmap_t1.to_mpl())
    ax[0].set_title('cMRF T1 (s)')
    ax[0].set_axis_off()
    plt.colorbar(im)

    im = ax[1].imshow(t2.numpy(force=True), vmin=0, vmax=0.2, cmap=cmap_t2.to_mpl())
    ax[1].set_title('cMRF T2 (s)')
    ax[1].set_axis_off()
    plt.colorbar(im)

    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## Visualize and evaluate results
# We can plot the cMRF $T_1$ and $T_2$ maps:
# %%
show_image(t1_match, t2_match)
# %% [markdown]
# We can also plot the statistics of the cMRF $T_1$ and $T_2$ maps and compare them to pre-calculated reference values,
# obtained from a separate reference scan.
# %% tags=["hide-input"] mystnb={"code_prompt_show": "Show plotting code"}
# Pre-calculated $T_1$ and $T_2$ spin-echo reference values for the nine tubes
t1_mean_ref = torch.tensor([1.022, 0.336, 0.287, 1.379, 0.430, 0.430, 1.755, 0.557, 0.580])
t1_std_ref = torch.tensor([0.012, 0.005, 0.004, 0.024, 0.007, 0.003, 0.028, 0.006, 0.003])
t2_mean_ref = torch.tensor([0.030, 0.033, 0.113, 0.037, 0.032, 0.123, 0.185, 0.032, 0.120])
t2_std_ref = torch.tensor([0.009, 0.001, 0.003, 0.001, 0.001, 0.002, 0.003, 0.001, 0.004])


t1_mean_cmrf, t1_std_cmrf = image_statistics(t1_match)
t2_mean_cmrf, t2_std_cmrf = image_statistics(t2_match)


fig, ax = plt.subplots(1, 2, figsize=(12, 7))
ax[0].errorbar(t1_mean_ref, t1_mean_cmrf, t1_std_cmrf, t1_std_ref, fmt='o', color='teal')
ax[0].plot([0, 2000], [0, 2000], color='darkorange')
ax[0].text(
    200,
    1800,
    rf'$R^2$ = {r_squared(t1_mean_ref, t1_mean_cmrf):.4f}',
    fontsize=12,
    verticalalignment='top',
    horizontalalignment='left',
    bbox={'facecolor': 'white', 'alpha': 0.5},
)
ax[0].set_xlabel('T1 - Reference (s)', fontsize=14)
ax[0].set_ylabel('T1 - cMRF (s)', fontsize=14)
ax[0].grid()
ax[0].set_aspect('equal', adjustable='box')

ax[1].errorbar(t2_mean_ref, t2_mean_cmrf, t2_std_cmrf, t2_std_ref, fmt='o', color='teal')
ax[1].plot([0, 200], [0, 200], color='darkorange')
ax[1].text(
    20,
    180,
    rf'$R^2$ = {r_squared(t2_mean_ref, t2_mean_cmrf):.4f}',
    fontsize=12,
    verticalalignment='top',
    horizontalalignment='left',
    bbox={'facecolor': 'white', 'alpha': 0.5},
)
ax[1].set_xlabel('T2 - Reference (s)', fontsize=14)
ax[1].set_ylabel('T2 - cMRF (s)', fontsize=14)
ax[1].grid()
ax[1].set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()

# %% tags=["hide-cell"]
# Assertion verifies if cMRF results match the pre-calculated reference values
torch.testing.assert_close(t1_mean_ref, t1_mean_cmrf, atol=0, rtol=0.15)
torch.testing.assert_close(t2_mean_ref, t2_mean_cmrf, atol=0, rtol=0.15)

# %%
