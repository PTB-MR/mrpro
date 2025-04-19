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
# Imports
from pathlib import Path

import matplotlib.pyplot as plt
import mrpro
import numpy as np
import torch
from cmap import Colormap
from mrpro.operators.Operator import Operator
from scipy import odr

# Get the raw data from zenodo

# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show download details"}
# Download data from zenodo

data_folder = '/echo/redsha01/Sequences_Evaluation/mrpro/examples/scripts/cMRF_example_folder/'

# %% [markdown]
#
# Define image reconstruction and parameter estimation methods for cMRF and reference sequences.


# %%
# Function to mimic the averaging reconstruction of the actual acquired data when creatig the simulated signal model
class SignalAverage(Operator[torch.Tensor, tuple[torch.Tensor,]]):
    """Signal Averaging Operator.

    This operator can be used to average multiple signal amplitudes over a window of several signal ampltitudes.
    """

    def __init__(self, idx: torch.Tensor, dim: int = 0) -> None:
        """Initialize SignalAverage.

        Parameters
        ----------
        idx
            acquisition indices over which the signals are averaged
        dim
            averaging takes place over the dimension dim + 1
        """
        super().__init__()
        if idx.ndim != 2:
            raise ValueError('idx must have exactly 2 dimensions and shape (n_sets, n_points_per_average)')
        self.idx = idx
        self.dim = dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Perform signal averaging.

        Given x values as input_signal, the averaged signal over the dimension dim + 1 and the indices idx
        will be returned

        Parameters
        ----------
        x
            signal amplitudes of all acquisitions

        Returns
        -------
        x_mean
            tuple of averaged signal with shape (x_mean,)
        """
        if self.dim >= x.ndim or self.dim < -x.ndim:
            raise ValueError(f'Dimension {self.dim} out of range for input with {x.ndim} dimensions')
        dim = self.dim % x.ndim
        index = (*(slice(None),) * (dim), self.idx)
        x_indexed = x[*index]
        x_mean = x_indexed.mean(dim + 1)
        return (x_mean,)


# Function to reconstruct the cMRF scans and estimate the $T_1$ and $T_2$ maps.
def reco_cmrf_scans(
    pname: Path, scan_name: str, t1: torch.Tensor, t2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct cMRF scans."""
    n_lines_per_img = 20
    n_lines_overlap = 10
    # Image reconstruction of average image
    kdata = mrpro.data.KData.from_file(pname / scan_name, mrpro.data.traj_calculators.KTrajectoryIsmrmrd())
    avg_recon = mrpro.algorithms.reconstruction.DirectReconstruction(kdata)
    # Split data into dynamics and reconstruct
    dyn_idx = mrpro.utils.split_idx(torch.arange(0, 47), n_lines_per_img, n_lines_overlap)
    dyn_idx = torch.cat([dyn_idx + ind * 47 for ind in range(15)], dim=0)
    kdata_dyn = kdata.split_k1_into_other(dyn_idx, other_label='repetition')
    dyn_recon = mrpro.algorithms.reconstruction.DirectReconstruction(kdata_dyn, csm=avg_recon.csm)
    img = dyn_recon(kdata_dyn).data[:, 0, :, :]

    # Dictionary settings
    t1, t2 = torch.broadcast_tensors(t1[None, :], t2[:, None])
    t1_all = t1.flatten().to(dtype=torch.float32)
    t2_all = t2.flatten().to(dtype=torch.float32)
    t1 = t1_all[t1_all >= t2_all]
    t2 = t2_all[t1_all >= t2_all]
    m0 = torch.ones_like(t1)

    # Acquisition times and echo time
    acq_t_ms = kdata.header.acq_info.acquisition_time_stamp[0, 0, 0, :, 0]
    te = 1.555 / 1000

    # This model simulates a cardiac MR fingerprinting sequence using the extended phase graph formalism.
    epg_mrf_fisp = mrpro.operators.models.cMRF.CardiacFingerprinting(acq_t_ms, te)
    # The simulated signal is then averaged over 45 acquisition windows consisting of 20 acquisitions with an overlap of
    # ten acquisitions
    model = SignalAverage(dyn_idx, dim=0) @ epg_mrf_fisp
    # Appending m0-, t1- and t2-values to create a dictionary of signals according to the signal model calculated before
    dictionary = mrpro.operators.DictionaryMatchOp(model).append(m0, t1, t2)
    # Select the closest values in the dictionary for each voxel based on dot-product similarity
    m0_match, t1_match, t2_match = dictionary(img)
    return t1_match, t2_match


# Function to calculate the mean and standard deviation of the $T_1$- and $T_2$-values in the ROI in the image data
def image_statistics(idat: torch.Tensor, mask_name: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate mean value and standard deviation in the ROI."""
    if mask_name is not None:
        mask = np.squeeze(np.load(mask_name))

    number_of_tubes = 9
    mean = []
    std_deviation = []

    for idx_value in range(number_of_tubes):
        mean.append(torch.mean(idat[mask == idx_value + 1]))
        std_deviation.append(torch.std(idat[mask == idx_value + 1]))

    mean = torch.Tensor(mean) * 1000  # convert time values from second to milliseconds
    std_deviation = torch.Tensor(std_deviation) * 1000  # convert time values from second to milliseconds
    return mean, std_deviation


# Function to calculate the coefficient of determination $R^2$ for the pre-calculated reference time values and the cMRF
# time values
def r_squared(mean_ref: torch.Tensor, mean_cmrf: torch.Tensor) -> float:
    """Calculate of coefficient of determination."""

    # Function for the linear fit
    def linear(parameter: torch.Tensor, x: torch.Tensor):
        return parameter[0] * x + parameter[1]

    data = odr.RealData(mean_ref, mean_cmrf)
    model = odr.Model(linear)
    odr_instance = odr.ODR(data, model, beta0=[1, 0])
    output = odr_instance.run()

    # Calculation of R^2
    residual = mean_cmrf - linear(output.beta, mean_ref)
    res_total = torch.sum((mean_cmrf - torch.mean(t1_mean_cmrf)) ** 2)
    res_sq = torch.sum(residual**2)
    r2 = 1 - res_sq / res_total
    return r2


# %% [markdown]
# ## Run through all datasets and calculate $T_1$ and $T_2$ maps
# Now we can go through the acquisition at the scanner, reconstruct the cMRF and reference scans, estimate $T_1$ and
# $T_2$ maps.
# %%
# Define the $T_1$ and $T_2$ values to be included in the dictionaries
t1 = (
    torch.cat((torch.arange(50, 2000 + 10, 10), torch.arange(2020, 3000 + 20, 20), torch.arange(3050, 5000 + 50, 50)))
    / 1000
)
t2 = torch.cat((torch.arange(6, 100 + 2, 2), torch.arange(105, 200 + 5, 5), torch.arange(220, 500 + 20, 20))) / 1000

cmrf_t1_maps = []
cmrf_t2_maps = []

# Current path of data
pname = data_folder / Path('scanner1/')

# cMRF $T_1$ and $T_2$ maps
t1_map_cmrf, t2_map_cmrf = reco_cmrf_scans(pname, 'cMRF.h5', t1, t2)
cmrf_t1_maps.append(t1_map_cmrf)
cmrf_t2_maps.append(t2_map_cmrf)

# %% [markdown]
# ## Visualize and evaluate results
# Now we visualize and compare all the results.
# %%
# Create recommended colormaps
cmap_t1 = Colormap('lipari')
cmap_t2 = Colormap('navia')
# Plot $T_1$ and $T_2$ maps
fig, ax = plt.subplots(2, 1)
for cax in ax.flatten():
    cax.set_axis_off()
im = ax[0].imshow(cmrf_t1_maps[0][0], vmin=0, vmax=2, cmap=cmap_t1.to_mpl())
ax[0].set_title('cMRF T1 (ms)')
plt.colorbar(im)
im = ax[1].imshow(cmrf_t2_maps[0][0], vmin=0, vmax=0.2, cmap=cmap_t2.to_mpl())
ax[1].set_title('cMRF T2 (ms)')
plt.colorbar(im)
plt.tight_layout()
plt.show()


# %%
# Pre-calculated $T_1$ and $T_2$ spin-echo reference values (mean values and standard deviations of nine tubes)
t1_mean_ref = torch.tensor(
    [1022.1182, 336.3131, 287.9800, 1379.6000, 430.5941, 430.7805, 1755.2736, 557.5124, 580.6965]
)
t1_std_ref = torch.tensor([11.9088, 5.3222, 4.0151, 23.9967, 6.8682, 3.1816, 28.1794, 5.6234, 2.5456])
t2_mean_ref = torch.tensor([29.6749, 33.0909, 113.3586, 37.0100, 32.2871, 122.6585, 185.3682, 32.6070, 120.7962])
t2_std_ref = torch.tensor([0.9688, 1.055, 3.5473, 1.2806, 1.4131, 2.5434, 3.3448, 1.4000, 4.2746])

# $T_1$ and $T_2$ cMRF values (mean values and standard deviations of nine tubes)
t1_mean_cmrf, t1_std_cmrf = image_statistics(cmrf_t1_maps[0][0], pname / 'mask.npy')
t2_mean_cmrf, t2_std_cmrf = image_statistics(cmrf_t2_maps[0][0], pname / 'mask.npy')

# Plot $T_1$ and $T_2$ data with coefficient of determination $R^2$
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

for pidx in range(2):
    ax[pidx].set_xlabel(f'T{int(pidx + 1)} - Reference (ms)', fontsize=14)
    ax[pidx].set_ylabel(f'T{int(pidx + 1)} - cMRF (ms)', fontsize=14)
    ax[pidx].grid()
    ax[pidx].set_aspect('equal', adjustable='box')

plt.tight_layout()

# %% [markdown]
# ## Assertion of cMRF results
# Assertion verifies if cMRF results match the pre-calculated reference values
assert torch.max(abs((t1_mean_ref - t1_mean_cmrf) / t1_mean_ref)) < 0.15, (
    'Relative difference of cMRF T1 values and reference T1 values is hgher than 15%'
)
assert torch.max(abs((t2_mean_ref - t2_mean_cmrf) / t2_mean_ref)) < 0.15, (
    'Relative difference of cMRF T2 values and reference T2 values is hgher than 15%'
)

# %%
