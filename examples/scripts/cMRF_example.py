# %% [markdown]
# # Cardia MRF reconstructions
#
# This notebook provides the image reconstruction and parameter estimation methods required to reproduce the multi-scanner comparison carried out in the paper.


# %% [markdown]
#
# ## Overview
# In this notebook the cardiac MR Fingerprinting (cMRF) data acquired at four different scanners and the corresponding spin-echo reference sequences are reconstructed and
# $T_1$ and $T_2$ maps are estimated. Average $T_1$ and $T_2$ are calculated in circular ROIs for different tissue types represented in the phantom.

# %% [markdown]
#
# In this example, we are going to:
# - Download data
# - Define image reconstruction and parameter estimation methods for cMRF and reference sequences
# - Run through all datasets and calculate $T_1$ and $T_2$ maps
# - Visualize results
# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show download details"}
# Get the raw data from zenodo

data_folder = '/echo/redsha01/Sequences_Evaluation/mrpro/examples/scripts/cMRF_example_folder/'
# %% [markdown]
#
# We will use the following libraries:
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import mrpro
import torch

# %% [markdown]
#
# Define image reconstruction and parameter estimation methods for cMRF and reference sequences
# For all scans we carry out dictionary matching to estimate the quantitative parameters from a series of qualitative images. So let's start by defining a function for this.
from mrpro.operators.Operator import Operator


class SignalAverage(Operator[torch.Tensor, tuple[torch.Tensor,]]):
    def __init__(self, idx: torch.Tensor, dim: int = 0) -> None:
        super().__init__()
        if idx.ndim != 2:
            raise ValueError('idx must have exactly 2 dimensions and shape (n_sets, n_points_per_average)')
        self.idx = idx
        self.dim = dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        if self.dim >= x.ndim or self.dim < -x.ndim:
            raise ValueError(f'Dimension {self.dim} out of range for input with {x.ndim} dimensions')
        dim = self.dim % x.ndim

        index = (*(slice(None),) * (dim), self.idx)
        x_indexed = x[*index]
        return (x_indexed.mean(dim + 1),)


# Finally we define the function to reconstruct the cMRF data and estimate the $T_1$ and $T_2$ maps.


# Function to reconstruct the cMRF scans
def reco_cMRF_scans(pname, scan_name, t1, t2):
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
    acq_t_ms = kdata.header.acq_info.acquisition_time_stamp[0, 0, 0, :, 0]
    te = 1.52 / 1000

    epg_mrf_fisp = mrpro.operators.models.cMRF.CardiacFingerprinting(acq_t_ms, te)
    model = SignalAverage(dyn_idx, dim=0) @ epg_mrf_fisp
    dictionary = mrpro.operators.DictionaryMatchOp(model).append(m0, t1, t2)
    # Select the closest values in the dictionary for each voxel based on cosine similarity
    m0_match, t1_match, t2_match = dictionary(img)
    return t1_match, t2_match


# %% [markdown]
# ## Run through all datasets and calculate $T_1$ and $T_2$ maps
#
# Now we can go through the acquisitions at the different scanners, reconstruct the cMRF and reference scans, estimate $T_1$ and $T_2$ maps

# Define the T1 and T2 values to be included in the dictionaries
t1 = (
    torch.cat((torch.arange(50, 2000 + 10, 10), torch.arange(2020, 3000 + 20, 20), torch.arange(3050, 5000 + 50, 50)))
    / 1000
)
t2 = torch.cat((torch.arange(6, 100 + 2, 2), torch.arange(105, 200 + 5, 5), torch.arange(220, 500 + 20, 20))) / 1000

cmrf_t1_maps = []
cmrf_t2_maps = []

# Current path of data
pname = data_folder / Path('scanner1/')

# cMRF T1 and T2 maps
t1_map_cmrf, t2_map_cmrf = reco_cMRF_scans(pname, 'cMRF.h5', t1, t2)
cmrf_t1_maps.append(t1_map_cmrf)
cmrf_t2_maps.append(t2_map_cmrf)

# %% [markdown]
# ## Visualize and evaluate results

# Now we visualize and compare all the results.

# Create recommended colormaps

# Plot T1 and T2 maps

fig, ax = plt.subplots(2, 1)
for cax in ax.flatten():
    cax.set_axis_off()
im = ax[0].imshow(cmrf_t1_maps[0][0], vmin=0, vmax=2)
ax[0].set_title('cMRF T1 (ms)')
plt.colorbar(im)
im = ax[1].imshow(cmrf_t2_maps[0][0], vmin=0, vmax=0.2)
ax[1].set_title('cMRF T2 (ms)')
plt.colorbar(im)
plt.tight_layout()
plt.show()


# %%
