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
# - Download data from zenodo
# - Define image reconstruction and parameter estimation methods for cMRF and reference sequences
# - Define evaluation methods
# - Run through all datasets and calculate $T_1$ and $T_2$ maps
# - Visualise and evaluate results
# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show download details"}
# Get the raw data from zenodo

data_folder = '/echo/redsha01/Sequences_Evaluation/mrpro/examples/scripts/cMRF_example_folder/'

# %% [markdown]
#
# We will use the following libraries:
# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
import mrpro

import zenodo_get
from matplotlib.colors import ListedColormap
from pathlib import Path
from einops import rearrange
from mrpro.utils import split_idx
from mrpro.data import KData, DcfData, IData, CsmData
from mrpro.data.traj_calculators import KTrajectoryIsmrmrd
from mrpro.algorithms.reconstruction import DirectReconstruction
from mrpro.operators.models.cMRF import CardiacFingerprinting

# %% [markdown]
#
# Define image reconstruction and parameter estimation methods for cMRF and reference sequences
# For all scans we carry out dictionary matching to estimate the quantitative parameters from a series of qualitative images. So let's start by defining a function for this.

from mrpro.operators.Operator import Operator
class SignalAverage(Operator[torch.Tensor, tuple[torch.Tensor,]]):
    def __init__(self, idx:torch.Tensor, dim:int=0)->None:
        super().__init__()
        if idx.ndim != 2:
            raise ValueError("idx must have exactly 2 dimensions and shape (n_sets, n_points_per_average)")
        self.idx = idx
        self.dim=dim

    def forward(self,x:torch.Tensor) -> tuple[torch.Tensor]:
        if self.dim>=x.ndim or self.dim<-x.ndim:
            raise ValueError(f"Dimension {self.dim} out of range for input with {x.ndim} dimensions")
        dim=self.dim%x.ndim

        index = (*(slice(None),)*(dim), self.idx)
        x_indexed=x[*index]
        return (x_indexed.mean(dim+1),)

#Function to calculate the dictionary matching
def dictionary_matching(img_data, model, dictionary_values):
    dictionary_values = dictionary_values.to(dtype=torch.float32)
    (signal_dictionary,) = model(torch.ones(1), dictionary_values)
    signal_dictionary = signal_dictionary.to(dtype=torch.complex64)
    vector_norm = torch.linalg.vector_norm(signal_dictionary, dim=0)
    signal_dictionary /= vector_norm
    signal_dictionary = signal_dictionary.to(img_data.dtype)

    # Calculate the dot-product
    # and select for each voxel the values that correspond to the maximum of the dot-product
    n_y, n_x = img_data.shape[-2:]

    dot_product = torch.mm(rearrange(img_data, 'other 1 z y x->(z y x) other'), signal_dictionary)
    idx_best_match = torch.argmax(torch.abs(dot_product), dim=1)
    return rearrange(dictionary_values[idx_best_match], '(y x)->1 1 y x', y=n_y, x=n_x)


# Finally we define the function to reconstruct the cMRF data and estimate the $T_1$ and $T_2$ maps.

#Function to reconstruct the cMRF scans
def reco_cMRF_scans(pname, scan_name, fa, t1, t2):
    n_lines_per_img = 20
    n_lines_overlap= 10

    #Image reconstruction of average image
    kdata = KData.from_file(pname / scan_name, KTrajectoryIsmrmrd())
    avg_recon = DirectReconstruction(kdata)

    #Split data into dynamics and reconstruct
    dyn_idx = split_idx(torch.arange(0,47), n_lines_per_img, n_lines_overlap)
    dyn_idx = torch.cat([dyn_idx + ind*47 for ind in range(15)], dim=0)

    kdata_dyn = kdata.split_k1_into_other(dyn_idx, other_label='repetition')

    dyn_recon = DirectReconstruction(kdata_dyn, csm=avg_recon.csm)
    dcf_data_dyn = rearrange(avg_recon.dcf.data, 'coil k2 k1 other k0->other coil k2 k1 k0')
    dcf_data_dyn = rearrange(dcf_data_dyn[dyn_idx.flatten(),...], '(other k1) 1 coil k2 k0->other coil k2 k1 k0', k1=dyn_idx.shape[-1])
    dyn_recon.dcf = DcfData(dcf_data_dyn)

    img = dyn_recon(kdata_dyn).rss()[:,0,:,:]

    #Dictionary settings
    t1, t2 = torch.broadcast_tensors(t1[None,:], t2[:,None])
    t1_all = t1.flatten().to(dtype=torch.float32)
    t2_all = t2.flatten().to(dtype=torch.float32)

    t1 = t1_all[t1_all >= t2_all]
    t2 = t2_all[t1_all >= t2_all]
    m0 = torch.ones_like(t1)

    #Dictionary calculation
    n_rf_pulses_per_block = 47 # 47 RF pulses in each block
    acq_t_ms = kdata.header.acq_info.acquisition_time_stamp[0,0,0,:,0]
    delay_between_blocks = [acq_t_ms[n_block*n_rf_pulses_per_block] - acq_t_ms[n_block*n_rf_pulses_per_block-1] for n_block in range(1,3*5)]
    delay_between_blocks.append(delay_between_blocks[-1]) # last delay is not needed but makes computations easier


    te = 1.52/1000
    epg_mrf_fisp = CardiacFingerprinting(acq_t_ms, te)
    (signal_dictionary,) = epg_mrf_fisp.forward(m0, t1, t2)



    signal_dictionary = rearrange(signal_dictionary[dyn_idx.flatten(),...], '(other k1) t->other t k1', k1=dyn_idx.shape[-1])
    signal_dictionary = torch.mean(signal_dictionary, dim=-1)
    signal_dictionary = signal_dictionary.abs()

    #Normalise dictionary entries
    vector_norm = torch.linalg.vector_norm(signal_dictionary, dim=0)
    signal_dictionary /= vector_norm

    #Dictionary matching
    n_y, n_x = img.shape[-2:]
    dot_product = torch.mm(rearrange(img.abs(), 'other y x->(y x) other'), signal_dictionary)
    idx_best_match = torch.argmax(torch.abs(dot_product), dim=1)
    t1_match = rearrange(t1[idx_best_match], '(y x)->y x', y=n_y, x=n_x)
    t2_match = rearrange(t2[idx_best_match], '(y x)->y x', y=n_y, x=n_x)
    model = epg_mrf_fisp
    model = SignalAverage(dyn_idx, dim=0)@epg_mrf_fisp
    dictionary = mrpro.operators.DictionaryMatchOp(model).append(m0, t1,t2)
    # Select the closest values in the dictionary for each voxel based on cosine similarity
    m0_start, t1_match, t2_match = dictionary(img.abs())
    return t1_match, t2_match

# %% [markdown]
# ## Run through all datasets and calculate $T_1$ and $T_2$ maps
#
# Now we can go through the acquisitions at the different scanners, reconstruct the cMRF and reference scans, estimate $T_1$ and $T_2$ maps

#Define the T1 and T2 values to be included in the dictionaries
t1 = torch.cat((torch.arange(50, 2000+10, 10), torch.arange(2020, 3000+20, 20), torch.arange(3050,5000+50,50)))/1000
t2 = torch.cat((torch.arange(6, 100+2, 2), torch.arange(105, 200+5, 5), torch.arange(220,500+20,20)))/1000

#Read in flip angle pattern
fname_angle = data_folder / Path('cMRF_fa_705rep.txt')

with open(fname_angle, "r") as file:
    fa = torch.as_tensor([float(line) for line in file.readlines()])/180 * torch.pi

cmrf_t1_maps = []
cmrf_t2_maps = []



#Current path of data
pname = data_folder / Path(f'scanner1/')

#cMRF T1 and T2 maps
t1_map_cmrf, t2_map_cmrf = reco_cMRF_scans(pname, 'cMRF.h5', fa, t1, t2)
cmrf_t1_maps.append(t1_map_cmrf)
cmrf_t2_maps.append(t2_map_cmrf)

# %% [markdown]
# ## Visualise and evaluate results
#
# Now we visualise and compare all the results.

# Create recommended colormaps
cmap_t1 = ListedColormap(np.loadtxt(data_folder / Path('lipari.csv')))
cmap_t2 = ListedColormap(np.loadtxt(data_folder / Path('navia.csv')))

# Plot T1 and T2 maps

fig, ax = plt.subplots(2,1)
for cax in ax.flatten():
    cax.set_axis_off()

im = ax[0].imshow(cmrf_t1_maps[0], vmin=0, vmax=2, cmap=cmap_t1)
ax[0].set_title('cMRF T1 (ms)')
plt.colorbar(im)
im = ax[1].imshow(cmrf_t2_maps[0], vmin=0, vmax=0.2, cmap=cmap_t2)
ax[1].set_title('cMRF T2 (ms)')
plt.colorbar(im)
plt.tight_layout()
plt.show()



# %%
