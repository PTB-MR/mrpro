# %% [markdown]
# # Basic example for reconstruction of radial interleaved sampled data
# This notebook is meant as a first introduction into mrpro-based
# reconstruction of radial interleaved data.
# %%
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import zenodo_get

from mrpro.data import CsmData
from mrpro.data import IData
from mrpro.data import SpatialDimension
from mrpro.data._DcfData import DcfData
from mrpro.data._KData import KData
from mrpro.data.traj_calculators._KTrajectoryRadial2D import KTrajectoryRadial2D
from mrpro.operators import SensitivityOp
from mrpro.operators._FourierOp import FourierOp

# %% [markdown]

# # Download example data
# The example data is taken from (https://zenodo.org/records/10669837).
# It entails data acquired using radial-interleaved sequences written in pulseq.
# %%
# Create a temporary directory to save the downloaded files
data_folder = Path(tempfile.mkdtemp())
print(f'Data will be saved to: {data_folder}')

DATASET = '10664974'
data_folder = Path(tempfile.mkdtemp())
zenodo_get.zenodo_get([DATASET, '-r', 5, '-o', data_folder])  # r: retries

# %%
filepath = data_folder
h5_filename = 'meas_MID00044_FID00370_20240130_CEST_interleafed_radial_256px_fov256_8mm_50spokes_100m.h5'
# %% [markdown]
# # Read in data
# Data is read in using the downloaded .h5 file and a corresponding
# trajectory calculator (2D Golden radial in this case). Given the trajectory
# calculator, mrpro will automatically read, sort and validate the data.
# %%
data = KData.from_file(
    ktrajectory=KTrajectoryRadial2D(),
    filename=f'{filepath}/{h5_filename}',
)
# %% [markdown]
# # Reconstruction
# Given the loaded data, the Fourier Operator is initialized next,
# with which the appropriate Fourier Transform is carried out.
# Furthermore, density compensation and coil sensitivity maps are created.

# %%
# perform FT and CSM
ft_op = FourierOp(
    recon_shape=SpatialDimension(1, 256, 256),
    encoding_shape=SpatialDimension(1, 256, 256),
    traj=data.traj,
    oversampling=SpatialDimension(1, 1, 1),
)

# %%
# Density compensation
dcf = DcfData.from_traj_voronoi(traj=data.traj)

# %%
# Calculate Coil sensitivity maps
(xcoils,) = ft_op.H(data.data * dcf.data)
idata = IData.from_tensor_and_kheader(xcoils, data.header)

smoothing_width = SpatialDimension(z=1, y=5, x=5)
csm = CsmData.from_idata_walsh(idata, smoothing_width)
sensitivity_op = SensitivityOp(csm)
(image,) = sensitivity_op.H(xcoils)

# %% [markdown]
# # See the results
# %%
plt.matshow(image.abs()[0, 0, 0, :, :])  # plotting single slice
# %%
