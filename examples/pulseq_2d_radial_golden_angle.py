# %% [markdown]
# # Reconstruction of 2D golden angle radial data from pulseq sequence

# Copyright 2024 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %% [markdown]
# If you want to run this notebook in binder you need to still install the MRpro package.
# This only needs to be done once in a binder session. Open a terminal (File -> New -> Terminal) and run:
# ```
# pip install -e ".[notebook]"
# ```
# This will install the MRpro package. Any other required python packages should already be present in this
# docker image.

# %%
# Imports
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import requests
import torch
from mrpro.data import CsmData
from mrpro.data import DcfData
from mrpro.data import IData
from mrpro.data import KData
from mrpro.data.traj_calculators import KTrajectoryIsmrmrd
from mrpro.data.traj_calculators import KTrajectoryPulseq
from mrpro.data.traj_calculators import KTrajectoryRadial2D
from mrpro.operators import FourierOp
from mrpro.operators import SensitivityOp

# %%
# define zenodo records URL and create a temporary directory and h5-file
zenodo_url = 'https://zenodo.org/records/10854057/files/'
fname = 'pulseq_radial_2D_402spokes_golden_angle_with_traj.h5'
data_folder = Path(tempfile.mkdtemp())
data_file = tempfile.NamedTemporaryFile(dir=data_folder, mode='wb', delete=False, suffix='.h5')

# %%
# Download raw data using requests
response = requests.get(zenodo_url + fname, timeout=30)
data_file.write(response.content)

# %% [markdown]
# ### Image reconstruction using KTrajectoryIsmrmrd
# This will use the trajectory that is stored in the ISMRMRD file.

# %%
# Read the raw data and the trajectory from ISMRMRD file
kdata = KData.from_file(data_file.name, KTrajectoryIsmrmrd())

# Calculate dcf using the trajectory
dcf = DcfData.from_traj_voronoi(kdata.traj)

# Reconstruct average image for coil map estimation
fourier_op = FourierOp(
    recon_matrix=kdata.header.recon_matrix,
    encoding_matrix=kdata.header.encoding_matrix,
    traj=kdata.traj,
)
(img,) = fourier_op.adjoint(kdata.data * dcf.data[:, None, ...])

# Calculate and apply coil maps
idata = IData.from_tensor_and_kheader(img, kdata.header)
csm = CsmData.from_idata_walsh(idata)
csm_op = SensitivityOp(csm)
(img_from_ismrmrd_traj,) = csm_op.adjoint(img)

# %% [markdown]
# ### Image reconstruction using KTrajectoryRadial2D
# This will calculate the trajectory using the radial 2D trajectory calculator.
# Please note that there is currently a mismatch between the actual trajectory
# that was used to acquire the data and the calculated trajectory. This leads
# to a deviation in the reconstructed image.

# %%
# Read raw data and calculate trajectory using KTrajectoryRadial2D
kdata = KData.from_file(data_file.name, KTrajectoryRadial2D())

# Calculate dcf using the calculated trajectory
dcf = DcfData.from_traj_voronoi(kdata.traj)

# Reconstruct average image for coil map estimation
fourier_op = FourierOp(
    recon_matrix=kdata.header.recon_matrix,
    encoding_matrix=kdata.header.encoding_matrix,
    traj=kdata.traj,
)
(img,) = fourier_op.adjoint(kdata.data * dcf.data[:, None, ...])

# Calculate and apply coil maps
idata = IData.from_tensor_and_kheader(img, kdata.header)
csm = CsmData.from_idata_walsh(idata)
csm_op = SensitivityOp(csm)
(img_from_rad2d_traj_calc,) = csm_op.adjoint(img)

# %% [markdown]
# ### Image reconstruction using KTrajectoryPulseq
# This will calculate the trajectory from the pulseq sequence file
# using the PyPulseq trajectory calculator.

# %%
# download the sequence file from zenodo
zenodo_url = 'https://zenodo.org/records/10868061/files/'
seq_fname = 'pulseq_radial_2D_402spokes_golden_angle.seq'
seq_file = tempfile.NamedTemporaryFile(dir=data_folder, mode='wb', delete=False, suffix='.seq')
response = requests.get(zenodo_url + seq_fname, timeout=30)
seq_file.write(response.content)


# %%
# Read raw data and calculate trajectory using KTrajectoryRadial2D
kdata = KData.from_file(data_file.name, KTrajectoryPulseq(seq_path=seq_file.name))

# Calculate dcf using the calculated trajectory
dcf = DcfData.from_traj_voronoi(kdata.traj)

# Reconstruct average image for coil map estimation
fourier_op = FourierOp(
    recon_matrix=kdata.header.recon_matrix,
    encoding_matrix=kdata.header.encoding_matrix,
    traj=kdata.traj,
)
(img,) = fourier_op.adjoint(kdata.data * dcf.data[:, None, ...])

# Calculate and apply coil maps
idata = IData.from_tensor_and_kheader(img, kdata.header)
csm = CsmData.from_idata_walsh(idata)
csm_op = SensitivityOp(csm)
(img_from_pulseq_traj_calc,) = csm_op.adjoint(img)


# %%
titles = ['KTrajectoryIsmrmrd', 'KTrajectoryRadial2D', 'KTrajectoryPulseq']
plt.subplots(1, len(titles))
for i, img in enumerate([img_from_ismrmrd_traj, img_from_rad2d_traj_calc, img_from_pulseq_traj_calc]):
    plt.subplot(1, len(titles), i + 1)
    plt.imshow(torch.abs(img[0, 0, 0, :, :]))
    plt.title(titles[i])
    plt.axis('off')

# %%
# Clean-up by removing temporary directory
shutil.rmtree(data_folder)
