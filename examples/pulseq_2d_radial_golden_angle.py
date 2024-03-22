# %% [markdown]
# # Reconstruction of 2D cartesian data from siemens sequence

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

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
from mrpro.data.traj_calculators import KTrajectoryRadial2D
from mrpro.operators import FourierOp
from mrpro.operators import SensitivityOp

# %%
# define zenodo records URL and create temporary directory
zenodo_url = 'https://zenodo.org/records/10854057/files/'
fname = 'pulseq_radial_2D_402spokes_golden_angle_with_traj.h5'
data_folder = Path(tempfile.mkdtemp())
data_file = tempfile.NamedTemporaryFile(dir=data_folder, mode='wb', delete=False, suffix='.h5')

# %%
# Download raw data using requests
response = requests.get(zenodo_url + fname, timeout=10)
data_file.write(response.content)

# %%
# Read raw data trajectory from ISMRMRD file
kdata = KData.from_file(data_file.name, KTrajectoryIsmrmrd())

# Calculate dcf
dcf = DcfData.from_traj_voronoi(kdata.traj)

# Reconstruct average image for coil map estimation
fourier_op = FourierOp(
    recon_matrix=kdata.header.recon_matrix,
    encoding_matrix=kdata.header.encoding_matrix,
    traj=kdata.traj,
)
(img,) = fourier_op.adjoint(kdata.data * dcf.data[:, None, ...])

# %%
# Calculate coilmaps
idata = IData.from_tensor_and_kheader(img, kdata.header)
csm = CsmData.from_idata_walsh(idata)
csm_op = SensitivityOp(csm)

# %%
# Coil combination
(img_from_ismrmrd_traj,) = csm_op.adjoint(img)

# %% [markdown]
# ### Image reconstruction using KTrajectoryRadial2D

# %%
# Read raw data and calculate trajectory using KTrajectoryRadial2D
kdata = KData.from_file(data_file.name, KTrajectoryRadial2D())

# Calculate dcf
dcf = DcfData.from_traj_voronoi(kdata.traj)

# Reconstruct average image for coil map estimation
fourier_op = FourierOp(
    recon_matrix=kdata.header.recon_matrix,
    encoding_matrix=kdata.header.encoding_matrix,
    traj=kdata.traj,
)
(img,) = fourier_op.adjoint(kdata.data * dcf.data[:, None, ...])

# %%
# Calculate coilmaps
idata = IData.from_tensor_and_kheader(img, kdata.header)
csm = CsmData.from_idata_walsh(idata)
csm_op = SensitivityOp(csm)

# %%
# Coil combination
(img_from_rad2d_traj_calc,) = csm_op.adjoint(img)

# %% [markdown]
# ### Image reconstruction using KTrajectoryPulseq
# This is not implemented yet because the seq-file is not available in the zenodo record.

# %%
titles = ['KTrajectoryIsmrmrd', 'KTrajectoryRadial2D']
plt.subplots(1, 2)
for i, img in enumerate([img_from_ismrmrd_traj, img_from_rad2d_traj_calc]):
    plt.subplot(1, len(titles), i + 1)
    plt.imshow(torch.abs(img[0, 0, 0, :, :]))
    plt.title(titles[i])
    plt.axis('off')

# %%
print('here')
