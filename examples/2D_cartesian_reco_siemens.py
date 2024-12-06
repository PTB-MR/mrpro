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
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from mrpro.data import CsmData
from mrpro.data import IData
from mrpro.data import KData
from mrpro.data.traj_calculators import KTrajectoryCartesian
from mrpro.operators import CartesianSamplingOp
from mrpro.operators import FourierOp
from mrpro.operators import SensitivityOp

# %%
data_folder = Path(R'/echo/_allgemein/projects/8_13/MRPro/2024_03_19/siemens_data/2D_cartesian')
fnames = list(data_folder.glob('*.h5'))
# %% [markdown]
# ## Image reconstruction
# Image reconstruction involves the following steps:
# - Reading in the raw data and the trajectory from the ismrmrd raw data file
# - Calculating the density compensation function (dcf)
# - Reconstructing one image averaging over the entire relaxation period

# %%
# Read raw data and trajectory
image_list = []
title_list = []
header_list = []

for fname in fnames:
    kdata = KData.from_file(data_folder / fname, KTrajectoryCartesian())

    # Reconstruct average image for coil map estimation
    Sop = CartesianSamplingOp(kdata.header.encoding_matrix, kdata.traj)
    fourier_op = FourierOp(
        recon_matrix=kdata.header.recon_matrix, encoding_matrix=kdata.header.encoding_matrix, traj=kdata.traj
    )
    (img,) = fourier_op.adjoint(Sop.adjoint(kdata.data)[0])
    print(img.shape)

    # Calculate coilmaps
    idata = IData.from_tensor_and_kheader(img, kdata.header)
    csm = CsmData.from_idata_walsh(idata)
    csm_op = SensitivityOp(csm)

    # Coil combination
    (img,) = csm_op.adjoint(img)

    title_list.append(fname.name.replace('2D_cart_', ''))
    image_list.append(torch.squeeze(img))
    header_list.append(idata.header)

# %% [markdown]
# ## Data in pixel coordinates
# %%
n_images = len(image_list)
n_cols = 2
n_rows = int(torch.ceil(torch.tensor((n_images) / n_cols)))
plt.subplots(n_rows, n_cols)
for i, img in enumerate(image_list):
    plt.subplot(n_rows, n_cols, i + 1)
    plt.title(title_list[i])

    if img.dim() == 3:
        img = img[0]

    res_x = header_list[i].fov.x / img.shape[0]
    res_y = header_list[i].fov.y / img.shape[1]

    n_rows_img = torch.tensor(img.shape[0])
    n_cols_img = torch.tensor(img.shape[1])

    plt.imshow(
        torch.abs(img),
        interpolation=None,
        extent=(
            -float(torch.floor(n_cols_img / 2) - 0.5),
            float(torch.ceil(n_cols_img / 2) - 0.5),
            -float(torch.floor(n_rows_img / 2) - 0.5),
            float(torch.ceil(n_rows_img / 2) - 0.5),
        ),
    )
    plt.gca().set_aspect(res_x / res_y)

# %% [markdown]
# ## Data in Field of view coordinates
# %%
n_images = len(image_list)
n_cols = 2
n_rows = int(torch.ceil(torch.tensor((n_images) / n_cols)))
plt.subplots(n_rows, n_cols)
for i, img in enumerate(image_list):
    plt.subplot(n_rows, n_cols, i + 1)
    plt.title(title_list[i])

    if img.dim() == 3:
        img = img[0]

    res_x = header_list[i].fov.x / img.shape[0]
    res_y = header_list[i].fov.y / img.shape[1]

    n_rows_img = torch.tensor(img.shape[0])
    n_cols_img = torch.tensor(img.shape[1])

    plt.imshow(
        torch.abs(img),
        interpolation=None,
        extent=(
            -header_list[i].fov.x / 2 - res_y / 2,
            header_list[i].fov.x / 2 + res_y / 2,
            -header_list[i].fov.y / 2 - res_x / 2,
            header_list[i].fov.y / 2 + res_x / 2,
        ),
    )
    plt.gca().set_xticks((-header_list[i].fov.x / 2, 0, header_list[i].fov.x / 2))
    plt.gca().set_yticks((-header_list[i].fov.y / 2, 0, header_list[i].fov.y / 2))

# %%
