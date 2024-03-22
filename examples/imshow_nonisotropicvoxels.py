# %% [markdown]
# # Plotting example of data with non-isotropic voxel size

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
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Plotting of image data
# Plotting with matplotlib imshow needs special attention when plotting non-isotropic voxel data:
# - The extent is set such that coordinates are located in the center of the voxel
# - The coordinates can be either pixel coordinates or spatial coordinates

# %%
fig = plt.figure(constrained_layout=True)
fig.suptitle('Plotting of image data')
subfigs = fig.subfigures(nrows=2, ncols=1)

encoding_matrices = ((4, 4), (5, 5))
subfig_titles = ('Even', 'Odd')
res_x = 1.5
res_y = 3


for idx, encoding_matrix in enumerate(encoding_matrices):
    subfig = subfigs[idx]
    subfig.suptitle(f'%s' % subfig_titles[idx], ha='right', va='center', x=-0.05, y=0.5, rotation=90)

    axs = subfig.subplots(nrows=1, ncols=2)

    fov = tuple(l * r for l, r in zip(encoding_matrix, (res_y, res_x)))

    # the position of np.ceil and np.floor brackets are rather important.
    axs[0].imshow(np.random.rand(encoding_matrix[0], encoding_matrix[1]), interpolation=None,  extent=(-np.floor(encoding_matrix[1]/2)-0.5, np.ceil(encoding_matrix[1]/2)-0.5, -np.floor(encoding_matrix[0]/2)-0.5, np.ceil(encoding_matrix[0]/2)-0.5))
    axs[0].set_aspect(res_x / res_y)
    axs[0].set_title('Pixel coordinates %dx%d' % (encoding_matrix[0], encoding_matrix[1]))
    axs[0].set_ylabel('y coord [px]')
    axs[0].set_xlabel('x coord [px]')

    axs[1].imshow(np.random.rand(encoding_matrix[0], encoding_matrix[1]), interpolation=None, extent=(-fov[0]/2-res_y/2, fov[0]/2+res_y/2, -fov[1]/2-res_x/2, fov[1]/2+res_x/2))

    # axs[1].imshow(np.random.rand(encoding_matrix[0], encoding_matrix[1]), interpolation=None, extent=((-np.floor(encoding_matrix[1]/2)-0.5)*res_y, (np.ceil(encoding_matrix[1]/2)-0.5)*res_y, (-np.floor(encoding_matrix[0]/2)-0.5)*res_x, (np.ceil(encoding_matrix[0]/2)-0.5)*res_x))

    axs[1].set_xticks((-fov[0]/2, 0, fov[0]/2))
    axs[1].set_yticks((-fov[1]/2, 0, fov[1]/2))
    axs[1].set_title('FOV coordinates %.1fx%.1f' % (fov[0], fov[1]))
    axs[0].set_ylabel('FOVy coord [mm]')
    axs[0].set_xlabel('FOVx coord [mm]')

# %%
