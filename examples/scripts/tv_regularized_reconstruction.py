# %% [markdown]
# # TV-regularized reconstruction

# %%
# Imports
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import zenodo_get
from mrpro.algorithms.reconstruction import (
    DirectReconstruction,
    RegularizedIterativeSENSEReconstruction,
    TotalVariationDenoising,
    TotalVariationRegularizedReconstruction,
)
from mrpro.data import CsmData, IData, KData
from mrpro.data.traj_calculators import KTrajectoryIsmrmrd
from mrpro.operators import FiniteDifferenceOp
from mrpro.utils import split_idx


# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show plotting details"}
def show_dynamic_images(img: torch.Tensor, vmin: float = 0, vmax: float = 0.8, title: str | None = None) -> None:
    """Show a few time frames of the dynamic images and a plot along time.

    Parameters
    ----------
    img
        image tensor to be displayed
    vmin
        vmin for display
    vmax
        vmax for display
    title
        figure title
    """
    fig, ax = plt.subplots(1, 4, squeeze=False, figsize=(16, 4))
    if title:
        fig.suptitle(title, fontsize=16)
    for cax in ax.flatten():
        cax.axis('off')
    img = img / img.max()
    for jnd in range(4):
        if jnd == 3:
            ax[0, jnd].imshow(
                torch.squeeze(img[..., img.shape[-1] // 2]), vmin=vmin, vmax=vmax, cmap='gray', aspect='auto'
            )
            ax[0, jnd].set_title('Temporal profile')
        else:
            ax[0, jnd].imshow(torch.squeeze(img[jnd, ...]), vmin=vmin, vmax=vmax, cmap='gray')
            ax[0, jnd].set_title(f'Frame {jnd}')


# %% [markdown]
# #### Prepare data
# First, download and read-in the raw data. Then reconstruct coil-resolved images which are used to estimate the coil
# sensitivity maps. Finally, split the data into different dynamics.

# %%
# Download raw data in ISMRMRD format from zenodo into a temporary directory
dataset = '13207352'

tmp = tempfile.TemporaryDirectory()  # RAII, automatically cleaned up
data_folder = Path(tmp.name)
zenodo_get.zenodo_get([dataset, '-r', 5, '-o', data_folder])  # r: retries

# %%
# Read raw data and trajectory
kdata = KData.from_file(data_folder / '2D_GRad_map_t1.h5', KTrajectoryIsmrmrd())


# Calculate coil maps
reconstruction = DirectReconstruction(kdata, csm=None)
csm = CsmData.from_idata_walsh(reconstruction(kdata))

# Split data into dynamics
idx_dynamic = split_idx(torch.argsort(kdata.header.acq_info.acquisition_time_stamp[0, 0, :, 0]), 30, 0)
kdata_dynamic = kdata.split_k1_into_other(idx_dynamic, other_label='repetition')

# %% [markdown]
# #### Direct reconstruction
# Reconstruct dynamic images using the adjoint of the acquisition operator and sampling density compensation.

# %%
direct_reconstruction = DirectReconstruction(kdata_dynamic, csm=csm)
img_direct = direct_reconstruction(kdata_dynamic)
show_dynamic_images(img_direct.rss())


# %% [markdown]
# #### TV-regularized reconstruction using PDHG
# Reconstruct images by solving
#
# $ \mathcal{F}(x) = \frac{1}{2}||Ex - y||_2^2 + \lambda \| \nabla x \|_1 $
#
# using PDHG.
#
# Because we have 2D dynamic images, we can apply the TV-regularization along x,y and time.
# For this we set the regularization weight along dimensions -1 (x), -2 (y) and -5 (time).
#
# For this data we chose the regularization along space and time as 5e-6.
#
# For more information on this reconstruction method have a look at <project:tv_minimization_reconstruction_pdhg.ipynb>.

# %%
regularization_weight_space = 5e-6
regularization_weight_time = 5e-6
tv_reconstruction = TotalVariationRegularizedReconstruction(
    kdata_dynamic,
    csm=csm,
    max_iterations=100,
    regularization_weight=(regularization_weight_time, 0, 0, regularization_weight_space, regularization_weight_space),
)
img_tv = tv_reconstruction(kdata_dynamic)
show_dynamic_images(img_tv.rss())

# %% [markdown]
# #### TV-regularized reconstruction using ADMM
# In the above example, PDHG repeatedly applies the acquisition operator and its adjoint during the iterations, which
# is computationally demanding and hence takes a long time. Another option is to use the Alternating Direction Method
# of Multipliers (ADMM) [[S. Boyd et al, 2011](http://dx.doi.org/10.1561/2200000016)], which solves the general problem
#
# $ \min_x f(x) + g(z) \quad \text{subject to} \quad  Ax + Bz = c $
#
# If we use $f(x) = \lambda \| \nabla x \|_1$, $g(z)= \frac{1}{2}||Ez - y||_2^2$, $A = I$, $B= -I$ and $c = 0$
#
# then we can define a scaled form of the ADMM algorithm which solves
#
# $ \mathcal{F}(x) = \frac{1}{2}||Ex - y||_2^2 + \lambda \| \nabla x \|_1 $
#
# by doing
#
# $x_{k+1} = \argmin_x \lambda \| \nabla x \|_1 + \frac{\rho}{2}||x - z_k + u_k||_2^2$
#
# $z_{k+1} = \argmin_z \frac{1}{2}||Ez - y||_2^2 + \frac{\rho}{2}||x_{k+1} - z + u_k||_2^2$
#
# $u_{k+1} = u_k + x_{k+1} - z_{k+1}$
#
# The first step is TV-based denoising of $x$, the second step is a regularized iterative SENSE update of $z$ and the
# final step updates the dual variable $u$.

# %%
data_weight = 0.5
n_admm_iterations = 4
tv_denoising = TotalVariationDenoising(
    regularization_weight=(
        regularization_weight_time / data_weight,
        0,
        0,
        regularization_weight_space / data_weight,
        regularization_weight_space / data_weight,
    ),
    max_iterations=100,
)
regularized_iterative_sense = RegularizedIterativeSENSEReconstruction(
    kdata_dynamic, csm=csm, n_iterations=20, regularization_weight=data_weight
)
regularized_iterative_sense.dcf = None

img_z = img_direct.clone()
img_x = img_direct.clone()
img_u = torch.zeros_like(img_direct.data)
for _ in range(n_admm_iterations):
    # Denoising
    tv_denoising.initial_image = img_x.data
    img_x = tv_denoising(IData(img_z.data - img_u, img_direct.header))

    # Regularized iterative SENSE
    regularized_iterative_sense.regularization_data = img_x.data + img_u
    img_z = regularized_iterative_sense(kdata_dynamic)

    # Update u
    img_u = img_u + img_x.data - img_z.data

show_dynamic_images(img_x.rss(), title='ADMM 1: x')
show_dynamic_images(img_z.rss(), title='ADMM 1: z')
show_dynamic_images(img_u.abs(), title='ADMM 1: u')

img_tv_admm = img_z.rss()


# %% [markdown]
# #### TV-regularized reconstruction using ADMM
# Another option which avoids pdhg altogether is to use
#
# $f(x) = \lambda \| x \|_1$, $g(z)= \frac{1}{2}||Ez - y||_2^2$, $A = I$, $B= -\nabla$ and $c = 0$
#
# then we can define a scaled form of the ADMM algorithm which solves
#
# $ \mathcal{F}(x) = \frac{1}{2}||Ex - y||_2^2 + \lambda \| \nabla x \|_1 $
#
# by doing
#
# $x_{k+1} = \argmin_x \lambda \| x \|_1 + \frac{\rho}{2}||x - \nabla z_k + u_k||_2^2$
#
# $z_{k+1} = \argmin_z \frac{1}{2}||Ez - y||_2^2 + \frac{\rho}{2}||x_{k+1} - \nabla z + u_k||_2^2$
#
# $u_{k+1} = u_k + x_{k+1} - \nabla z_{k+1}$
#
# The first step is soft-thresholding of $x$: $S_{\lambda/\rho}(\nabla z_k - u_k)$, the second step is a regularized
# iterative SENSE update of $z$ and the final step updates the dual variable $u$.

# %%
nabla_operator = FiniteDifferenceOp(dim=(0, -2, -1), mode='forward')
data_weight = 0.5
regularization_weight = regularization_weight_time / (data_weight)

regularized_iterative_sense = RegularizedIterativeSENSEReconstruction(
    kdata_dynamic,
    csm=csm,
    n_iterations=10,
    regularization_weight=data_weight,
    regularization_op=nabla_operator,
)
regularized_iterative_sense.dcf = None

img_z = img_direct.clone()
img_u = torch.zeros_like(img_direct.data)
for _ in range(n_admm_iterations):
    # Denoising by soft-thresholding
    img_x_nabla = torch.view_as_complex(
        torch.nn.functional.softshrink(torch.view_as_real(nabla_operator(img_z.data)[0] - img_u), regularization_weight)
    )

    # Regularized iterative SENSE
    regularized_iterative_sense.regularization_data = img_x_nabla + img_u
    img_z = regularized_iterative_sense(kdata_dynamic)

    # Update u
    img_u = img_u + img_x_nabla - nabla_operator(img_z.data)[0]


show_dynamic_images(torch.sqrt(torch.sum(img_x_nabla**2, dim=0)).abs(), title='ADMM 2: x')
show_dynamic_images(img_z.rss(), title='ADMM 2: z')
show_dynamic_images(torch.sqrt(torch.sum(img_u**2, dim=0)).abs(), title='ADMM 2: u')

img_tv_admm_nabla = img_z.rss()

# %% [markdown]
# #### Compare different approaches

# %%
show_dynamic_images(img_direct.rss(), title='Direct')
show_dynamic_images(img_tv.rss(), title='PDHG')
show_dynamic_images(img_tv_admm, title='ADMM 1')
show_dynamic_images(img_tv_admm_nabla, title='ADMM 2')
