# %% [markdown]
# # TV-regularized reconstruction

# %%
import matplotlib.pyplot as plt
import torch
from mrpro.algorithms.reconstruction import DirectReconstruction, TotalVariationRegularizedReconstruction, TotalVariationDenoising, RegularizedIterativeSENSEReconstruction
from mrpro.data import CsmData, KData, IData
from mrpro.data.traj_calculators import KTrajectoryIsmrmrd
from mrpro.utils import split_idx


# %% [markdown]
# We define a plotting function to look at some of the dynamic frames and also a plot along the time dimension.

# %%
def show_dynamic_images(img: torch.Tensor, vmin: float = 0, vmax: float = 0.8) -> None:
    """Show dynamic images.

    Parameters
    ----------
    img
        image tensor to be displayed
    vmin
        vmin for display
    vmax
        vmax for display
    """
    fig, ax = plt.subplots(1, 4, squeeze=False, figsize=(16, 4))
    img = img / img.max()
    for jnd in range(4):
        if jnd == 3:
            ax[0, jnd].imshow(torch.squeeze(img[..., 64]), vmin=vmin, vmax=vmax, cmap='gray', aspect='auto')
        else:
            ax[0, jnd].imshow(torch.squeeze(img[jnd, ...]), vmin=vmin, vmax=vmax, cmap='gray')


# %% [markdown]
# #### Prepare data
# First, download and read-in the raw data. Then reconstruct coil-resolved images which are used to estimate the coil
# sensitivity maps. Finally, split the data into different dynamics.

# %%
# Load in the data from the ISMRMRD file
from pathlib import Path

fname = Path('/Users/kolbit01/Documents/PTB/Data/mrpro/raw/2D_GRad_map_t1_traj_2s.h5')
kdata = KData.from_file(fname, KTrajectoryIsmrmrd())
kdata.header.recon_matrix.x = 128
kdata.header.recon_matrix.y = 128

# Calculate coil maps
reconstruction = DirectReconstruction(kdata, csm=None)
csm = CsmData.from_idata_walsh(reconstruction(kdata))

# Split data into dynamics
idx_dynamic = split_idx(torch.argsort(kdata.header.acq_info.acquisition_time_stamp[0, 0, :, 0]), 30, 0)
kdata_dynamic = kdata.split_k1_into_other(idx_dynamic, other_label='repetition')

# %% [markdown]
# #### Direct Reconstruction
# Reconstruct dynamic images using the adjoint of the acquisition operator and sampling density compensation.

# %%
direct_reconstruction = DirectReconstruction(kdata_dynamic, csm=csm)
img_direct = direct_reconstruction(kdata_dynamic)
show_dynamic_images(img_direct.rss())

# %% [markdown]
# #### TV-Regularized Reconstruction using PDHG
# Reconstruct images by solving
#
# $ \mathcal{F}(x) = \frac{1}{2}||Ax - y||_2^2 + \lambda \| \nabla x \|_1 $
#
# using PDHG.
#
# Because we have 2D dynamic images we can apply the TV-regularisation along x,y and time.
# For this we set the regularisation weight along dimensions -1 (x), -2 (y) and -5 (time).
#
# For more information on this reconstruction method have a look at the tv_minimization_reconstruction_pdhg example.

# %%
tv_reconstruction = TotalVariationRegularizedReconstruction(
    kdata_dynamic, csm=csm, n_iterations=100, regularization_weight=(0.1, 0, 0, 0.1, 0.1)
)
img_tv = tv_reconstruction(kdata_dynamic)
show_dynamic_images(img_tv.rss())

# %% [markdown]
# #### TV-Regularized Reconstruction using ADMM
# In the above example we need to apply the acquisition operator during the PDHG iterations which is computationally
# demanding and hence takes a long time. Another option is to use the Alternating Direction Method of Multipliers (ADMM)
# which solves the general problem
#
# $ \min_x f(x) + g(z) \quad \text{subject to} \quad  Ax + Bz = c $
#
# If we use $f(x) = \lambda \| \nabla x \|_1$, $g(z)= \frac{1}{2}||Az - y||_2^2$, $A = I$, $B= -I$ and $c = 0$
#
# then we can define a scaled form of the ADMM algorithm which solves
#
# $ \mathcal{F}(x) = \frac{1}{2}||Ax - y||_2^2 + \lambda \| \nabla x \|_1 $
#
# by doing
#
# $x_{k+1} = argmin_x \lambda \| \nabla x \|_1 + \frac{\rho}{2}||x - z_k + u_k||_2^2$
#
# $z_{k+1} = argmin_z \frac{1}{2}||Az - y||_2^2 + \frac{\rho}{2}||x_{k+1} - z + u_k||_2^2$
#
# $u_{k+1} = u_k + x_{k+1} - z_{k+1}$
#
# The first step is TV-based denoising of $x$, the second step is a regularized iterative SENSE update of $z$ and the
# final step updates the helper variable $u$.


# %%
data_weight = 0.5
tv_denoising = TotalVariationDenoising(regularization_weight=(0.1/data_weight, 0, 0, 0.1/data_weight, 0.1/data_weight), n_iterations=100)
regularised_iterative_sense = RegularizedIterativeSENSEReconstruction(kdata_dynamic, csm=csm, n_iterations=10, regularization_weight=data_weight)
n_adam_iterations = 4
img_z = img_direct.clone()
img_u = torch.zeros_like(img_direct.data)
for _ in range(n_adam_iterations):
    # Denoising
    img_x = tv_denoising(IData(img_z.data - img_u, img_direct.header))
    show_dynamic_images(img_x.rss())

    # Iterative SENSE
    regularised_iterative_sense.regularization_data = img_x.data + img_u
    img_z = regularised_iterative_sense(kdata_dynamic)
    show_dynamic_images(img_z.rss())

    # Update u
    img_u = img_u + img_x.data - img_z.data
    show_dynamic_images(img_u[:,0,...].abs())


# %%
show_dynamic_images(img_x.rss())
show_dynamic_images(img_z.rss())
show_dynamic_images(img_u[:,0,...].abs())
