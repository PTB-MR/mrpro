# %% [markdown]
# # Regularized Iterative SENSE Reconstruction of 2D golden angle radial data
# Here we use the `mrpro.algorithms.reconstruction.RegularizedIterativeSENSEReconstruction` class to reconstruct
# undersampled images from 2D radial data.
# %% tags=["hide-cell"]
# Download raw data from Zenodo
import tempfile
from pathlib import Path

import zenodo_get

dataset = '14617082'

tmp = tempfile.TemporaryDirectory()  # RAII, automatically cleaned up
data_folder = Path(tmp.name)
zenodo_get.zenodo_get([dataset, '-r', 5, '-o', data_folder])  # r: retries
# %% [markdown]
# ### Image reconstruction
# We use the `~mrpro.algorithms.reconstruction.RegularizedIterativeSENSEReconstruction` class to reconstruct images
# from 2D radial data. It solves the following reconstruction problem:
#
# Let's assume we have obtained the k-space data $y$ from an image $x$ with an acquisition model (Fourier transforms,
# coil sensitivity maps...) $A$ then we can formulate the forward problem as:
#
# $ y = Ax + n $
#
# where $n$ describes complex Gaussian noise. The image $x$ can be obtained by minimizing the functionl $F$
#
# $ F(x) = ||W^{\frac{1}{2}}(Ax - y)||_2^2 $
#
# where $W^\frac{1}{2}$ is the square root of the density compensation function (which corresponds to a diagonal
# operator). Because this is an ill-posed problem, we can add a regularization term to stabilize the problem and obtain
# a solution with certain properties:
#
# $ F(x) = ||W^{\frac{1}{2}}(Ax - y)||_2^2 + l||Bx - x_{reg}||_2^2$
#
# where $l$ is the strength of the regularization, $B$ is a linear operator and $x_{reg}$ is a regularization image.
# With this functional $F$ we obtain a solution which is close to $x_{reg}$ and to the acquired data $y$.
#
# Setting the derivative of the functional $F$ to zero and rearranging yields
#
# $ (A^H W A + l B) x = A^H W y + l x_{reg}$
#
# which is a linear system $Hx = b$ that needs to be solved for $x$.
#
# One important question of course is, what to use for $x_{reg}$. For dynamic images (e.g. cine MRI) low-resolution
# dynamic images or high-quality static images have been proposed. In recent years, also the output of neural-networks
# has been used as an image regulariser.
#
# In this example we are going to use a high-quality image to regularize the reconstruction of an undersampled image.
# Both images are obtained from the same data acquisition (one using all the acquired data ($x_{reg}$) and one using
# only parts of it ($x$)). This of course is an unrealistic case but it will allow us to study the effect of the
# regularization.


# %% [markdown]
# ### Reading of both fully sampled and undersampled data
# We read the raw data and the trajectory from the ISMRMRD file.
# We load both, the fully sampled and the undersampled data.
# The fully sampled data will be used to estimate the coil sensitivity maps and as a regularization image.
# The undersampled data will be used to reconstruct the image.

# %%
# Read the raw data and the trajectory from ISMRMRD file
import mrpro

kdata_fullysampled = mrpro.data.KData.from_file(
    data_folder / 'radial2D_402spokes_golden_angle_with_traj.h5',
    mrpro.data.traj_calculators.KTrajectoryIsmrmrd(),
)
kdata_undersampled = mrpro.data.KData.from_file(
    data_folder / 'radial2D_24spokes_golden_angle_with_traj.h5',
    mrpro.data.traj_calculators.KTrajectoryIsmrmrd(),
)
# %% [markdown]
# ##### Image $x_{reg}$ from fully sampled data
# We first reconstruct the fully sampled image to use it as a regularization image.
# In a real-world scenario, we would not have this image and would have to use a low-resolution image as a prior, or use
# a neural network to estimate the regularization image.

# %%
# Estimate coil maps. Here we use the fully sampled data to estimate the coil sensitivity maps.
# In a real-world scenario, we would either a calibration scan (e.g. a separate fully sampled scan) to estimate the coil
# sensitivity maps or use ESPIRiT or similar methods to estimate the coil sensitivity maps from the undersampled data.
direct_reconstruction = mrpro.algorithms.reconstruction.DirectReconstruction(kdata_fullysampled)
csm = direct_reconstruction.csm
assert csm is not None

# unregularized iterative SENSE reconstruction of the fully sampled data
iterative_sense_reconstruction = mrpro.algorithms.reconstruction.IterativeSENSEReconstruction(
    kdata_fullysampled, csm=csm, n_iterations=3
)
img_iterative_sense = iterative_sense_reconstruction(kdata_fullysampled)

# %% [markdown]
# ##### Image $x$ from undersampled data
# We now reconstruct the undersampled image using the fully sampled image first without regularization,
# and with with an regularization image.

# %%
# Unregularized iterative SENSE reconstruction of the undersampled data
iterative_sense_reconstruction = mrpro.algorithms.reconstruction.IterativeSENSEReconstruction(
    kdata_undersampled, csm=csm, n_iterations=6
)
img_us_iterative_sense = iterative_sense_reconstruction(kdata_undersampled)

# %%
# Regularized iterativ SENSE reconstruction of the undersampled data

regularized_iterative_sense_reconstruction = mrpro.algorithms.reconstruction.RegularizedIterativeSENSEReconstruction(
    kdata_undersampled,
    csm=csm,
    n_iterations=6,
    regularization_data=img_iterative_sense.data,
    regularization_weight=1.0,
)
img_us_regularized_iterative_sense = regularized_iterative_sense_reconstruction(kdata_undersampled)

# %% [markdown]
# ##### Display the results
# Besides the fully sampled image, we display two undersampled images:
# The first one is obtained by unregularized iterative SENSE, the second one using regularization.
# %% tags=["hide-cell"]
import matplotlib.pyplot as plt
import torch


def show_images(*images: torch.Tensor, titles: list[str] | None = None) -> None:
    """Plot images."""
    n_images = len(images)
    _, axes = plt.subplots(1, n_images, squeeze=False, figsize=(n_images * 3, 3))
    for i in range(n_images):
        axes[0][i].imshow(images[i], cmap='gray')
        axes[0][i].axis('off')
        if titles:
            axes[0][i].set_title(titles[i])
    plt.show()


# %%
show_images(
    img_iterative_sense.rss()[0, 0],
    img_us_iterative_sense.rss()[0, 0],
    img_us_regularized_iterative_sense.rss()[0, 0],
    titles=['Fully sampled', 'Iterative SENSE R=20', 'Regularized Iterative SENSE R=20'],
)

# %% [markdown]
# ### Behind the scenes
# We now investigate the steps that are done in the regularized iterative SENSE reconstruction and
# perform them manually. This also demonstrates how to use the `mrpro` operators and algorithms
# to build your own reconstruction pipeline.

# %% [markdown]
# ##### Set-up the density compensation operator $W$ and acquisition model $A$
#
# This is very similar to <project:iterative_sense_reconstruction.ipynb> .
# For more details, please refer to that notebook.
# %%
dcf_operator = mrpro.data.DcfData.from_traj_voronoi(kdata_undersampled.traj).as_operator()
fourier_operator = mrpro.operators.FourierOp.from_kdata(kdata_undersampled)
csm_operator = csm.as_operator()
acquisition_operator = fourier_operator @ csm_operator

# %% [markdown]
# ##### Calculate the right-hand-side of the linear system
# We calculated $b = A^H W y + l x_{reg}$.
# Here, we make use of operator composition using ``@``.

# %%
regularization_weight = 1.0
regularization_image = img_iterative_sense.data

(right_hand_side,) = (acquisition_operator.H @ dcf_operator)(kdata_undersampled.data)
right_hand_side = right_hand_side + regularization_weight * regularization_image

# %% [markdown]
# ##### Set-up the linear self-adjoint operator $H$
# We define $H= A^H W A + l$. We use the `~mrpro.operators.IdentityOp` and make
# use of operator composition using ``@``, addition using ``+`` and multiplication using ``*``.
# The resulting operator is a `~mrpro.operators.LinearOperator` object.

# %%
operator = (
    acquisition_operator.H @ dcf_operator @ acquisition_operator + mrpro.operators.IdentityOp() * regularization_weight
)

# %% [markdown]
# ##### Run conjugate gradient
# We solve the linear system $Hx = b$ using the conjugate gradient method.
# Here, we use early stopping after 8 iterations. Instead, we could also use a tolerance to stop the iterations when
# the residual is small enough.
# %%
img_manual = mrpro.algorithms.optimizers.cg(
    operator, right_hand_side, initial_value=right_hand_side, max_iterations=8, tolerance=0.0
)
# %% [markdown]
# #####  Display the reconstructed image
# We can now compare our 'manual' reconstruction with the regularized iterative SENSE reconstruction
# obtained using `~mrpro.algorithms.reconstruction.RegularizedIterativeSENSEReconstruction`.

# %%
show_images(
    img_us_regularized_iterative_sense.rss()[0, 0],
    img_manual.abs()[0, 0, 0],
    titles=['Regularized Iterative SENSE R=20', '"Manual" Regularized Iterative SENSE R=20'],
)
# %% [markdown]
# We can also check if the results are equal by comparing the actual image data.
# If the assert statement does not raise an exception, the results are equal.
# %%
assert torch.allclose(img_us_regularized_iterative_sense.data, img_manual)

# %% [markdown]
# ### Next steps
#
# We are cheating here because we used the fully sampled image as a regularization. In real world applications
# we would not have that. One option is to apply a low-pass filter to the undersampled k-space data to try to reduce the
# streaking artifacts and use that as a regularization image. Try that and see if you can also improve the image quality
# compared to the unregularised images.
