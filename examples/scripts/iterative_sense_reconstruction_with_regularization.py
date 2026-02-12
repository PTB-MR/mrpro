# %% [markdown]
# # Regularized iterative SENSE reconstruction of 2D golden angle radial data
# Here we use the `~mr2.algorithms.reconstruction.RegularizedIterativeSENSEReconstruction` class to reconstruct
# undersampled images from 2D radial data.

# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show download details"}
# Download raw data from Zenodo
import os
import tempfile
from pathlib import Path

import zenodo_get

tmp = tempfile.TemporaryDirectory()  # RAII, automatically cleaned up
data_folder = Path(tmp.name)
zenodo_get.download(
    record='14617082', retry_attempts=5, output_dir=data_folder, access_token=os.environ.get('ZENODO_TOKEN')
)

# %% [markdown]
# ### Image reconstruction
# We use the `~mr2.algorithms.reconstruction.RegularizedIterativeSENSEReconstruction` class to reconstruct images
# from 2D radial data. It solves the following reconstruction problem:
#
# Let's assume we have obtained the k-space data $y$ from an image $x$ with an acquisition model (Fourier transforms,
# coil sensitivity maps...) $A$ then we can formulate the forward problem as:
#
# $ y = Ax + n $
#
# where $n$ describes complex Gaussian noise. The image $x$ can be obtained by minimizing the functional $F$
#
# $ F(x) = ||Ax - y||_2^2 + \lambda||Bx - x_\mathrm{reg}||_2^2$
#
# where $\lambda$ is the strength of the regularization, $B$ is a linear operator and $x_\mathrm{reg}$ is a
# regularization image.
# With this functional $F$ we obtain a solution which is close to $x_\mathrm{reg}$ and to the acquired data $y$.
#
# Setting the derivative (see https://www.matrixcalculus.org) of the functional $F$ to zero and rearranging yields
#
# $ (A^H A + l B^H B) x = A^H y + \lambda B^H x_\mathrm{reg}$
#
# which is a linear system $Hx = b$ that needs to be solved for $x$.
#
# One important question of course is, what to use as $x_\mathrm{{reg}}$ and $B$. For dynamic images (e.g. cine MRI)
# low-resolution dynamic images or high-quality static images have been proposed.
# In recent years, the output of neural networks has also been used, i.e. $x_{\mathrm{reg}} = u_{\theta}(x_0)$
# $B=\mathrm{Id}$ for a pre-trained network $u_{\theta}$ and initial image $x_0$ [[Kofler et al. 2020](https://doi.org/10.1088/1361-6560/ab990e)].
#
# In this example we are going to use a high-quality image to regularize the reconstruction of an undersampled image.
# Both images are obtained from the same data acquisition - one using all the acquired data ($x_{\mathrm{reg}}$),
# and one using only parts of it ($x$). This is, of course, an unrealistic case but it will allow us to demonstrate
# the effect of the regularization.
#
# ```{note}
# In [Pruessmann et al. 2001](https://doi.org/10.1002/mrm.1241) the k-space density is used to reweight the
# loss to achieve faster convergence. This increases reconstruction error, see [Ong F., Uecker M., Lustig M. 2020](https://doi.org/10.1109/TMI.2019.2954121).
# We follow a recommendation by [Fessler and Noll](https://ece-classes.usc.edu/ee591/library/Fessler-Iterative%20Reconstruction.pdf)
# and use the DCF to obtain a good starting point.
# ```

# %% [markdown]
# ### Reading of both fully sampled and undersampled data
# We read the raw data and the trajectory from the ISMRMRD file.
# We load both, the fully sampled and the undersampled data.
# The fully sampled data will be used to estimate the coil sensitivity maps and as a regularization image.
# The undersampled data will be used to reconstruct the image.

# %%
# Read the raw data and the trajectory from ISMRMRD file
import mr2

kdata_fullysampled = mr2.data.KData.from_file(
    data_folder / 'radial2D_402spokes_golden_angle_with_traj.h5',
    mr2.data.traj_calculators.KTrajectoryIsmrmrd(),
)
kdata_undersampled = mr2.data.KData.from_file(
    data_folder / 'radial2D_24spokes_golden_angle_with_traj.h5',
    mr2.data.traj_calculators.KTrajectoryIsmrmrd(),
)

# %% [markdown]
# ##### Obtain image $x_{\mathrm{reg}}$ from fully sampled data
# We first reconstruct the fully sampled image to use it as a regularization image.
# In a real-world scenario, we would not have this image and would have to use a low-resolution image as a prior, or use
# a neural network to estimate the regularization image.

# %%
# Estimate coil maps. Here we use the fully sampled data to estimate the coil sensitivity maps.
# In a real-world scenario, we would either a calibration scan (e.g. a separate fully sampled scan) to estimate the coil
# sensitivity maps or use ESPIRiT or similar methods to estimate the coil sensitivity maps from the undersampled data.
direct_reconstruction = mr2.algorithms.reconstruction.DirectReconstruction(kdata_fullysampled)
csm = direct_reconstruction.csm
assert csm is not None

# unregularized iterative SENSE reconstruction of the fully sampled data
iterative_sense_reconstruction = mr2.algorithms.reconstruction.IterativeSENSEReconstruction(
    kdata_fullysampled, csm=csm, n_iterations=3
)
img_iterative_sense = iterative_sense_reconstruction(kdata_fullysampled)

# %% [markdown]
# ##### Image $x$ from undersampled data
# We now reconstruct the undersampled image using the fully sampled image first without regularization,
# and with a regularization image.

# %%
# Unregularized iterative SENSE reconstruction of the undersampled data
iterative_sense_reconstruction = mr2.algorithms.reconstruction.IterativeSENSEReconstruction(
    kdata_undersampled, csm=csm, n_iterations=6
)
img_us_iterative_sense = iterative_sense_reconstruction(kdata_undersampled)

# %%
# Regularized iterative SENSE reconstruction of the undersampled data

regularized_iterative_sense_reconstruction = mr2.algorithms.reconstruction.RegularizedIterativeSENSEReconstruction(
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

# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show plotting details"}
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
# perform them manually. This also demonstrates how to use the `mr2` operators and algorithms
# to build your own reconstruction pipeline.

# %% [markdown]
# ##### Set-up the acquisition model $A$
#
# This is very similar to <project:iterative_sense_reconstruction_radial2D.ipynb> .
# For more details, please refer to that notebook.

# %%
fourier_operator = mr2.operators.FourierOp.from_kdata(kdata_undersampled)
csm_operator = csm.as_operator()
acquisition_operator = fourier_operator @ csm_operator

# %% [markdown]
# ##### Set-up the right-hand side $b$
# We calculate $b = A^H y + \lambda B^H x_\mathrm{reg}$, using the identity operator as $B$ and $\lambda = 1.0$.

# %%
regularization_weight = 1.0
regularization_image = img_iterative_sense.data
regularization_operator = mr2.operators.IdentityOp()

(regularization,) = (regularization_weight * regularization_operator.H)(regularization_image)
(right_hand_side,) = (acquisition_operator.H)(kdata_undersampled.data)
right_hand_side = right_hand_side + regularization

# %% [markdown]
# ##### Set-up the linear self-adjoint operator $H$
# We define $H = A^H A + \lambda B^HB$. We can use `~mr2.operators.LinearOperator.gram` to get an efficient
# implementation of $A^H A$. We use the `~mr2.operators.IdentityOp` and make
# use of operator addition using ``+`` and multiplication using ``*``.
# The resulting operator is a `~mr2.operators.LinearOperator` object.

# %%
operator = acquisition_operator.gram + mr2.operators.IdentityOp() * regularization_weight

# %% [markdown]
# ##### Run conjugate gradient
# We solve the linear system $Hx = b$ using the conjugate gradient method.
# Here we use a density compensated adjoint reconstruction to obtain a good starting point,
# $x_0 = A^H W y$ with $W$ being the density compensation operator.
# We use a tolerance of $1e-7$ for the residual as a stopping criterion.

# %%
dcf_operator = mr2.data.DcfData.from_traj_voronoi(kdata_undersampled.traj).as_operator()
(initial_value,) = (acquisition_operator.H @ dcf_operator)(kdata_undersampled.data)
(img_manual,) = mr2.algorithms.optimizers.cg(operator, right_hand_side, initial_value=initial_value, tolerance=1e-7)

# %% [markdown]
# #####  Display the reconstructed image
# We can now compare our 'manual' reconstruction with the regularized iterative SENSE reconstruction
# obtained using `~mr2.algorithms.reconstruction.RegularizedIterativeSENSEReconstruction`.

# %%
show_images(
    img_us_regularized_iterative_sense.rss()[0, 0],
    img_manual.abs()[0, 0, 0],
    titles=['RegularizedIterativeSense', 'Manual'],
)

# %% [markdown]
# We can verify the results by comparing the actual image data.
# If the assert statement does not raise an exception, the results are equal.

# %%
torch.testing.assert_close(img_us_regularized_iterative_sense.data, img_manual)

# %% [markdown]
# ### Next steps
#
# We are cheating here because we used the fully sampled image as a regularization. In real world applications
# we would not have that. One option is to apply a low-pass filter to the undersampled k-space data to try to reduce the
# streaking artifacts and use that as a regularization image. Try that and see if you can also improve the image quality
# compared to the unregularized images.

# %%
