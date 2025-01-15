# %% [markdown]
# # Iterative SENSE Reconstruction of 2D golden angle radial data
# Here we use the IterativeSENSEReconstruction class to reconstruct images from ISMRMRD 2D radial data


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
# We use the IterativeSENSEReconstruction class to reconstruct images from 2D radial data.
# IterativeSENSEReconstruction solves the following reconstruction problem:
#
# Let's assume we have obtained the k-space data $y$ from an image $x$ with an acquisition model (Fourier transforms,
# coil sensitivity maps...) $A$ then we can formulate the forward problem as:
#
# $ y = Ax + n $
#
# where $n$ describes complex Gaussian noise. The image $x$ can be obtained by minimizing the functional $F$
#
# $ F(x) = ||W^{\frac{1}{2}}(Ax - y)||_2^2 $
#
# where $W^\frac{1}{2}$ is the square root of the density compensation function (which corresponds to a diagonal
# operator).
#
# Setting the derivative of the functional $F$ to zero and rearranging yields
#
# $ A^H W A x = A^H W y$
#
# which is a linear system $Hx = b$ that needs to be solved for $x$.
# %%
import mrpro

# %% [markdown]
# ##### Read-in the raw data
# We read the raw k-space data and the trajectory from the ISMRMRD file.
# %%
trajectory_calculator = mrpro.data.traj_calculators.KTrajectoryIsmrmrd()
kdata = mrpro.data.KData.from_file(data_folder / 'radial2D_402spokes_golden_angle_with_traj.h5', trajectory_calculator)

# %% [markdown]
# ##### Direct reconstruction for comparison
# For comparison, we first can carry out a direct reconstruction using the
# `~mrpro.algorithms.reconstruction.DirectReconstruction` class.
# See also <project:direct_reconstruction.py>.

# %%
direct_reconstruction = mrpro.algorithms.reconstruction.DirectReconstruction(kdata)
img_direct = direct_reconstruction(kdata)

# %% [markdown]
# ##### Iterative SENSE reconstruction
# Now let's use the `~mrpro.algorithms.reconstruction.IterativeSENSEReconstruction` class to reconstruct the image
# using the iterative SENSE algorithm. We can reuse the coil maps from the direct reconstruction.
# We use early stopping after 4 iterations.
# %%
# Set-up the iterative SENSE reconstruction
iterative_sense_reconstruction = mrpro.algorithms.reconstruction.IterativeSENSEReconstruction(
    kdata, csm=direct_reconstruction.csm, n_iterations=4
)
# Run the reconstruction
img = iterative_sense_reconstruction(kdata)

# %% [markdown]
# ### Behind the scenes
# We now peek behind the scenes to see how the iterative SENSE reconstruction is implemented.
# %% [markdown]
# ##### Set-up density compensation operator $W$
# We create need a density compensation operator $W$ to weight the loss.
#
# ```{note}
# Using a weighted loss in iterative SENSE is not necessary, and there has been some discussion about
# the benefits and drawbacks. Currently, the iterative SENSE reconstruction in mrpro uses a weighted loss.
# This will be changed in the future.
# ```
# %%
# The density compensation operator is calculated from the trajectory
dcf_operator = mrpro.data.DcfData.from_traj_voronoi(kdata.traj).as_operator()


# %% [markdown]
# ##### Set-up the acquisition model $A$
# We need `~mrpro.operators.FourierOp` and `~mrpro.data.CsmData` operators to set up the acquisition model $A$.
# This makes use of operator composition using the ``@`` operator.
# %%
# Define Fourier operator using the trajectory and header information in kdata
fourier_operator = mrpro.operators.FourierOp.from_kdata(kdata)

# Define coil sensitivity maps operator
img_coilwise = mrpro.data.IData.from_tensor_and_kheader(*fourier_operator.H(*dcf_operator(kdata.data)), kdata.header)
csm_operator = mrpro.data.CsmData.from_idata_walsh(img_coilwise).as_operator()

# Create the acquisition operator A
acquisition_operator = fourier_operator @ csm_operator

# %% [markdown]
# ##### Calculate the right-hand-side of the linear system
# Next, we need to calculate $b = A^H W y$.

# %%
(right_hand_side,) = (acquisition_operator.H @ dcf_operator)(kdata.data)

# %% [markdown]
# ##### Set-up the linear self-adjoint operator $H$
# We setup $H = A^H W A$, using the ``dcf_operator`` and ``acquisition_operator``.

# %%
operator = acquisition_operator.H @ dcf_operator @ acquisition_operator

# %% [markdown]
# ##### Run conjugate gradient
# Finally, we solve the linear system $Hx = b$ using the conjugate gradient method.
# Again, we use early stopping after 4 iterations. Instead, we could also use a tolerance
# to stop the iterations when the residual is below a certain threshold.

# %%
img_manual = mrpro.algorithms.optimizers.cg(
    operator, right_hand_side, initial_value=right_hand_side, max_iterations=4, tolerance=0.0
)

# %% [markdown]
# ##### Display the results
# We can now compare the results of the iterative SENSE reconstruction with the direct reconstruction.
# Both versions, the one using the `~mrpro.algorithms.reconstruction.IterativeSENSEReconstruction` class
# and the manual implementation should result in identical images.
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
# Display the reconstructed image
show_images(
    img_direct.rss()[0, 0],
    img.rss()[0, 0],
    img_manual.abs()[0, 0, 0],
    titles=['Direct', 'Iterative SENSE', 'Manual Iterative SENSE'],
)
# %% [markdown]
# ### Check for equal results
#  inally, we check if two images are really identical.
# %%
# If the assert statement did not raise an exception, the results are equal.
assert torch.allclose(img.data, img_manual)
# %% [markdown]
# ### Next steps
# We can also reconstruct undeersampled data or use a regularization term in the optimization problem.
# For the latter, see the example in <project:iterative_sense_reconstruction_with_regularization.ipynb>.
