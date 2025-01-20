# %% [markdown]
# # Different ways to obtain the Trajectory
# This example builds upon the <project:direct_reconstruction.ipynb> example and demonstrates three ways
# to obtain the trajectory information required for image reconstruction:
# - using the trajectory that is stored in the ISMRMRD file
# - calculating the trajectory using the radial 2D trajectory calculator
# - calculating the trajectory from the pulseq sequence file using the PyPulseq trajectory calculator


# %% tags=["hide-cell"]
# Download raw data from Zenodo
import tempfile
from pathlib import Path

import mrpro
import torch
import zenodo_get

dataset = '14617082'

tmp = tempfile.TemporaryDirectory()  # RAII, automatically cleaned up
data_folder = Path(tmp.name)
zenodo_get.zenodo_get([dataset, '-r', 5, '-o', data_folder])  # r: retries

# %% [markdown]
# ### Using KTrajectoryIsmrmrd - Trajectory saved in ISMRMRD file
# Passing an instance of `~mrpro.data.traj_calculators.KTrajectoryIsmrmrd` to
# when loading the data tells the `~mrpro.data.KData` object to use the trajectory
# that is stored in the ISMRMRD file.
# ```{note}
# Often the trajectory information has not been stored in the ISMRMRD file,
# in which case loading the trajectory this way will raise an error.
# ```

# %%
# Read the raw data and the trajectory from ISMRMRD file
kdata = mrpro.data.KData.from_file(
    data_folder / 'radial2D_402spokes_golden_angle_with_traj.h5',
    mrpro.data.traj_calculators.KTrajectoryIsmrmrd(),
)

# Reconstruct image
reconstruction = mrpro.algorithms.reconstruction.DirectReconstruction(kdata)
img_using_ismrmrd_traj = reconstruction(kdata)

# %% [markdown]
# ### Using KTrajectoryRadial2D - Specific trajectory calculator
# For some common trajectories, we provide specific trajectory calculators.
# These calculators often require only a few parameters to be specified,
# such as the angle between spokes in the radial trajectory. Other parameters
# will be taken from the ISMRMRD file.
# This will calculate the trajectory using the radial 2D trajectory calculator.
# ```{note}
# You can also implement your own trajectory calculator by subclassing
# `~mrpro.data.traj_calculators.KTrajectoryCalculator`.
# ```

# %%
# Read raw data and calculate trajectory using KTrajectoryRadial2D
golden_angle = torch.pi * 0.618034
kdata = mrpro.data.KData.from_file(
    data_folder / 'radial2D_402spokes_golden_angle_with_traj.h5',
    mrpro.data.traj_calculators.KTrajectoryRadial2D(golden_angle),
)

# Reconstruct image
reconstruction = mrpro.algorithms.reconstruction.DirectReconstruction(kdata)
img_using_rad2d_traj = reconstruction(kdata)

# %% [markdown]
# ### Using KTrajectoryPulseq - Trajectory from pulseq sequence file
# This will calculate the trajectory from the pulseq sequence file
# using the PyPulseq trajectory calculator. This method
# requires the pulseq sequence file that was used to acquire the data.
# The path to the sequence file is provided as an argument to KTrajectoryPulseq.

# %%
# Read raw data and calculate trajectory using KTrajectoryPulseq
seq_path = data_folder / 'radial2D_402spokes_golden_angle.seq'
kdata = mrpro.data.KData.from_file(
    data_folder / 'radial2D_402spokes_golden_angle_with_traj.h5',
    mrpro.data.traj_calculators.KTrajectoryPulseq(seq_path),
)

# Reconstruct image
reconstruction = mrpro.algorithms.reconstruction.DirectReconstruction(kdata)
img_using_pulseq_traj = reconstruction(kdata)

# %% [markdown]
# ### Plot the different reconstructed images
# All three images are reconstructed using the same raw data and should look almost identical.
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
    img_using_ismrmrd_traj.rss()[0, 0],
    img_using_rad2d_traj.rss()[0, 0],
    img_using_pulseq_traj.rss()[0, 0],
    titles=['KTrajectoryIsmrmrd', 'KTrajectoryRadial2D', 'KTrajectoryPulseq'],
)

# %% [markdown]
# Tada! We have successfully reconstructed images using three different trajectory calculators.
# ```{note}
# Which of these three methods is the best depends on the specific use case:
# If a trajectory is already stored in the ISMRMRD file, it is the most convenient to use.
# If a pulseq sequence file is available, the trajectory can be calculated using the PyPulseq trajectory calculator.
# Otherwise, a trajectory calculator needs to be implemented for the specific trajectory used.
# ```
