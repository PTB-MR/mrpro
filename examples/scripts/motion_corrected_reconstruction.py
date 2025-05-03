# %% [markdown]
# # Motion-corrected image reconstruction
# Physiological motion (e.g. due to breathing or the beating of the heart) is a common challenge for MRI. Motion during
# the data acquisition can lead to motion artifacts in the reconstructed image. Mathematically speaking, the motion
# occurring during data acquisition can be described by linear operators $M_m$ for each motion state $m$:
#
# $ y = \sum_m A_m M_m x_{\mathrm{true}} + n, $
#
# where $y$ is the acquired k-space data, $A_m$ is the acquisition operator describing which data was acquired in motion
# state $m$ and $n$ describes complex Gaussian noise.

# %% [markdown]
# There are different approaches of how to minimize the impact of motion. The simplest approach is to acquire data only
# in a single motion state by e.g. asking the subject to hold their breath or synchronize the acquisition with the heart
# beat. This reduces the above problem to
#
# $ y = A x_{\mathrm{true}} + n, $
#
# but this is not always possible and can also lead to long acquisition times.
#
# A more efficient approach is to acquire data in different motion states, estimate $M_m$ and solve the above problem.
# This is often referred to motion-corrected image reconstruction (MCIR). For some examples have a look at
# [Kolbitsch et al., JNM 2017](http://doi.org/10.2967/jnumed.115.171728),
# [Ippoliti et al., MRM 2019](http://doi.wiley.com/10.1002/mrm.27867) or
# [Mayer et al., JNM 2021](http://doi.org/10.1007/s00259-020-05180-4)
#
# Here we will show how to do a MCIR of a free-breathing acquisition of the thorax following these steps:
# 1. Estimate a respiratory self-navigator from the acquired k-space data
# 2. Use the self-navigator to separate the data into different breathing states
# 3. Reconstruct dynamic images of different breathing states
# 4. Estimate non-rigid motion fields from the dynamic images
# 5. Use the motion fields to obtain a motion-corrected image
#
# To achieve high image quality a TV-regularized image reconstruction is used here. To safe time we will use only a few
# iterations. Increase 'n_iterations_tv' in the code to 100 to get a better image quality.

# %%
n_iterations_tv = 100

# %% [markdown]
# ### Data acquisition
# The data was acquired with a Golden Radial Phase Encoding (GRPE) sampling scheme
# [[Prieto et al., MRM 2010](http://doi.org/10.1002/mrm.22446)]. This sampling scheme combines a Cartesian readout with
# radial phase encoding in the 2D ky-kz plane. The central k-space line (i.e. 1D projection of the object along the
# foot-head direction) is acquired repeatedly and can be used as a respiratory self-navigator. This sequence was
# implemented in pulseq and also available as a seq-file. The FOV of the scan was 288 x 288 x 288 $mm^3$ with an
# isotropic resolution of 1.9 mm.
# ```{note}
# To keep reconstruction times short, we used a short acquisition of less than one minute. We also only split the data
# into 4 motion states with rather large overlap (sliding-window) between motion states. For reliable motion correction
# at least 6 motion states should be used.
# ```


# %%
# ### Imports
# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show download details"}
# Download raw data and pre-calculated motion fields from zenodo into a temporary directory
import tempfile
from pathlib import Path

import numpy as np
import torch
import zenodo_get
from einops import rearrange, repeat
from mrpro.algorithms.optimizers import cg, pdhg
from mrpro.algorithms.reconstruction import IterativeSENSEReconstruction
from mrpro.data import CsmData, IData, KData, KTrajectory, Rotation
from mrpro.data.traj_calculators import KTrajectoryRpe
from mrpro.operators import (
    AveragingOp,
    FastFourierOp,
    FiniteDifferenceOp,
    FourierOp,
    GridSamplingOp,
    LinearOperatorMatrix,
    ProximableFunctionalSeparableSum,
    SensitivityOp,
)
from mrpro.operators.functionals import L1NormViewAsReal, L2NormSquared, ZeroFunctional
from mrpro.utils import split_idx, unsqueeze_right

dataset = '15288250'

tmp = tempfile.TemporaryDirectory()  # RAII, automatically cleaned up
data_folder = Path(tmp.name)
zenodo_get.zenodo_get([dataset, '-r', 5, '-o', data_folder])  # r: retries


# %%
def tv_reg_reco(kdata, csm, img_initial, reg_weight=0.1, reg_weight_t=0.1, n_iterations=100):
    fourier_operator = FourierOp.from_kdata(kdata)
    csm_operator = SensitivityOp(csm)
    acquisition_operator = fourier_operator @ csm_operator

    if img_initial.data.shape[0] == 1:
        tv_dim = (-3, -2, -1)
        regularization_weight = torch.tensor([reg_weight, reg_weight, reg_weight])
    else:
        tv_dim = (-5, -3, -2, -1)
        regularization_weight = torch.tensor([reg_weight_t, reg_weight, reg_weight, reg_weight])

    nabla_operator = FiniteDifferenceOp(dim=tv_dim, mode='forward')
    l2 = 0.5 * L2NormSquared(target=kdata.data)
    l1 = L1NormViewAsReal(weight=unsqueeze_right(regularization_weight, kdata.data.ndim))

    f = ProximableFunctionalSeparableSum(l2, l1)
    g = ZeroFunctional()
    K = LinearOperatorMatrix(((acquisition_operator,), (nabla_operator,)))

    initial_values = (img_initial.data.clone(),)

    (img_pdhg,) = pdhg(f=f, g=g, operator=K, initial_values=initial_values, max_iterations=n_iterations)
    return IData(data=img_pdhg, header=img_initial.header)


def rearrange_k2_k1_into_k1(kdata: KData) -> KData:
    """Rearrange kdata from (... k2 k1 ...) to (... 1 (k2 k1) ...).

    Parameters
    ----------
    kdata
        K-space data (other coils k2 k1 k0)

    Returns
    -------
        K-space data (other coils 1 (k2 k1) k0)
    """
    # Rearrange data
    kdat = rearrange(kdata.data, '... coils k2 k1 k0->... coils 1 (k2 k1) k0')

    # Rearrange trajectory
    ktraj = rearrange(kdata.traj.as_tensor(), 'dim ... k2 k1 k0-> dim ... 1 (k2 k1) k0')

    header = kdata.header.apply(
        lambda field: rearrange(field, 'dim ... k2 k1 k0-> dim ... 1 (k2 k1) k0')
        if isinstance(field, Rotation | torch.Tensor)
        else field
    )

    return KData(header, kdat, KTrajectory.from_tensor(ktraj))


# %% [markdown]
# ### Motion-corrupted image reconstruction
# As a first step we will reconstruct the image using all the acquired data which will lead to an image corrupted by
# respiratory motion.
# ```{note}
# To reduce the file size we have already applied coil compression reducing the 21 physical coils to 6 compressed coils.
# We also removed the readout oversampling.
# ```
# %%
kdata = KData.from_file(data_folder / 'grpe_t1_free_breathing.mrd', KTrajectoryRpe(angle=torch.pi * 0.618034))

# Calculate coil maps
csm_maps = CsmData.from_kdata_inati(kdata, smoothing_width=5, downsampled_size=64)

#  SENSE reconstruction
iterative_sense = IterativeSENSEReconstruction(kdata, csm=csm_maps)
img = iterative_sense.forward(kdata)

# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show plotting details"}
import matplotlib.pyplot as plt


def show_views(image: torch.Tensor) -> None:
    """Plot coronal, transversal and sagittal view."""
    image = torch.squeeze(image / image.max())
    image_views = [image[:, 92, :], torch.fliplr(image[:, :, 90]), image[100, :, :]]
    _, axes = plt.subplots(1, 3, squeeze=False, figsize=(12, 6))
    for idx, (view, title) in enumerate(zip(image_views, ['Coronal', 'Transversal', 'Sagittal'], strict=False)):
        axes[0, idx].imshow(torch.rot90(view), vmin=0, vmax=0.25, cmap='grey')
        axes[0, idx].set_title(title, fontsize=18)
        axes[0, idx].set_xticks([])
        axes[0, idx].set_yticks([])
    plt.show()


# %%
# Visualize anatomical views of 3D image
show_views(img.rss())


# %% [markdown]
# ### 1. Respiratory self-navigator
# We are going to obtain a self-navigator from the k-space center line ky = kz = 0 following these steps:
# - Get all readout lines for ky = kz = 0
# - Apply a 1D FFT along the readout to get the projection of the object
# - Carry out SVD over all readout points and all coils to get the main signal components
# - Select the SVD component the largest frequency contribution between 0.2 and 0.5 Hz (realistic breathing frequency)


# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show self-navigator calculation"}
def get_respiratory_self_navigator_from_grpe(
    kdata: KData, respiratory_frequency_range: tuple[float, float] = (0.2, 0.5)
) -> torch.Tensor:
    """Get respiratory self-navigator from GRPE data set."""
    # Get all readout lines for ky = kz = 0
    ky0_kz0_idx = torch.where((kdata.traj.ky == 0) & (kdata.traj.kz == 0))
    navigator_data = kdata.data[ky0_kz0_idx[0], :, ky0_kz0_idx[2], ky0_kz0_idx[3], :]
    navigator_time = kdata.header.acq_info.acquisition_time_stamp[ky0_kz0_idx[0], 0, ky0_kz0_idx[2], ky0_kz0_idx[3], 0]

    # Apply a 1D FFT along the readout to get the projection of the object
    fft_op_1d = FastFourierOp(dim=(-1,))
    navigator_data = torch.abs(fft_op_1d(navigator_data)[0])

    # Carry out SVD over all readout points and all coils to get the main signal components
    navigator_data = rearrange(navigator_data, 'k1k2 coil k0 -> k1k2 (coil k0)')
    svd_navigator_data, _, _ = torch.linalg.svd(navigator_data - navigator_data.mean(dim=0, keepdim=True))

    # Select the SVD component the largest frequency contribution closest to the expected respiratory frequency
    dt = torch.mean(torch.diff(navigator_time, dim=0))
    fft_svd_navigator_data = torch.abs(torch.fft.fft(svd_navigator_data, dim=0))
    f_hz = torch.linspace(0, 1 / dt, svd_navigator_data.shape[0])
    fft_svd_navigator_data_in_resp_window = fft_svd_navigator_data[
        (f_hz >= respiratory_frequency_range[0]) & (f_hz <= respiratory_frequency_range[1]), :
    ]
    respiratory_navigator = svd_navigator_data[
        :, torch.argmax(torch.max(fft_svd_navigator_data_in_resp_window, dim=0)[0])
    ]

    # Interpolate navigator from k-space center ky=kz=0 to all phase encoding points
    return torch.as_tensor(
        np.interp(kdata.header.acq_info.acquisition_time_stamp[0, 0, 0, :, 0], navigator_time, respiratory_navigator)
    )


# %%
# To separate all phase encoding points into different motion states we combine ky and kz points along k1 first
kdata = rearrange_k2_k1_into_k1(kdata)

respiratory_navigator = get_respiratory_self_navigator_from_grpe(kdata)

plt.figure()
acquisition_time = kdata.header.acq_info.acquisition_time_stamp[0, 0, 0, :, 0]
plt.plot(acquisition_time - acquisition_time.min(), respiratory_navigator)
plt.xlabel('Acquisition time (s)')
plt.ylabel('Navigator signal (a.u.)')

# %% [markdown]
# ### 2. Split data into different breathing states
# The self-navigator is used to split the data into different motion phases. We use a sliding window approach to ensure
# we have got enough data in each motion state.

resp_idx = split_idx(
    torch.argsort(respiratory_navigator), int(kdata.data.shape[-2] * 0.36), int(kdata.data.shape[-2] * 0.18)
)
kdata_resp_resolved = kdata.split_k1_into_other(resp_idx, other_label='repetition')

n_points_per_motion_state = int(kdata.data.shape[-2] * 0.36)
sorted_idx = respiratory_navigator.argsort()
split_idx = sorted_idx.unfold(0, n_points_per_motion_state, n_points_per_motion_state//2)
kdata_resp_resolved = kdata[..., split_idx, :]

# %% [markdown]
# ### 3. Reconstruct dynamic images of different breathing states

# %%
recon_resp_resolved = IterativeSENSEReconstruction(kdata_resp_resolved, csm=csm_maps)
img_resp_resolved = recon_resp_resolved.forward(kdata_resp_resolved)

img_resp_resolved_tv = tv_reg_reco(
    kdata_resp_resolved,
    csm_maps,
    img_initial=img_resp_resolved,
    reg_weight=1e-7,
    reg_weight_t=2e-6,
    n_iterations=n_iterations_tv,
)


# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show plotting details"}
def show_motion_states(*images: torch.Tensor, ylabels: list[str] | None = None, slice_idx: int = 92) -> None:
    """Plot first and last motion state and difference image."""
    n_images = len(images)
    _, axes = plt.subplots(n_images, 3, squeeze=False, figsize=(n_images * 6, 8))
    [a.set_xticks([]) for a in axes.flatten()]
    [a.set_yticks([]) for a in axes.flatten()]
    for i in range(n_images):
        image = torch.squeeze(images[i] / images[i].max())
        axes[i, 0].imshow(torch.rot90(image[0, :, slice_idx, :]), vmin=0, vmax=0.2, cmap='grey')
        axes[i, 0].set_title('MS 1', fontsize=18)
        axes[i, 0].set_ylabel(ylabels[i], fontsize=18)
        axes[i, 1].imshow(torch.rot90(image[-1, :, slice_idx, :]), vmin=0, vmax=0.2, cmap='grey')
        axes[i, 1].set_title(f'MS {image.shape[0]}', fontsize=18)
        axes[i, 2].imshow(
            torch.rot90(torch.abs(image[0, :, slice_idx, :] - image[-1, :, slice_idx, :])),
            vmin=0,
            vmax=0.2,
            cmap='grey',
        )
        axes[i, 2].set_title(f'|MS 1 - MS {image.shape[0]}|', fontsize=18)
    plt.show()


# %%
show_motion_states(
    img_resp_resolved.rss(), img_resp_resolved_tv.rss(), ylabels=('Iterative SENSE', 'TV-regularization')
)

# %% [markdown]
# ### 4. Estimate the motion fields from the dynamic images
# The motion fields can be estimated with an image registration package such as
# [mirtk](https://mirtk.github.io/commands/register.html). Here we registered each of the dynamic respiratory phases to
# the first motion state using the free-form deformation (FFD) registration approach with a control point spacing of 9.
# Normalized mutual information was used as a similarity metric and a LogJac penalty weight of 0.001 was applied.
#
# We load the displacement fields and create a motion operator:

# %%
mf = torch.as_tensor(np.load(data_folder / 'grpe_t1_free_breathing_displacement_fields.npy'), dtype=torch.float32)
# The motion fields describe the motion transformation in voxel units. We need to convert them to the convention of
# the grid sampling operator, which is in [-1, 1] for each dimension.
for dim in range(3):
    mf[..., dim] /= kdata_resp_resolved.data.shape[-1] / 2
n_motion_states = mf.shape[0]
unity_matrix = repeat(torch.cat((torch.eye(3), torch.zeros(3, 1)), dim=1), 'h w->n_ms h w', n_ms=n_motion_states)
grid = torch.nn.functional.affine_grid(unity_matrix, (n_motion_states, 1, *mf.shape[1:-1]))
grid += mf
grid[grid > 1] = 1
grid[grid < -1] = -1
moco_op = GridSamplingOp(grid, input_shape=kdata.header.recon_matrix)

# %% [markdown]
# ### 5. Use the motion fields to obtain a motion-corrected image
# Now we obtain a motion-corrected image $x$ by minimizing the functionl $F$
#
# $ F(x) = ||\sum_m(A_mx - y_m)||_2^2 \quad$ with $\quad A_m = F_m M_m C $
#
# where $C$ describes the coil-sensitivity maps, $M_m$ is the motion transformation of motion state $m$ and $F_m$
# describes the Fourier transform of all of the k-space points obtained in motion state $m$.
#
# One way to solve this problem with MRpro is to use the respiratory-resolved k-space data from above. Because we used a
# sliding window approach to split the data into different motion states, this computationally a bit more demanding than
# it needs to be.

# %%
# Create acquisition operator
fourier_op = recon_resp_resolved.fourier_op
dcf_op = recon_resp_resolved.dcf.as_operator()
csm_op = SensitivityOp(csm_maps)
averaging_op = AveragingOp(dim=0)
acquisition_operator = fourier_op @ moco_op @ csm_op @ averaging_op.H

(initial_value,) = acquisition_operator.H(dcf_op(kdata_resp_resolved.data)[0])
(right_hand_side,) = acquisition_operator.H(kdata_resp_resolved.data)
operator = acquisition_operator.H @ acquisition_operator

# Minimize the functional
img_mcir = cg(operator, right_hand_side, initial_value=initial_value, max_iterations=30, tolerance=0.0)


# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show plotting details"}
def show_images(*images: torch.Tensor, titles: list[str] | None = None) -> None:
    """Plot images."""
    n_images = len(images)
    _, axes = plt.subplots(1, n_images, squeeze=False, figsize=(n_images * 3, 3))
    for i in range(n_images):
        image = torch.squeeze(images[i] / images[i].max())
        axes[0][i].imshow(torch.rot90(image[:, 93, :]), cmap='gray', vmin=0, vmax=0.18)
        axes[0][i].axis('off')
        if titles:
            axes[0][i].set_title(titles[i])
    plt.show()


# %%
show_images(img.rss(), img_mcir.abs(), titles=('Uncorrected', 'MCIR'))
