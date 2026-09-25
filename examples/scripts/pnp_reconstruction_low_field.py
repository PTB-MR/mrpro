# %% [markdown]
# # Plug-and-Play (PnP) Reconstruction of low-field MRI data
# # Plug-and-Play (PnP) Reconstruction of low-field MRI data

# %% [markdown]
# ### Image reconstruction
# Here, we use the Plug-and-Play (PnP) priors for model based reconstruction
# [[Venkatakrishnan, Bouman \& Wohlberg, IEEE GlobalSIP 2013](https://doi.org/10.1109/GlobalSIP.2013.6737048)].
#
# Let $y$ denote the k-space data of the image $x_{\mathrm{true}}$ sampled with an acquisition model $A$
# (Fourier transform, coil sensitivity maps, ...), i.e the forward problem is given as
#
# $ y = Ax_{\mathrm{true}} + n, $
#
# where $n$ describes complex Gaussian noise.
#
# The formulation of a regularized image reconstruction with the general regulariser $\beta s(v)$ is
#
# $ min_{x}  \frac{1}{2}||Ax - y||_2^2 + \beta s(x) $
#
# which can be rewritten using variable splitting to
#
# $ min_{x, v}  \frac{1}{2}||Ax - y||_2^2 + \beta s(v) $ such that $ x= v$.
#
# The augmented Lagrangian is then
#
# $ \mathcal{F}(x,v,u) = \frac{1}{2}||Ax - y||_2^2 +  \beta s(v) + \frac{\lambda}{2}||x-v+u||_2^2 -
#  \frac{\lambda}{2}||u||_2^2 \quad \quad \quad (1)$
#
# for the variables x and v and the scaled dual variable u. An ADMM algorithm can then be used to solve this in three
# steps:
#
# $x_{k+1} = \argmin_x  \frac{1}{2}||Ax - y||_2^2 + \frac{\lambda}{2}||x-v+u||_2^2 \quad \quad \quad (2a)$
#
# $v_{k+1} = \argmin_v \beta s(v) + \frac{\lambda}{2}||x-v+u||_2^2  \quad \quad \quad (2b)$
#
# $u_{k+1} = u_k + (x_{k+1} - v_{k+1})  \quad \quad \quad (2c)$

# %% [markdown]
# ### Load data
# We will use the low-field data obtained on a 50 mT Halbach-based MRI system at the Leiden University Medical Center.

# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show download details"}
# Download raw data from Zenodo
import os
import tempfile
import zipfile
from pathlib import Path

import zenodo_get

# tmp = tempfile.TemporaryDirectory()  # RAII, automatically cleaned up
# data_folder = Path(tmp.name)
# zenodo_get.download(
#     record='19661402',
#     retry_attempts=5,
#     output_dir=data_folder,
#     file_glob=('LLR.zip',),
#     access_token=os.environ.get('ZENODO_TOKEN'),
# )
# with zipfile.ZipFile(data_folder / Path('LLR.zip'), 'r') as zip_ref:
#     zip_ref.extractall(data_folder)
# %%
data_folder = Path('/tmp/tmpvs57g2j9')
# %% [markdown]
# ### Direct Reconstruction
# We do a simple direct reconstruction of the data.

# %%
import mrpro

kdata = mrpro.data.KData.from_file(
    data_folder / 'LLR/noise_corr_off/9003/IR_T1w_TSE_underS_R1p7.h5',
    mrpro.data.traj_calculators.KTrajectoryCartesian(),
)
recon = mrpro.algorithms.reconstruction.DirectReconstruction(kdata)
img_direct = recon(kdata)

# %% [markdown]
# ### TV-based Denoising
# We apply a TV-based denoiser to the image data.

# %%
img_tv_denoised = mrpro.algorithms.total_variation_denoising(
    img_direct, regularization_dim=(-3, -2, -1), regularization_weight=0.02, tolerance=1e-6
)

# %% [markdown]
# ### Wavelet-based Denoising
# We apply a wavelet-based denoiser to the image data.
#%%
img_wavelet_denoised = mrpro.algorithms.wavelet_denoising(
    img_direct, regularization_dim=(-3, -2, -1), regularization_weight=0.02, wavelet_name='db4', level=None
)
# %% [markdown]
# ### Compare the results
# We now compare the results of the direct reconstruction and the denoised images.

# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show plotting details"}
import matplotlib.pyplot as plt
import torch


def show_images(
    *images: torch.Tensor,
    titles: list[str] | None = None,
    nrows: int = 1,
    ncols: int | None = None,
    empty_positions: set[tuple[int, int]] | None = None,
) -> None:
    """Plot images."""
    n_images = len(images)
    ncols = ncols or n_images
    empty_positions = empty_positions or set()
    _, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(ncols * 3, nrows * 3))
    image_index = 0
    for row in range(nrows):
        for col in range(ncols):
            axes[row][col].axis('off')
            if (row, col) in empty_positions or image_index >= n_images:
                continue
            axes[row][col].imshow(images[image_index], cmap='gray')
            if titles:
                axes[row][col].set_title(titles[image_index])
            image_index += 1
    plt.show()


# %%
pnp_recon_tv = mrpro.algorithms.reconstruction.PlugAndPlayPriorsReconstruction(
    kdata=kdata,
    denoiser=lambda img: mrpro.algorithms.total_variation_denoising(
        img, regularization_dim=(-3, -2, -1), regularization_weight=0.02, tolerance=1e-6
    ),
    admm_regularization_strength=0.02,
    max_iterations=10,
    max_iterations_cg=10,
    tolerance=1e-6,
    tolerance_cg=1e-6,
)
img_pnp_tv = pnp_recon_tv(kdata)
#%%
pnp_recon_wavelet = mrpro.algorithms.reconstruction.PlugAndPlayPriorsReconstruction(
    kdata=kdata,
    denoiser=lambda img: mrpro.algorithms.wavelet_denoising(
        img, regularization_dim=(-3, -2, -1), regularization_weight=0.02, wavelet_name='db4', level=None
    ),
    admm_regularization_strength=0.02,
    max_iterations=10,
    max_iterations_cg=10,
    tolerance=1e-6,
    tolerance_cg=1e-6,
)
img_pnp_wavelet = pnp_recon_wavelet(kdata)
#%%
# add patch based denoisng
patch_based_dicts = "/echo/allgemein/projects/MRpro/pre_trained-dictionaries/patch_lasso_filters/"
patch_based_file = "K288_d4x6x6.pt"
patch_based_dict = torch.load(patch_based_dicts + patch_based_file).to(torch.complex64)
dict_op = mrpro.operators.DictionaryOp(patch_based_dict, dim=(-3, -2, -1)).H

patch_based_recon = mrpro.algorithms.patch_based_denoising(
    idata=img_direct,
    dictionary_op=dict_op,
    patch_dim=(-3, -2, -1),
    patch_size=(4, 6, 6),
    regularization_weight=0.1,
)
#%%
# add patch based denoisng in Plug-and-Play reconstruction
pnp_recon_patch_based = mrpro.algorithms.reconstruction.PlugAndPlayPriorsReconstruction(
    kdata=kdata,
    denoiser=lambda img: mrpro.algorithms.patch_based_denoising(
        idata=img,
        dictionary_op=dict_op,
        patch_dim=(-3, -2, -1),
        patch_size=(4, 6, 6),
        regularization_weight=0.1,
    ),
    admm_regularization_strength=0.5,
    max_iterations=10,
    max_iterations_cg=10,
    tolerance=1e-6,
    tolerance_cg=1e-6,
)
img_pnp_patch_based = pnp_recon_patch_based(kdata)
#%%
# add convolutional synthesis dictionary based denoisng
path_name = (
    "/echo/allgemein/projects/MRpro/pre_trained-dictionaries/conv_synthesis_filters/"
)
file_name = "d_filter_sporco_K32_k11x11_lmbda1em01_fltlmbd2em01.pt"
kernel = torch.load(path_name + file_name)

low_pass_parameter = 1.0
regularization_weight = 3e-2
max_iterations_low_pass_filtering = 12
tolerance_low_pass_filtering = 1e-4

denoised_image_conv_synth = mrpro.algorithms.conv_synthesis_dictionary_denoising(
    img_direct,
    kernel,
    low_pass_parameter=low_pass_parameter,
    regularization_weight=regularization_weight,
    max_iterations_low_pass_filtering=max_iterations_low_pass_filtering,
    max_iterations_pgd=128,
)
#%%
# add convolutional synthesis dictionary based denoisng in Plug-and-Play reconstruction
pnp_recon_conv_synth = mrpro.algorithms.reconstruction.PlugAndPlayPriorsReconstruction(
    kdata=kdata,
    denoiser=lambda img: mrpro.algorithms.conv_synthesis_dictionary_denoising(
        img,
        kernel,
        low_pass_parameter=low_pass_parameter,
        regularization_weight=regularization_weight,
        max_iterations_low_pass_filtering=max_iterations_low_pass_filtering,
        max_iterations_pgd=128,
    ),
    admm_regularization_strength=0.02,
    max_iterations=10,
    max_iterations_cg=10,
    tolerance=1e-6,
    tolerance_cg=1e-6,
)
img_pnp_conv_synth = pnp_recon_conv_synth(kdata)
#%%
import importlib

# if not importlib.util.find_spec("mrpro"):
#     %pip install ismrmrd==1.14.2
#     # %pip install mrpro[notebooks]

# if not importlib.util.find_spec("SNRAware"):
#     !git clone https://github.com/microsoft/SNRAware.git
#     !cd SNRAware && pip install .
# %%
!wget --directory-prefix=./small/ https://huggingface.co/microsoft/SNRAware/resolve/main/small/snraware_small_model.pts
!wget --directory-prefix=./small/ https://huggingface.co/microsoft/SNRAware/resolve/main/small/snraware_small_model.yaml
#%%
from omegaconf import OmegaConf
from snraware.projects.mri.denoising.inference import apply_model
from snraware.projects.mri.denoising.inference_model import load_scripted_model
import numpy as np
from einops import rearrange

# image_normalized = rearrange(image_normalized, "y x -> x y 1")
gmap = np.ones([img_direct.shape[0], img_direct.shape[1], 1])

model_parameter_path = "./small/snraware_small_model.pts"
model_config_path = "./small/snraware_small_model.yaml"

model = load_scripted_model(model_parameter_path)
#%%
# use SNRaware in Plug-and-Play reconstruction
# build a wrapper around the SNRaware denoiser to use it in the Plug-and-Play reconstruction
device = torch.device("cpu")


def snraware_denoiser(img: torch.Tensor) -> torch.Tensor:
    """Apply SNRaware while preserving the MRpro tensor layout and type."""
    image_hwt = rearrange(img.squeeze(), "T W H -> H W T").detach().cpu().numpy()
    denoised_hwt = apply_model(
        model=model,
        data=image_hwt,
        gmap=np.ones(image_hwt.shape[:2] + (1,)),
        batch_size=1,
        device=device,
        verbose=True,
    )
    denoised_twh = rearrange(denoised_hwt, "H W T -> T W H")
    return torch.as_tensor(denoised_twh, device=img.device, dtype=img.dtype)

# %%
pnp_recon_SNRaware_1_it = mrpro.algorithms.reconstruction.PlugAndPlayPriorsReconstruction(
    kdata=kdata,
    denoiser=snraware_denoiser,
    admm_regularization_strength=0.02,
    max_iterations=1,
    max_iterations_cg=10,
    tolerance=1e-6,
    tolerance_cg=1e-6,
)
img_pnp_SNRaware_1_it = pnp_recon_SNRaware_1_it(kdata)
#%%
pnp_recon_SNRaware = mrpro.algorithms.reconstruction.PlugAndPlayPriorsReconstruction(
    kdata=kdata,
    denoiser=snraware_denoiser,
    admm_regularization_strength=0.02,
    max_iterations=10,
    max_iterations_cg=10,
    tolerance=1e-6,
    tolerance_cg=1e-6,
)
img_pnp_SNRaware = pnp_recon_SNRaware(kdata)
#%%
# see the collapsed cell above for the implementation of show_images
slice_pos = img_direct.shape[-3] // 2
show_images(
    img_direct.rss().squeeze()[slice_pos],
    img_tv_denoised.rss().squeeze()[slice_pos],
    img_pnp_tv.rss().squeeze()[slice_pos],
    img_pnp_SNRaware_1_it.rss().squeeze()[slice_pos],
    img_pnp_SNRaware.rss().squeeze()[slice_pos],
    titles=[
        'Direct',
        'TV-Denoising',
        'PnP TV-Denoising',
        'SNRaware Denoising',
        'PnP SNRaware-Denoising'
    ],
)
# %%
# change plotting to show the difference images.
# First row shows Direct, TV-Denoising and SNRaware_1_it,
# second row shows PnP TV-Denoising and PnP SNRaware-Denoising
# third row shows the difference images between TV-Denoising and PnP TV-Denoising,
# and between SNRaware_1_it and PnP SNRaware-Denoising
slice_pos = img_direct.shape[-3] // 2
show_images(
    img_direct.rss().squeeze()[slice_pos],
    img_tv_denoised.rss().squeeze()[slice_pos],
    img_wavelet_denoised.rss().squeeze()[slice_pos],
    patch_based_recon.rss().squeeze()[slice_pos],
    denoised_image_conv_synth.rss().squeeze()[slice_pos],
    img_pnp_SNRaware_1_it.rss().squeeze()[slice_pos],
    img_pnp_tv.rss().squeeze()[slice_pos],
    img_pnp_wavelet.rss().squeeze()[slice_pos],
    img_pnp_patch_based.rss().squeeze()[slice_pos],
    img_pnp_conv_synth.rss().squeeze()[slice_pos],
    img_pnp_SNRaware.rss().squeeze()[slice_pos],
    (img_tv_denoised.rss().squeeze()[slice_pos] - img_pnp_tv.rss().squeeze()[slice_pos]),
    (img_wavelet_denoised.rss().squeeze()[slice_pos] - img_pnp_wavelet.rss().squeeze()[slice_pos]),
    (patch_based_recon.rss().squeeze()[slice_pos] - img_pnp_patch_based.rss().squeeze()[slice_pos]),
    (denoised_image_conv_synth.rss().squeeze()[slice_pos] - img_pnp_conv_synth.rss().squeeze()[slice_pos]),
    (img_pnp_SNRaware_1_it.rss().squeeze()[slice_pos] - img_pnp_SNRaware.rss().squeeze()[slice_pos]),
    titles=[
        'Direct',
        'TV-Denoising',
        'Wavelet-Denoising',
        'Patch-based Denoising',
        'Convolutional Synthesis Denoising',
        'SNRaware Denoising',
        'PnP TV-Denoising',
        'PnP Wavelet-Denoising',
        'PnP Patch-based Denoising',
        'PnP Convolutional Synthesis Denoising',
        'PnP SNRaware-Denoising',
        'Diff. TV vs PnP TV',
        'Diff. Wavelet vs\n PnP Wavelet',
        'Diff. Patch-based vs\n PnP Patch-based',
        'Diff. Conv. Synthesis vs\n PnP Conv. Synthesis',
        'Diff. SNRaware vs\n PnP SNRaware'
    ],
    nrows=3,
    ncols=6,
    empty_positions={(1, 0), (2, 0)},
)

# %%
