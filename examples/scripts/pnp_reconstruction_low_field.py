# %% [markdown]
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

tmp = tempfile.TemporaryDirectory()  # RAII, automatically cleaned up
data_folder = Path(tmp.name)
zenodo_get.download(
    record='19661402',
    retry_attempts=5,
    output_dir=data_folder,
    file_glob=('LLR.zip',),
    access_token=os.environ.get('ZENODO_TOKEN'),
)
with zipfile.ZipFile(data_folder / Path('LLR.zip'), 'r') as zip_ref:
    zip_ref.extractall(data_folder)


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
# ### Compare the results
# We now compare the results of the direct reconstruction and the denoised images.

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
pnp_recon = mrpro.algorithms.reconstruction.PlugAndPlayPriorsReconstruction(
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
img_pnp = pnp_recon(kdata)

#%%
import importlib

if not importlib.util.find_spec("mrpro"):
    %pip install ismrmrd==1.14.2
    # %pip install mrpro[notebooks]

if not importlib.util.find_spec("SNRAware"):
    !git clone https://github.com/microsoft/SNRAware.git
    !cd SNRAware && pip install .
# %%
!wget --directory-prefix=./small/ https://huggingface.co/microsoft/SNRAware/resolve/main/small/snraware_small_model.pts
!wget --directory-prefix=./small/ https://huggingface.co/microsoft/SNRAware/resolve/main/small/snraware_small_model.yaml
#%%
from omegaconf import OmegaConf
from snraware.projects.mri.denoising.inference import apply_model
from snraware.projects.mri.denoising.inference_model import load_scripted_model
import numpy as np
# direct_recon = mrpro.algorithms.reconstruction.DirectReconstruction(kdata)
# idata = direct_recon(kdata)
# idat = idata.data.cpu().numpy().squeeze()
# image = idat[2 * idat.shape[0] // 3]
# noise = idat[12]

# image_normalized = image / np.std(noise)

# fig, ax = plt.subplots(1, 3, figsize=(3 * 6, 6))
# ax[0].imshow(np.rot90(np.abs(image)), cmap="gray")
# ax[1].imshow(np.rot90(np.abs(noise)), cmap="gray")
# ax[2].imshow(np.rot90(np.abs(image_normalized)), cmap="gray")

#%%
# set up the model
from einops import rearrange

# image_normalized = rearrange(image_normalized, "y x -> x y 1")
gmap = np.ones([img_direct.shape[0], img_direct.shape[1], 1])

model_parameter_path = "./small/snraware_small_model.pts"
model_config_path = "./small/snraware_small_model.yaml"

model = load_scripted_model(model_parameter_path)
config = OmegaConf.load(model_config_path)

batch_size = 1
cutout = tuple(config.dataset.cutout_shape)
if cutout is not None:
    cutout = tuple(cutout)
overlap = (16, 16, 8)
#%%
# denoise the image slices
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
# print("Using device:", device)

# scaling_factor = 0.5

# image_denoised = apply_model(
#     model=model,
#     data=image_normalized,
#     gmap=gmap,
#     scaling_factor=scaling_factor,
#     cutout=cutout,
#     overlap=overlap,
#     batch_size=batch_size,
#     device=device,
#     verbose=True,
# )
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
        # scaling_factor=scaling_factor,
        cutout=cutout,
        overlap=overlap,
        batch_size=batch_size,
        device=device,
        verbose=True,
    )
    denoised_twh = rearrange(denoised_hwt, "H W T -> T W H")
    return torch.as_tensor(denoised_twh, device=img.device, dtype=img.dtype)


pnp_recon = mrpro.algorithms.reconstruction.PlugAndPlayPriorsReconstruction(
    kdata=kdata,
    denoiser=snraware_denoiser,
    admm_regularization_strength=0.02,
    max_iterations=10,
    max_iterations_cg=10,
    tolerance=1e-6,
    tolerance_cg=1e-6,
)
img_pnp = pnp_recon(kdata)
#%%
# see the collapsed cell above for the implementation of show_images
slice_pos = img_direct.shape[-3] // 2
show_images(
    img_direct.rss().squeeze()[slice_pos],
    img_tv_denoised.rss().squeeze()[slice_pos],
    img_pnp.rss().squeeze()[slice_pos],
    titles=[
        'Direct',
        'TV-Denoising',
        'PnP Reconstruction',
    ],
)

# %%
