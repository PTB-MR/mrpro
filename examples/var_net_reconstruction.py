# %% [markdown]
# # Iterative SENSE Reconstruction of 2D golden angle radial data
# Here we use the IterativeSENSEReconstruction class to reconstruct images from ISMRMRD 2D radial data
# %%
# define zenodo URL of the example ismrmd data
zenodo_url = 'https://zenodo.org/records/10854057/files/'
fname = 'pulseq_radial_2D_402spokes_golden_angle_with_traj.h5'
# %%
# Download raw data


# data_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.h5')
# response = requests.get(zenodo_url + fname, timeout=30)
# data_file.write(response.content)
# data_file.flush()

from pathlib import Path

data_file = Path(
    '/echo/_allgemein/projects/MRpro/MRPro/2024_03_19/pulseq_data/radial_2D_with_traj/radial_2D_402spokes_golden_angle/pulseq_radial_2D_402spokes_golden_angle_with_traj.h5'
)

FLAG_UNET = False

import mrpro
import torch

# %% [markdown]
# #### Read-in the raw data

# %%
# Use the trajectory that is stored in the ISMRMRD file
trajectory = mrpro.data.traj_calculators.KTrajectoryIsmrmrd()
# Load in the Data from the ISMRMRD file
kdata = mrpro.data.KData.from_file(data_file, trajectory)
kdata.header.recon_matrix.x = 256
kdata.header.recon_matrix.y = 256

# %% [markdown]
# #### Data undersampling

# %%
idx_us = torch.arange(0, 20)[None, :]
kdata_us = kdata.split_k1_into_other(idx_us, other_label='repetition')

# %% [markdown]
# #### Standard reconstruction for comparison

# %%
# Direct reconstruction
direct_reconstruction = mrpro.algorithms.reconstruction.DirectReconstruction(kdata)
img_direct = direct_reconstruction(kdata)

direct_reconstruction_us = mrpro.algorithms.reconstruction.DirectReconstruction(kdata_us, csm=direct_reconstruction.csm)
img_us_direct = direct_reconstruction_us(kdata_us)

# Iterative SENSE reconstruction
iterative_sense_reconstruction = mrpro.algorithms.reconstruction.IterativeSENSEReconstruction(
    kdata, csm=direct_reconstruction.csm, n_iterations=4
)
img_itSENSE = iterative_sense_reconstruction(kdata)

iterative_sense_reconstruction_us = mrpro.algorithms.reconstruction.IterativeSENSEReconstruction(
    kdata_us, csm=direct_reconstruction.csm, n_iterations=20
)
img_us_itSENSE = iterative_sense_reconstruction_us(kdata_us)

import matplotlib.pyplot as plt

vis_im = [img_direct.rss(), img_itSENSE.rss(), img_us_direct.rss(), img_us_itSENSE.rss()]
vis_title = ['Direct', 'Iterative SENSE', 'Direct R=20', 'Iterative SENSE R = 20']
fig, ax = plt.subplots(2, 4, squeeze=False, figsize=(16, 8))
for ind in range(4):
    ax[0, ind].imshow(vis_im[ind][0, 0, ...])
    ax[0, ind].set_title(vis_title[ind])
    ax[1, ind].imshow(torch.abs(vis_im[ind][0, 0, ...] - vis_im[0][0, 0, ...]))

# %% [markdown]
# #### Define network

# %% [markdown]
# ##### Block with convolution layers

# %%
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    A simple block of 2D convolutional layers
    """

    def __init__(self, n_ch_in, n_ch_out, n_convs, kernel_size=3, bias=True, padding_mode='zeros'):
        super().__init__()

        padding = int(np.floor(kernel_size / 2))
        conv_block_list = []
        conv_block_list.extend([nn.Conv2d(n_ch_in, n_ch_out, kernel_size, padding=padding, bias=bias, padding_mode=padding_mode), nn.LeakyReLU()])

        for i in range(n_convs - 1):
            conv_block_list.extend(
                [nn.Conv2d(n_ch_out, n_ch_out, kernel_size, padding=padding, bias=bias, padding_mode=padding_mode), nn.LeakyReLU()]
            )
        self.conv_block = nn.Sequential(*conv_block_list)

    def forward(self, x):
        return self.conv_block(x)


# %% [markdown]
# ##### Encoder


# %%
class Encoder(nn.Module):
    def __init__(
        self, n_ch_in, n_enc_stages, n_convs_per_stage, n_filters, kernel_size=3, bias=True, padding_mode='zeros'
    ):
        super().__init__()
        n_ch_list = [n_ch_in]
        for ne in range(n_enc_stages):
            n_ch_list.append(int(n_filters) * 2**ne)

        self.enc_blocks = nn.ModuleList(
            [
                ConvBlock(n_ch_list[i], n_ch_list[i + 1], n_convs_per_stage, kernel_size, bias, padding_mode)
                for i in range(len(n_ch_list) - 1)
            ]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        for block in self.enc_blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features


# %% [markdown]
# ##### Decoder


# %%
class Decoder(nn.Module):
    def __init__(self, n_ch_in, n_dec_stages, n_convs_per_stage, kernel_size=3, bias=False, padding_mode='zeros'):
        super().__init__()

        n_ch_list = []
        for ne in range(n_dec_stages):
            n_ch_list.append(int(n_ch_in * (1 / 2) ** ne))

        self.interp_mode = 'bilinear'

        padding = int(np.floor(kernel_size / 2))
        self.upconvs = nn.ModuleList(
            [
                nn.Conv2d(
                    n_ch_list[i],
                    n_ch_list[i + 1],
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=bias,
                    padding_mode=padding_mode,
                )
                for i in range(len(n_ch_list) - 1)
            ]
        )
        self.dec_blocks = nn.ModuleList(
            [
                ConvBlock(n_ch_list[i], n_ch_list[i + 1], n_convs_per_stage, kernel_size, bias, padding_mode)
                for i in range(len(n_ch_list) - 1)
            ]
        )

    def forward(self, x, encoder_features):
        for i in range(len(self.dec_blocks)):
            enc_features = encoder_features[i]
            enc_features_shape = enc_features.shape
            x = nn.functional.interpolate(x, enc_features_shape[2:], mode=self.interp_mode, align_corners=False)
            x = self.upconvs[i](x)
            x = torch.cat([x, enc_features], dim=1)
            x = self.dec_blocks[i](x)
        return x


# %% [markdown]
# ##### UNet


# %%
class UNet(nn.Module):
    def __init__(
        self,
        n_ch_in=2,
        n_ch_out=2,
        n_enc_stages=3,
        n_convs_per_stage=2,
        n_filters=16,
        kernel_size=3,
        bias=True,
        residual_connection=True,
        padding_mode='zeros',
    ):
        super().__init__()
        self.encoder = Encoder(n_ch_in, n_enc_stages, n_convs_per_stage, n_filters, kernel_size, bias, padding_mode)
        self.decoder = Decoder(
            n_filters * (2 ** (n_enc_stages - 1)), n_enc_stages, n_convs_per_stage, kernel_size, bias, padding_mode
        )

        self.c1x1 = nn.Conv2d(n_filters, n_ch_out, kernel_size=1, padding=0, bias=bias)

        self.residual_connection = residual_connection
        if residual_connection:
            if n_ch_in != n_ch_out:
                raise ValueError(
                    'For using the residual connection, the number of input and output channels of the \
                    network must be the same.\
                    Given {n_ch_in} and {n_ch_out}.'
                )

    def forward(self, x):
        enc_features = self.encoder(x)
        dec = self.decoder(enc_features[-1], enc_features[::-1][1:])
        out = self.c1x1(dec)
        if self.residual_connection:
            out = out + x
        return out


# %% [markdown]
# ##### NUFFT Cascade

# %%
from einops import rearrange


class NUFFTCascade(nn.Module):
    def __init__(
        self,
        operator_norm: float,
        acquisition_operator: mrpro.operators.FourierOp,
        dcf_operator: mrpro.operators.DensityCompensationOp,
        factor: float,
        max_iter: int = 10,
        n_filters=64,
        kernel_size=3,
    ) -> None:
        super(NUFFTCascade, self).__init__()
        self.acquisition_operator = acquisition_operator
        self.dcf_operator = dcf_operator
        self.factor = factor
        self.max_iter = max_iter
        self.operator_norm = operator_norm
        self.unet = UNet(
            n_ch_in=2,
            n_ch_out=2,
            n_enc_stages=3,
            n_convs_per_stage=2,
            n_filters=n_filters,
            bias=False,
            kernel_size=kernel_size,
            residual_connection=False,
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=kernel_size, padding='same', bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size, padding='same', bias=False
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size, padding='same', bias=False
            ),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=n_filters, out_channels=2, kernel_size=kernel_size, padding='same', bias=False),
        )

    def apply_CNN(self, x):
        other, coil, nz, ny, nx = x.shape
        x = rearrange(torch.view_as_real(x), 'other coil z y x ch -> (other coil z) ch y x')

        x = self.cnn(x)

        x = torch.view_as_complex(
            rearrange(
                x, '(other coil z) ch y x -> other coil z y x ch', other=other, coil=coil, z=nz, y=ny, x=nx, ch=2
            ).contiguous()
        )

        return x

    def apply_UNet(self, x):
        other, coil, nz, ny, nx = x.shape
        x = rearrange(torch.view_as_real(x), 'other coil z y x ch -> (other coil z) ch y x')

        x = self.unet(x)

        x = torch.view_as_complex(
            rearrange(
                x, '(other coil z) ch y x -> other coil z y x ch', other=other, coil=coil, z=nz, y=ny, x=nx, ch=2
            ).contiguous()
        )

        return x

    def forward(self, x, k_space_data):
        for _ in range(self.max_iter):
            x = x.clone()
            step_size = 2.0 / self.operator_norm**2
            res_acq = self.acquisition_operator.H(
                self.dcf_operator(self.acquisition_operator(x)[0] - (k_space_data.data))[0]
            )

            if FLAG_UNET:
                x = x - self.factor * step_size * res_acq[0] + self.apply_UNet(x)
            else:
                x = x - self.factor * step_size * res_acq[0] + self.apply_CNN(x)
        return x


# %% [markdown]
# ##### NUFFT Cascade

# %%
n_epochs = 20
n_itera = 50
learning_rate = 1e-4
batch_size = 1

lossesssss = []
tot_loss = 0
list_xreco = []

acquisition_op = direct_reconstruction_us.fourier_op @ direct_reconstruction_us.csm.as_operator()
operator_norm = acquisition_op.operator_norm(img_us_direct.data, dim=None)
model = NUFFTCascade(
    acquisition_operator=acquisition_op,
    dcf_operator=direct_reconstruction_us.dcf.as_operator(),
    operator_norm=operator_norm,
    factor=0.98,
    max_iter=4,
)
mse_loss = lambda x: F.mse_loss(torch.view_as_real(img_itSENSE.data), torch.view_as_real(x))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# %%
# Artificial data
if False:
    dcf_operator = mrpro.operators.DensityCompensationOp(
        2 * torch.sqrt(direct_reconstruction_us.dcf.data)
    )  ## 2*torch.sqrt() or not
    acquisition_operator = dcf_operator @ direct_reconstruction_us.fourier_op  # No need to add csm
    model.acquisition_operator = acquisition_operator

    initial_value = torch.randn(1, 1, 1, 256, 256, dtype=torch.complex64)
    op_norm_estimate = acquisition_operator.operator_norm(initial_value=initial_value, dim=None)
    model.operator_norm = op_norm_estimate

    kdata = acquisition_operator(img_itSENSE.data)[0]
    (xu,) = acquisition_operator.H(kdata)
    kdata_uus = mrpro.data.KData(data=kdata, header=kdata_us.header, traj=kdata_us.traj)


# %%
xu = img_us_itSENSE.data.clone()
for epoch in range(n_epochs):
    for i in range(n_itera):
        optimizer.zero_grad()
        xreco = model(xu, kdata_us.data)

        loss = mse_loss(xreco)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{n_epochs}], Iteration [{i+1}/{n_itera}], Loss: {loss.item():.4e}')

        lossesssss.append(loss.item())

        if i == (n_itera - 1):
            vis_im = [img_itSENSE.rss(), img_us_direct.rss(), img_us_itSENSE.rss(), xreco.detach().cpu().abs()[0, ...]]
            vis_title = ['Iterative SENSE', 'Direct R=20', 'Iterative SENSE R = 20', 'Var-Net R=20']
            fig, ax = plt.subplots(2, 4, squeeze=False, figsize=(16, 8))
            for ind in range(4):
                ax[0, ind].imshow(vis_im[ind][0, 0, ...])
                ax[0, ind].set_title(vis_title[ind])
                ax[1, ind].imshow(
                    torch.abs(vis_im[ind][0, 0, ...] - vis_im[0][0, 0, ...]), vmin=0, vmax=vis_im[0].max()*0.1
                )

            list_xreco.append(xreco)

plt.figure()
plt.plot(lossesssss)

#
# %%
