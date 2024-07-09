# %% Import and data

import matplotlib.pyplot as plt
import mrpro
import numpy as np
import torch
from mrpro.algorithms.reconstruction import DirectReconstruction
from mrpro.data import KData  # Import the KData class
from mrpro.data import KTrajectory
from mrpro.data import SpatialDimension
from mrpro.data.traj_calculators import KTrajectoryPulseq
from mrpro.phantoms import EllipsePhantom  # Adjust the import path as needed
from mrpro.phantoms.phantom_elements import EllipseParameters

# Base path for data files
base_path = '/home/bouill01/data/20240319_spiral_2D_256mm_220k0_128interleaves_golden_angle_vds/'

# Use the trajectory that is stored in the ISMRMRD file
# Load data from Pulseq file using KTrajectoryPulseq
h5_path = base_path + 'pulseq_spiral_2D_220k0_128interleaves_golden_angle_vds_with_traj.h5'
seq_path = base_path + '20240319_spiral_2D_256mm_220k0_128interleaves_golden_angle_vds.seq'

kdatapuls = KData.from_file(h5_path, KTrajectoryPulseq(seq_path=seq_path))
# %% shifting k-space


def shift_k_space_trajectory(kdatapuls: KData) -> KData:
    """
    Shift the k-space trajectory of the input KData.

    Args:
        kdatapuls (KData): The input k-space data with trajectory.

    Returns
    -------
        KData: The shifted k-space data with updated trajectory.
    """
    # Extract k-space trajectory from kdatapuls
    ky_pulseq = kdatapuls.traj.ky
    kx_pulseq = kdatapuls.traj.kx
    kz_pulseq = kdatapuls.traj.kz

    # Number of indices
    num_indices = ky_pulseq.shape[2]

    # Initialize lists to store shifted trajectories
    shifted_ky = ky_pulseq.clone()
    shifted_kx = kx_pulseq.clone()

    # Loop to apply the shift to each index
    for i in range(num_indices - 1):
        # Calculate the shift for the current index
        shifted_ky[:, :, i, :] -= ky_pulseq[:, :, i, 0]
        shifted_kx[:, :, i, :] -= kx_pulseq[:, :, i, 0]

    # Create shifted KTrajectory object
    shifted_traj = KTrajectory(kx=shifted_kx, ky=shifted_ky, kz=kz_pulseq)
    # Create shifted KData object
    shifted_kdatapuls = KData(data=kdatapuls.data, traj=shifted_traj, header=kdatapuls.header)

    return shifted_kdatapuls


shifted_kdatapuls = shift_k_space_trajectory(kdatapuls)
# %% Unet
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    A simple block of convolutional layers (1D, 2D or 3D)
    """

    def __init__(
        self,
        dim: int,
        n_ch_in: int,
        n_ch_out: int,
        n_convs: int,
        kernel_size: int = 3,
        bias: bool = True,
        padding_mode: str = 'zeros',
    ) -> None:
        super().__init__()

        if dim == 1:
            conv_op = nn.Conv1d
        elif dim == 2:
            conv_op = nn.Conv2d
        elif dim == 3:
            conv_op = nn.Conv3d

        padding = int(np.floor(kernel_size / 2))

        conv_block_list = []
        conv_block_list.extend(
            [
                conv_op(
                    n_ch_in,
                    n_ch_out,
                    kernel_size,
                    padding=padding,
                    bias=bias,
                    padding_mode=padding_mode,
                ),
                nn.LeakyReLU(),
            ]
        )

        for i in range(n_convs - 1):
            conv_block_list.extend(
                [
                    conv_op(
                        n_ch_out,
                        n_ch_out,
                        kernel_size,
                        padding=padding,
                        bias=bias,
                        padding_mode=padding_mode,
                    ),
                    nn.LeakyReLU(),
                ]
            )

        self.conv_block = nn.Sequential(*conv_block_list)

    def forward(self, x):
        return self.conv_block(x)


class Encoder(nn.Module):
    """
    Encoder part of the UNet model.
    """

    def __init__(
        self,
        dim: int,
        n_ch_in: int,
        n_enc_stages: int,
        n_convs_per_stage: int,
        n_filters: int,
        kernel_size: int = 3,
        bias: bool = True,
        padding_mode='zeros',
    ):
        super().__init__()

        n_ch_list = [n_ch_in]
        for ne in range(n_enc_stages):
            n_ch_list.append(int(n_filters) * 2**ne)

        self.enc_blocks = nn.ModuleList(
            [
                ConvBlock(
                    dim,
                    n_ch_list[i],
                    n_ch_list[i + 1],
                    n_convs_per_stage,
                    kernel_size=kernel_size,
                    bias=bias,
                    padding_mode=padding_mode,
                )
                for i in range(len(n_ch_list) - 1)
            ]
        )

        if dim == 1:
            pool_op = nn.MaxPool1d(2)
        elif dim == 2:
            pool_op = nn.MaxPool2d(2)
        elif dim == 3:
            pool_op = nn.MaxPool3d(2)

        self.pool = pool_op

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for block in self.enc_blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features


class Decoder(nn.Module):
    """
    The decoder part of the UNet.
    """

    def __init__(
        self,
        dim: int,
        n_ch_in: int,
        n_dec_stages: int,
        n_convs_per_stage: int,
        kernel_size: int = 3,
        bias: bool = False,
        padding_mode: str = 'zeros',
    ) -> None:
        super().__init__()

        n_ch_list = []
        for ne in range(n_dec_stages):
            n_ch_list.append(int(n_ch_in * (1 / 2) ** ne))

        if dim == 1:
            conv_op = nn.Conv1d
            interp_mode = 'linear'
        elif dim == 2:
            conv_op = nn.Conv2d
            interp_mode = 'bilinear'
        elif dim == 3:
            interp_mode = 'trilinear'
            conv_op = nn.Conv3d

        self.interp_mode = interp_mode

        padding = int(np.floor(kernel_size / 2))
        self.upconvs = nn.ModuleList(
            [
                conv_op(
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
                ConvBlock(
                    dim,
                    n_ch_list[i],
                    n_ch_list[i + 1],
                    n_convs_per_stage,
                    kernel_size=kernel_size,
                    bias=bias,
                    padding_mode=padding_mode,
                )
                for i in range(len(n_ch_list) - 1)
            ]
        )

    def forward(self, x: torch.Tensor, encoder_features: list[torch.Tensor]) -> torch.Tensor:
        for i in range(len(self.dec_blocks)):
            enc_features = encoder_features[i]
            enc_features_shape = enc_features.shape
            x = nn.functional.interpolate(x, enc_features_shape[2:], mode=self.interp_mode, align_corners=False)
            x = self.upconvs[i](x)
            x = torch.cat([x, enc_features], dim=1)
            x = self.dec_blocks[i](x)
        return x


class UNet(nn.Module):
    """
    UNet model for image reconstruction.

    Args:
        dim (int): Dimension of the input data (1, 2, or 3).
        n_ch_in (int): Number of input channels.
        n_ch_out (int): Number of output channels.
        n_filters (int): Base number of filters for the convolutional layers.
        n_enc_stages (int): Number of encoding stages.
        n_convs_per_stage (int): Number of convolutional layers per stage.
        kernel_size (int, optional): Kernel size for the convolutional layers. Defaults to 3.
        bias (bool, optional): Whether to use bias in the convolutional layers. Defaults to False.
        padding_mode (str, optional): Padding mode for the convolutional layers. Defaults to 'zeros'.
    """

    def __init__(
        self,
        dim: int,
        n_ch_in: int = 2,
        n_ch_out: int = 2,
        n_filters: int = 64,
        n_enc_stages: int = 4,
        n_convs_per_stage: int = 2,
        kernel_size: int = 3,
        bias: bool = True,
        residual_connection: bool = True,
        padding_mode: str = 'zeros',
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            dim,
            n_ch_in,
            n_enc_stages,
            n_convs_per_stage,
            n_filters,
            kernel_size=kernel_size,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.decoder = Decoder(
            dim,
            n_filters * (2 ** (n_enc_stages - 1)),
            n_enc_stages,
            n_convs_per_stage,
            kernel_size=kernel_size,
            bias=bias,
            padding_mode=padding_mode,
        )

        if dim == 1:
            conv_op = nn.Conv1d
        elif dim == 2:
            conv_op = nn.Conv2d
        elif dim == 3:
            conv_op = nn.Conv3d

        self.c1x1 = conv_op(n_filters, n_ch_out, kernel_size=1, padding=0, bias=bias)

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


# %% NufftCascade


class NUFFTCascade(nn.Module):
    def __init__(self, acquisition_operator, unet, nu, npcg, w, op_norm_estimate=None):
        super(NUFFTCascade, self).__init__()
        self.acquisition_operator = acquisition_operator
        self.unet = unet
        self.nu = nu
        self.npcg = npcg
        self.w = w
        self.w_raw = nn.Parameter(torch.tensor(-5.0, requires_grad=True))
        self.op_norm_stimate = acquisition_operator.operator_norm(initial_value, maxiterations=...)

    @property
    def w_reg(self):
        return (F.sigmoid(self.w_raw) + 1) / self.op_norm**2

    def forward(self, x, k_space_data):
        for _ in range(self.npcg):
            operator_test = self.acquisition_operator.H(self.acquisition_operator(x) - k_space_data)
            x = x - self.w_reg * operator_test - self.unet(x)

        return x


# %% Fonctiun creating Ellipses


def generate_random_ellipses(num_ellipses):
    """Génère une liste de paramètres d'ellipses avec des valeurs aléatoires."""
    ellipses = []
    for _ in range(num_ellipses):
        center_x = np.random.uniform(-0.5, 0.5)
        center_y = np.random.uniform(-0.5, 0.5)
        radius_x = np.random.uniform(0.05, 0.3)
        radius_y = np.random.uniform(0.05, 0.3)
        intensity = np.random.uniform(1, 50)
        ellipses.append(EllipseParameters(center_x, center_y, radius_x, radius_y, intensity))
    return ellipses


def create_rand_phantom(num_samples, num_ellipses_per_sample, kx, ky):
    """Crée plusieurs ensembles de données k-space à partir de plusieurs fantômes ellipsoïdes.

    Parameters
    ----------
    num_samples : int
        Le nombre d'ensembles de données k-space à générer.
    num_ellipses_per_sample : int
        Le nombre d'ellipses dans chaque fantôme.
    kx : torch.Tensor
        Les positions k-space dans la direction kx.
    ky : torch.Tensor
        Les positions k-space dans la direction ky.

    Returns
    -------
    List[torch.Tensor]
        Une liste de tenseurs contenant les phantoms.
    """
    phantom_list = []
    for _ in range(num_samples):
        ellipses = generate_random_ellipses(num_ellipses_per_sample)
        phantom = EllipsePhantom(ellipses)
        phantom_list.append(phantom)
    return phantom_list


# %% Initialisation
# Define the parameters for data generation
num_samples = 4  # Number of k-space data samples to generate
num_ellipses_per_sample = 8  # Number of ellipses in each phantom

# us_idx
us_idx = torch.arange(0, 70, 1)[None:]
# Define k-space grid
nx, ny = 220, 220
kx = shifted_kdatapuls.traj.kx
ky = shifted_kdatapuls.traj.ky
# Initialize lists to store data
data_list = []

# %% Creating Kspace_data
# Define image dimensions
image_dimensions = SpatialDimension(z=1, y=nx, x=ny)

# Create phantoms
phantom_samples = create_rand_phantom(num_samples, num_ellipses_per_sample, kx, ky)

kspace_data_samples = []
reconstructed_images = []
kspace_samples = []

for i, phantom in enumerate(phantom_samples):
    # Generate k-space data
    kspace_data = phantom.kspace(ky, kx)
    print(f'K-space data sample {i+1}: shape = {kspace_data.shape}')
    kspace_data_samples.append(kspace_data)

    kdata_object = KData(data=kspace_data.unsqueeze(0), header=shifted_kdatapuls.header, traj=shifted_kdatapuls.traj)

    kdata_object.header.recon_matrix.x = nx
    kdata_object.header.recon_matrix.y = ny

    kdata_us = KData.split_k1_into_other(kdata_object, torch.arange(0, 128, 4)[None, :], other_label='repetition')

    direct_reconstruction_fullsamp = DirectReconstruction.from_kdata(kdata_object)
    img_fullsamp = direct_reconstruction_fullsamp(kdata_object)
    reconstructed_img_fullsamp = img_fullsamp.data

    direct_reconstruction_us = DirectReconstruction.from_kdata(kdata_us)
    img_us = direct_reconstruction_us(kdata_us)
    reconstructed_img_us = img_us.data

    # Store reconstructed image
    reconstructed_images.append((reconstructed_img_fullsamp, reconstructed_img_us))
    kspace_samples.append((kdata_object, kdata_us))
    # Define Fourier operator using the trajectory and header information in kdata
    acquisition_operator = mrpro.operators.FourierOp.from_kdata(kdata_us)

    # Unpack the tuple and perform the subtraction
    kdata_simulated_us = acquisition_operator.H(kdata_us.data)[0]
    
    # Create a KData object with the simulated k-space data
    kdata_simulated_us = mrpro.data.KData(data=kdata_simulated_us, traj=kdata_us.traj, header=kdata_us.header)


data_list.append((kdata_object, kdata_simulated_us))
# %% Plot the k-space trajectory for kdata_us for each num_sample
for i, (reconstructed_img_fullsamp, reconstructed_img_us) in enumerate(reconstructed_images):
    # Plot the k-space trajectory for fully sampled kdata
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(kdata_object.traj.ky.flatten(), kdata_object.traj.kx.flatten(), 'o', label='k-space trajectory')
    plt.title(f'Fully Sampled k-space Trajectory - Sample {i+1}')
    plt.xlabel('ky')
    plt.ylabel('kx')
    plt.legend()
    plt.grid(True)

    # Plot the fully sampled reconstructed image
    plt.subplot(1, 2, 2)
    plt.imshow(torch.abs(reconstructed_img_fullsamp[0, 0, 0, :, :]), cmap='gray')
    plt.title(f'Fully Sampled Reconstructed Image - Sample {i+1}')
    plt.colorbar()
    plt.xlabel('Y')
    plt.ylabel('X')

    # Show the plot for fully sampled data
    plt.tight_layout()
    plt.show()

    # Plot the k-space trajectory for undersampled kdata
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(kdata_us.traj.ky.flatten(), kdata_us.traj.kx.flatten(), 'o', label='k-space trajectory')
    plt.title(f'Undersampled k-space Trajectory - Sample {i+1}')
    plt.xlabel('ky')
    plt.ylabel('kx')
    plt.legend()
    plt.grid(True)

    # Plot the undersampled reconstructed image
    plt.subplot(1, 2, 2)
    plt.imshow(torch.abs(reconstructed_img_us[0, 0, 0, :, :]), cmap='gray')
    plt.title(f'Undersampled Reconstructed Image - Sample {i+1}')
    plt.colorbar()
    plt.xlabel('Y')
    plt.ylabel('X')

    # Show the plot for undersampled data
    plt.tight_layout()
    plt.show()

# %% Dataset
from torch.utils.data import Dataset


class YourDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list  # List of (undersampled_kdata, fullysampled_kdata) tuples

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        fullysampled_kdata, undersampled_kdata = self.data_list[idx]

        # Convert data to tensors if not already
        xu = torch.tensor(undersampled_kdata, dtype=torch.float32)
        xf = torch.tensor(fullysampled_kdata, dtype=torch.float32)

        return xu, xf


# %%
#  Initialize hyperparameters
n_epochs = 10
learning_rate = 1e-4
batch_size = 16

# Initialize the necessary objects
## Initialize the necessary objects
# unet = UNet(dim=2, n_ch_in=2, n_ch_out=2, n_enc_stages=3, n_convs_per_stage=2, n_filters=16)
# model = NUFFTCascade(acquisition_operator, unet, nu=1, npcg=16, w=0.1,)
# loss_fct = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# #
# dataset = YourDataset(data_list)
# data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# # Model training
# for epoch in range(n_epochs):
#     for _, data in enumerate(data_loader):
#         xu, xf = data
#         optimizer.zero_grad()
#         xreco = model(xu)
#         loss = loss_fct(xreco, xf)
#         loss.backward()
#         optimizer.step()
#         print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

# print('Training finished.')


# %%
