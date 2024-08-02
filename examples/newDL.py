# %% Import and data

import mrpro
from mrpro.data import SpatialDimension
import numpy as np
import torch
import torch.nn.functional as F
from mrpro.algorithms.reconstruction.DirectReconstruction import DirectReconstruction
from mrpro.data._kdata.KData import KData  # Import the KData class
from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.traj_calculators.KTrajectoryPulseq import KTrajectoryPulseq
from mrpro.operators.FourierOp import FourierOp
from mrpro.phantoms.EllipsePhantom import EllipsePhantom  # Adjust the import path as needed
from mrpro.phantoms.phantom_elements import EllipseParameters

# Base path for data files
base_path = '/home/bouill01/data/20240319_spiral_2D_256mm_220k0_128interleaves_golden_angle_vds/'

# Use the trajectory that is stored in the ISMRMRD file
# Load data from Pulseq file using KTrajectoryPulseq
h5_path = base_path + 'pulseq_spiral_2D_220k0_128interleaves_golden_angle_vds_with_traj.h5'
seq_path = base_path + '20240319_spiral_2D_256mm_220k0_128interleaves_golden_angle_vds.seq'

kdatapuls = KData.from_file(h5_path, KTrajectoryPulseq(seq_path=seq_path))


# Local path
h5_path_cartesian = '/data/bouill01/99/20240801_cartesian_2D_256mm_256Nx_256Ny_15alpha/meas_MID00107_FID07003_20240801_cartesian_2D_256mm_256Nx_256Ny_15alpha.h5'
seq_path_cartesian = '/data/bouill01/99/20240801_cartesian_2D_256mm_256Nx_256Ny_15alpha/20240801_cartesian_2D_256mm_256Nx_256Ny_15alpha.seq'

kdata_cart = KData.from_file(h5_path_cartesian, KTrajectoryPulseq(seq_path=seq_path_cartesian))
kdata_cart.header.encoding_matrix = SpatialDimension(z=1, y=256, x=256)
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

shifted_kdatapuls.header.recon_matrix.x = 256
shifted_kdatapuls.header.recon_matrix.y = 256

import torch.nn as nn
#%%
class ConvBlock(nn.Module):
    """
    A simple block of convolutional layers (1D, 2D or 3D)
    """

    def __init__(
        self,
        n_ch_in,
        n_ch_out,
        n_convs,
        kernel_size=3,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()

        padding = int(np.floor(kernel_size / 2))

        conv_block_list = []
        conv_block_list.extend(
            [
                nn.Conv2d(
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
                    nn.Conv2d(
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
    def __init__(
        self,
        n_ch_in,
        n_enc_stages,
        n_convs_per_stage,
        n_filters,
        kernel_size=3,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()

        n_ch_list = [n_ch_in]
        for ne in range(n_enc_stages):
            n_ch_list.append(int(n_filters) * 2**ne)

        self.enc_blocks = nn.ModuleList(
            [
                ConvBlock(
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

        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        features = []
        for block in self.enc_blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features


class Decoder(nn.Module):
    def __init__(
        self,
        n_ch_in,
        n_dec_stages,
        n_convs_per_stage,
        kernel_size=3,
        bias=False,
        padding_mode="zeros",
    ):
        super().__init__()

        n_ch_list = []
        for ne in range(n_dec_stages):
            n_ch_list.append(int(n_ch_in * (1 / 2) ** ne))
            
        self.interp_mode = "bilinear"

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
                ConvBlock(
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

    def forward(self, x, encoder_features):
        for i in range(len(self.dec_blocks)):
            enc_features = encoder_features[i]
            enc_features_shape = enc_features.shape
            x = nn.functional.interpolate(
                x, enc_features_shape[2:], mode=self.interp_mode, align_corners=False
            )
            x = self.upconvs[i](x)
            x = torch.cat([x, enc_features], dim=1)
            x = self.dec_blocks[i](x)
        return x


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
        padding_mode="zeros",
    ):
        super().__init__()
        self.encoder = Encoder(
            n_ch_in,
            n_enc_stages,
            n_convs_per_stage,
            n_filters,
            kernel_size=kernel_size,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.decoder = Decoder(
            n_filters * (2 ** (n_enc_stages - 1)),
            n_enc_stages,
            n_convs_per_stage,
            kernel_size=kernel_size,
            bias=bias,
            padding_mode=padding_mode,
        )

        self.c1x1 = nn.Conv2d(n_filters, n_ch_out, kernel_size=1, padding=0, bias=bias)

        self.residual_connection = residual_connection
        if residual_connection:
            if n_ch_in != n_ch_out:
                raise ValueError(
                    "For using the residual connection, the number of input and output channels of the \
                    network must be the same.\
                    Given {n_ch_in} and {n_ch_out}."
                )

    def forward(self, x):
        enc_features = self.encoder(x)
        dec = self.decoder(enc_features[-1], enc_features[::-1][1:])
        out = self.c1x1(dec)
        if self.residual_connection:
            out = out + x
        return out
# %% NufftCascade
from einops import rearrange

class NUFFTCascade(nn.Module):
    def __init__(
        self,
        acquisition_operator: FourierOp,
        npcg: int,
        w: float,
        max_iter: int = 10,
        initial_value: torch.Tensor = None,
    ) -> None:
        super(NUFFTCascade, self).__init__()
        self.acquisition_operator = acquisition_operator
        self.npcg = npcg
        self.w = w
        self.w_raw = nn.Parameter(torch.tensor(-5.0, requires_grad=True))
        self.max_iter = max_iter
        self.initial_value = initial_value
        self.unet = UNet(n_ch_in=2, n_ch_out=2, n_enc_stages=3, n_convs_per_stage=2, n_filters=16)

    
    # @property
    def w_reg(self, initial_value):
        # Ensure acquisition_operator is provided
        if self.acquisition_operator is not None:
            op_norm_estimate = self.acquisition_operator.operator_norm(
                initial_value=initial_value.unsqueeze(0).unsqueeze(0).unsqueeze(0), max_iterations=self.max_iter, dim=None
            )
        else:
            # Provide a default value or handle the case where acquisition_operator is None
            op_norm_estimate = torch.tensor([[[1.0000]]])

        return (2 * F.sigmoid(self.w_raw)) / op_norm_estimate**2

    def apply_CNN(self, x):
        x = torch.view_as_real(x)
        x = rearrange(x, 'x y ch -> ch x y')
        layers1 = nn.Conv2d(
            in_channels=2,
            out_channels=32,
            kernel_size=3,
            padding=1,  ## kernel_size /2
            stride=1,
        )
        x = layers1(x)
        
        layers2 = nn.Conv2d(
            in_channels=32,
            out_channels=2,
            kernel_size=3,
            padding=1,  ## kernel_size /2
            stride=1,
        )
        x = layers2(x)
        
        
        x = rearrange(x, 'ch x y-> x y ch')
        x = torch.view_as_complex(x.contiguous())
        return x  # torch.Size([220, 220])

    def apply_UNet(self, x):
        x = torch.view_as_real(x).unsqueeze(0)
        x = rearrange(x, 'b x y ch -> b ch x y')
        x = self.unet(x)
        x = rearrange(x, 'b ch x y -> b x y ch')
        x = torch.view_as_complex(x.contiguous()).squeeze(0)
        return x

    def forward(self, x, k_space_data):
        for _ in range(self.npcg):
            res_acq_error = self.acquisition_operator.H(self.acquisition_operator(x.unsqueeze(0).unsqueeze(0).unsqueeze(0))[0] - (k_space_data.data))
            #xnn = self.apply_CNN(x)
            #xunet = self.apply_UNet(x)
            x = x - (self.w_reg(x) * res_acq_error[0]).squeeze(0).squeeze(0).squeeze(0) # - xunet # - xnn

        return x


# %% Fonctiun creating Ellipses


def generate_random_ellipses(num_ellipses):
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
    """Creates multiple k-space data sets from multiple ellipsoidal phantoms.

    Parameters
    ----------
    num_samples : int
        The number of k-space datasets to generate.
    num_ellipses_per_sample : int
        The number of ellipses in each phantom.
    kx : torch.Tensor
        The k-space positions in the kx direction.
    ky : torch.Tensor
        k-space positions in ky direction.

    Returns
    -------
    List[torch.Tensor]
        A list of tensors containing phantoms.
    """
    phantom_list = []
    for _ in range(num_samples):
        ellipses = generate_random_ellipses(num_ellipses_per_sample)
        phantom = EllipsePhantom(ellipses)
        phantom_list.append(phantom)
    return phantom_list


# %% Initialisation
# Define the parameters for data generation
num_samples = 3  # Number of k-space data samples to generate
num_ellipses_per_sample = 8  # Number of ellipses in each phantom

# us_idx
us_idx = torch.arange(0, 70, 1)[None:]
# Define k-space grid
nx, ny = 220, 220
kx = shifted_kdatapuls.traj.kx
ky = shifted_kdatapuls.traj.ky


# %% Creating Kspace_data (Not needed because doing it "on the fly")
# Define image dimensions
# image_dimensions = SpatialDimension(z=1, y=nx, x=ny)

# # Create phantoms
# phantom_samples = create_rand_phantom(num_samples, num_ellipses_per_sample, kx, ky)

# kspace_data_samples = phantom_samples[0].kspace(ky, kx)
# reconstructed_images = []
# kspace_samples = []

# for i, phantom in enumerate(phantom_samples):
#     # Generate k-space data
#     kspace_data = phantom.kspace(ky, kx)
#     print(f'K-space data sample {i+1}: shape = {kspace_data.shape}')

#     kdata_object = KData(data=kspace_data.unsqueeze(0), header=shifted_kdatapuls.header, traj=shifted_kdatapuls.traj)

#     kdata_object.header.recon_matrix.x = nx
#     kdata_object.header.recon_matrix.y = ny

#     kdata_us = KData.split_k1_into_other(kdata_object, torch.arange(0, 128, 4)[None, :], other_label='repetition')

#     direct_reconstruction_fullsamp = DirectReconstruction.from_kdata(kdata_object)
#     img_fullsamp = direct_reconstruction_fullsamp(kdata_object)
#     reconstructed_img_fullsamp = img_fullsamp.data

#     direct_reconstruction_us = DirectReconstruction.from_kdata(kdata_us)
#     img_us = direct_reconstruction_us(kdata_us)
#     reconstructed_img_us = img_us.data

#     if i == 0:
#         reconstructed_images_fullsamp=reconstructed_img_fullsamp
#         reconstructed_images_us=reconstructed_img_us
#         kspace_data_fullsamp=kdata_object.data
#         kspace_data_us=kdata_us.data
#     else :
#         reconstructed_images_fullsamp=torch.cat((reconstructed_images_fullsamp, reconstructed_img_fullsamp),2)
#         reconstructed_images_us=torch.cat((reconstructed_images_us, reconstructed_img_us),2)
#         kspace_data_fullsamp=torch.cat((kspace_data_fullsamp, kdata_object.data),2)
#         kspace_data_us=torch.cat((kspace_data_us, kdata_us.data),2)

# kdata_simulated_fullsamp = KData(
#     data = kspace_data_fullsamp,
#     header=shifted_kdatapuls.header,
#     traj=shifted_kdatapuls.traj
# )

# kdata_simulated_us = KData(
#     data = kspace_data_us,
#     header=kdata_us.header,
#     traj=kdata_us.traj
# )

# %% Plot the k-space trajectory for kdata_us for each num_sample
# for i in range (num_samples):
#     # Plot the k-space trajectory for fully sampled kdata
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.plot(kdata_object.traj.ky.flatten(), kdata_object.traj.kx.flatten(), 'o', label='k-space trajectory')
#     plt.title(f'Fully Sampled k-space Trajectory - Sample {i+1}')
#     plt.xlabel('ky')
#     plt.ylabel('kx')
#     plt.legend()
#     plt.grid(True)

#     # Plot the fully sampled reconstructed image
#     plt.subplot(1, 2, 2)
#     plt.imshow(torch.abs(reconstructed_images_fullsamp[0, 0, i, :, :]), cmap='gray')
#     plt.title(f'Fully Sampled Reconstructed Image - Sample {i+1}')
#     plt.colorbar()
#     plt.xlabel('Y')
#     plt.ylabel('X')

#     # Show the plot for fully sampled data
#     plt.tight_layout()
#     plt.show()

#     # Plot the k-space trajectory for undersampled kdata
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.plot(kdata_us.traj.ky.flatten(), kdata_us.traj.kx.flatten(), 'o', label='k-space trajectory')
#     plt.title(f'Undersampled k-space Trajectory - Sample {i+1}')
#     plt.xlabel('ky')
#     plt.ylabel('kx')
#     plt.legend()
#     plt.grid(True)

#     # Plot the undersampled reconstructed image
#     plt.subplot(1, 2, 2)
#     plt.imshow(torch.abs(reconstructed_images_us[0, 0, i, :, :]), cmap='gray')
#     plt.title(f'Undersampled Reconstructed Image - Sample {i+1}')
#     plt.colorbar()
#     plt.xlabel('Y')
#     plt.ylabel('X')

#     # Show the plot for undersampled data
#     plt.tight_layout()
#     plt.show()

# %% Custom Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


class CustomDataset(Dataset):
    def __init__(self, num_ellipses_per_sample):
        self.num_ellipses_per_sample = num_ellipses_per_sample

    def __len__(self):
        return 1

    def generate_random_ellipses(self):
        """Generate random ellipses based on self.num_ellipses_per_sample."""
        ellipses = []
        for _ in range(self.num_ellipses_per_sample):
            # Generate radius first to use it for constraining center coordinates
            radius_x = np.random.uniform(0.05, 0.4)
            radius_y = np.random.uniform(0.05, 0.4)
            
            # Calculate bounds for the center coordinates to keep the ellipse within [-0.5, 0.5] x [-0.5, 0.5]
            min_center_x = -0.4 + radius_x 
            max_center_x = 0.4 - radius_x
            min_center_y = -0.4 + radius_y
            max_center_y = 0.4 - radius_y
            
            center_x = np.random.uniform(min_center_x, max_center_x)
            center_y = np.random.uniform(min_center_y, max_center_y)
            
            intensity = np.random.uniform(1, 50)
            ellipses.append(EllipseParameters(center_x, center_y, radius_x, radius_y, intensity))
        
        return ellipses


    def __getitem__(self, index):
        # Generate ellipses for this sample
        ellipses = self.generate_random_ellipses()
        phantom = EllipsePhantom(ellipses)

        kspace_data = phantom.kspace(shifted_kdatapuls.traj.ky, shifted_kdatapuls.traj.kx)
        print(f'K-space data sample {index + 1}: shape = {kspace_data.shape}')
        

        kdata_object = KData(
            data=kspace_data.unsqueeze(0), header=shifted_kdatapuls.header, traj=shifted_kdatapuls.traj
        )

        kdata_us = KData.split_k1_into_other(kdata_object, torch.arange(0, 128, 10)[None, :], other_label='repetition')

        direct_reconstruction_fullsamp = DirectReconstruction.from_kdata(kdata_object)
        x_fullsamp = direct_reconstruction_fullsamp(kdata_object)

        direct_reconstruction_us = DirectReconstruction.from_kdata(kdata_us)
        x_us = direct_reconstruction_us(kdata_us)

        # xf_real = x_fullsamp.data.real
        # xf_imag = x_fullsamp.data.imag
        # xf = torch.stack((xf_real, xf_imag), dim=5)
        xf = x_fullsamp.data.squeeze(0).squeeze(0).squeeze(0)

        # xu_real = x_us.data.real
        # xu_imag = x_us.data.imag
        # xu = torch.stack((xu_real, xu_imag), dim=5)
        xu = x_us.data.squeeze(0).squeeze(0).squeeze(0)

        return xu, xf


# %% Initialize hyperparameters
n_epochs = 10
n_itera = 1
learning_rate = 1e-3
batch_size = 1
lossesssss = []
tot_loss = 0
list_xreco = []
list_xu = []
list_xf = []
# Initialize the necessary objects

model = NUFFTCascade(acquisition_operator=None, npcg=16, w=0.1, initial_value=None, max_iter=10)
loss_fct = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#
dataset = CustomDataset(num_ellipses_per_sample=8)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
#%% Function to visualize results
def visualize_results(epoch, iteration, xu, xf, xreco):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(torch.abs(xu).cpu().numpy(), cmap='gray')
    plt.title('Input Undersampled Image')
    plt.subplot(1, 3, 2)
    plt.imshow(torch.abs(xf).cpu().numpy(), cmap='gray')
    plt.title('Full-sampled Image')
    plt.subplot(1, 3, 3)
    plt.imshow(torch.abs(xreco).cpu().detach().numpy(), cmap='gray')
    plt.title('Reconstructed Image')
    plt.suptitle(f'Epoch {epoch+1}, Iteration {iteration+1}')
    plt.show()
    
    
def i2k(im):
    if len(im.shape) == 1: # Carry out 1D FFT: image space -> k-space
        kdat = (np.fft.fftshift(np.fft.fftn(np.fft.fftshift(im,(0)),
                        (im.shape[0],),(0,), norm=None), (0,)))
    else: # Carry out 2D FFT: image space -> k-space
        kdat = (np.fft.fftshift(np.fft.fftn(np.fft.fftshift(im,(0,1)),
                        (im.shape[0], im.shape[1]),(0,1), norm=None), (0,1)))
    
    return(kdat)
# %%
# Model training
for epoch in range(n_epochs):
    for i in range (n_itera):
        data = next(iter(data_loader))
        xu, xf = data

        # Remove the batch dimension 
        xu = xu.squeeze(0)
        xf = xf.squeeze(0)
        
        ku = torch.tensor(i2k(xu))

        
        kdata_us = KData(
            data=ku.unsqueeze(0).unsqueeze(0).unsqueeze(0), traj=kdata_cart.traj , header=kdata_cart.header
            )
        
        # Update acquisition operator and norm estimate based on current batch's undersampled data
       # Calculate the square root of the DCF data
        encoding_matrix = kdata_us.header.encoding_matrix
        traj = kdata_us.traj
        cartesian_sampling_operator = mrpro.operators.CartesianSamplingOp(encoding_matrix, traj)

        # Define the Fourier operator using the trajectory and header information in kdata
        # Here we assume the k-space data is on a Cartesian grid
        # fourier_operator = mrpro.operators.FourierOp.from_kdata(kdata_us)

        # Combine the Cartesian sampling and Fourier operators
        # sampling_fourier_operator = fourier_operator @ cartesian_sampling_operator
        # acquisition_operator = sampling_fourier_operator.H
        acquisition_operator = mrpro.operators.FourierOp.from_kdata(kdata_us)
        op_norm_estimate = acquisition_operator.operator_norm(initial_value= xu.unsqueeze(0).unsqueeze(0).unsqueeze(0), max_iterations=4, dim=None)

        # Set the updated acquisition operator and norm estimate to the model
        model.acquisition_operator = acquisition_operator

        optimizer.zero_grad()
        xreco = model(xu, kdata_us)

        loss = loss_fct(torch.view_as_real(xreco), torch.view_as_real(xf)) 
        
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{n_epochs}], Iteration [{i+1}/{n_itera}], Loss: {loss.item():.4f}')

        tot_loss += loss.item()
            
        if i == (n_itera-1):
            visualize_results(epoch, i, xu, xf, xreco)
            lossesssss.append(tot_loss/n_itera)
            tot_loss = 0
            list_xreco.append(xreco)
            list_xu.append(xu)
            list_xf.append(xf)

print('Training finished.')
print(f'losses = {lossesssss}')

# %%

import pickle

# Save the lists to files
with open('reconstructed_images.pkl', 'wb') as f:
    pickle.dump(list_xreco, f)

with open('undersampled_images.pkl', 'wb') as f:
    pickle.dump(list_xu, f)

with open('full_sampled_images.pkl', 'wb') as f:
    pickle.dump(list_xf, f)

print('Lists have been saved successfully.')
