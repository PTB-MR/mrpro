# %% Import and data

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import mrpro
from mrpro.operators.FourierOp import FourierOp
from mrpro.algorithms.reconstruction.DirectReconstruction import DirectReconstruction
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.data._kdata.KData import KData  # Import the KData class
from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.traj_calculators.KTrajectoryPulseq import KTrajectoryPulseq
from mrpro.phantoms.EllipsePhantom import EllipsePhantom  # Adjust the import path as needed
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

shifted_kdatapuls.header.recon_matrix.x = 220
shifted_kdatapuls.header.recon_matrix.y = 220

import torch.nn as nn


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
        
         # Ensure acquisition_operator is provided
        if acquisition_operator is not None:
            self.op_norm_estimate = acquisition_operator.operator_norm(
                initial_value=initial_value, 
                max_iterations=max_iter, 
                dim=(-3, -2, -1)
            )
        else:
            # Provide a default value or handle the case where acquisition_operator is None
            self.op_norm_estimate = torch.tensor([[[1.0000]]])
    @property
    def w_reg(self):
        return (2 * F.sigmoid(self.w_raw)) / self.op_norm_estimate**2
        
    def apply_CNN(self, x):
        x = torch.view_as_real(x)
        x = rearrange(x, "one x y ch -> one ch x y")
        layers = (nn.Conv2d( 
            in_channels=2,
            out_channels=2,
            kernel_size=3,
            padding=1, ## kernel_size /2
            stride=1))
        x = layers(x)
        x = rearrange(x, "one ch x y-> one x y ch")
        x = torch.view_as_complex(x.contiguous())
        # return layers
        return x #nn.Sequential(layers)

    def forward(self, x, k_space_data):
        for _ in range(self.npcg):
            res_acq_error = self.acquisition_operator.H(self.acquisition_operator(x)[0] - k_space_data.data)
            xnn = self.apply_CNN(x)
            x = x - self.w_reg * res_acq_error[0] - xnn

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
            center_x = np.random.uniform(-0.5, 0.5)
            center_y = np.random.uniform(-0.5, 0.5)
            radius_x = np.random.uniform(0.05, 0.3)
            radius_y = np.random.uniform(0.05, 0.3)
            intensity = np.random.uniform(1, 50)
            ellipses.append(EllipseParameters(center_x, center_y, radius_x, radius_y, intensity))
        return ellipses

    def __getitem__(self, index):
        # Generate ellipses for this sample
        ellipses = self.generate_random_ellipses()
        phantom = EllipsePhantom(ellipses)


        kspace_data = phantom.kspace(shifted_kdatapuls.traj.ky, shifted_kdatapuls.traj.kx)
        print(f'K-space data sample {index + 1}: shape = {kspace_data.shape}')

        kdata_object = KData(data=kspace_data.unsqueeze(0), header=shifted_kdatapuls.header, traj=shifted_kdatapuls.traj)

        kdata_us = KData.split_k1_into_other(kdata_object, torch.arange(0, 128, 4)[None, :], other_label='repetition')

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
        
        kf = kspace_data.squeeze(0)
        
        return xu, kf, xf

# %%
#  Initialize hyperparameters
n_epochs = 10
learning_rate = 1e-4
batch_size = 1


# Initialize the necessary objects

model = NUFFTCascade(acquisition_operator=None, npcg=16, w=0.1, initial_value=None, max_iter=10)
loss_fct = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#
dataset = CustomDataset(num_ellipses_per_sample=8)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

#%%
# Model training
for epoch in range(n_epochs):
    for _, data in enumerate(data_loader):
        xu, kf, xf = data

        kdata_object = KData(data=kf, header=shifted_kdatapuls.header, traj=shifted_kdatapuls.traj)

        kdata_us = KData.split_k1_into_other(kdata_object, torch.arange(0, 127, 4)[None, :], other_label='repetition')
        
        # Update acquisition operator and norm estimate based on current batch's undersampled data
        # Calculate dcf using the trajectory
        dcf_operator = mrpro.data.DcfData.from_traj_voronoi(kdata_us.traj).as_operator()

        # Define Fourier operator using the trajectory
        # and header information in kdata
        acquisition_operator = dcf_operator @ FourierOp.from_kdata(kdata_us)
        op_norm_estimate = acquisition_operator.operator_norm(initial_value=xu, max_iterations=10, dim=(-3, -2, -1))
        
  

        # Set the updated acquisition operator and norm estimate to the model
        model.acquisition_operator = adjoint_operator
        model.op_norm_estimate = op_norm_estimate
        
        optimizer.zero_grad()
        xreco = model(xu, kdata_us)
        
        loss = loss_fct(xreco, xf)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

print('Training finished.')


# %%
