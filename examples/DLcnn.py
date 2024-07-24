# %% Import and data

import mrpro
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
                initial_value=initial_value, max_iterations=max_iter, dim=(-3, -2, -1)
            )
        else:
            # Provide a default value or handle the case where acquisition_operator is None
            self.op_norm_estimate = torch.tensor([[[1.0000]]])

    @property
    def w_reg(self):
        return (2 * F.sigmoid(self.w_raw)) / self.op_norm_estimate**2

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


    def forward(self, x, k_space_data):
        for _ in range(self.npcg):
            dcf = mrpro.data.DcfData.from_traj_voronoi(kdata_us.traj)
            sqrt_dcf_dat = torch.sqrt(dcf.data)
            res_acq_error = self.acquisition_operator.H(self.acquisition_operator(x.unsqueeze(0).unsqueeze(0).unsqueeze(0))[0] - (k_space_data.data*sqrt_dcf_dat))
            xnn = self.apply_CNN(x)
            x = x - (self.w_reg * res_acq_error[0]).squeeze(0).squeeze(0).squeeze(0) - xnn

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

        kf = kspace_data.unsqueeze(0)

        return xu, kf, xf


# %%
#  Initialize hyperparameters
n_epochs = 10
n_itera = 100
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
#%%
import matplotlib.pyplot as plt
# Function to visualize results
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
# %%
# Model training
for epoch in range(n_epochs):
    for i in range (n_itera):
        data = next(iter(data_loader))
        xu, kf, xf = data

        # Remove the batch dimension 
        xu = xu.squeeze(0)
        kf = kf.squeeze(0)
        xf = xf.squeeze(0)
        
        kdata_object = KData(data=kf, header=shifted_kdatapuls.header, traj=shifted_kdatapuls.traj)

        kdata_us = KData.split_k1_into_other(kdata_object, torch.arange(0, 127, 10)[None, :], other_label='repetition')

        # Update acquisition operator and norm estimate based on current batch's undersampled data
       # Calculate the square root of the DCF data
        dcf_data = mrpro.data.DcfData.from_traj_voronoi(kdata_us.traj)
        sqrt_dcf_data = torch.sqrt(dcf_data.data)

        # Create a new DCF operator with the square-rooted DCF data
        sqrt_dcf_operator = mrpro.data.DcfData(sqrt_dcf_data).as_operator()

        # Define the acquisition operator using the new sqrt_dcf_operator
        acquisition_operator = sqrt_dcf_operator @ FourierOp.from_kdata(kdata_us)
        op_norm_estimate = acquisition_operator.operator_norm(initial_value= xu.unsqueeze(0).unsqueeze(0).unsqueeze(0), max_iterations=10, dim=None)

        # Set the updated acquisition operator and norm estimate to the model
        model.acquisition_operator = acquisition_operator
        model.op_norm_estimate = op_norm_estimate

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
with open('reconstructed_images_cnn.pkl', 'wb') as f:
    pickle.dump(list_xreco, f)

with open('undersampled_images_cnn.pkl', 'wb') as f:
    pickle.dump(list_xu, f)

with open('full_sampled_images_cnn.pkl', 'wb') as f:
    pickle.dump(list_xf, f)

print('Lists have been saved successfully.')
