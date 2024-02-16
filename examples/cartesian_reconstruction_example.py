# %%
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

from mrpro.data import CsmData
from mrpro.data import IData
from mrpro.data import SpatialDimension
from mrpro.data._KData import KData
from mrpro.data.traj_calculators._KTrajectoryCartesian import KTrajectoryCartesian
from mrpro.operators import SensitivityOp
from mrpro.operators._FourierOp import FourierOp

# %%

# flag for conversion and saving nifty files
# If you wish to save the resulting reconstructed image as a nifty fily, set  NIFTI = "True"
NIFTI = False

# %% [markdown]

# # Download example data
# The example data is taken from (https://zenodo.org/records/10669837).
# It entails GRE phantom scans with fully sampled cartesian trajectory.
# The sequence used was a pypulseq GRE sequence available under
# (https://github.com/imr-framework/pypulseq/blob/dev/pypulseq/seq_examples/scripts/write_gre.py)
# %%
# Create a temporary directory to save the downloaded files
data_folder = Path(tempfile.mkdtemp())
print(f'Data will be saved to: {data_folder}')

# Initialize attempt counter and set maximum number of attempts
attempt = 1
max_attempts = 3

# Start the download process
while attempt <= max_attempts:
    # Zenodo command for downloading
    zenodo_cmd = f'zenodo_get 10.5281/zenodo.10669837 -o {str(data_folder)}'

    # Execute the Zenodo command
    out = subprocess.call(zenodo_cmd, shell=True)

    # Check if the download was successful
    if out == 0:
        print('Download successful.')
        break
    else:
        print(f'Download failed! Attempt {attempt} of {max_attempts}')
        attempt += 1

if attempt > max_attempts:
    # Abort after reaching the maximum number of attempts
    raise ConnectionError('Zenodo download failed after 3 attempts!')

# %%
filepath = data_folder
seq_filename = 'gre_example.seq'
h5_filename = 'meas_MID00021_FID00266_pulseq.h5'

# %% [markdown]
# # Read in data
# Data is read in using the downloaded .h5 file and a corresponding
# trajectory calculator (cartesian in this case). Given the trajectory
# calculator, mrpro will automatically read, sort and validate the data.

# %%
data = KData.from_file(
    ktrajectory=KTrajectoryCartesian(),
    filename=f'{filepath}/{h5_filename}',
)

# %% [markdown]
# # Reconstruction
# Given the loaded data, the Fourier Operator is initialized next,
# with which the appropriate Fourier Transform is carried out.
# Furthermore, coil sensitivity maps are created to control for varying
# coil sensitivities throughout the scan.

# %%
# perform FT and CSM
ft_op = FourierOp(
    recon_shape=SpatialDimension(1, 256, 256),
    encoding_shape=SpatialDimension(1, 256, 256),
    traj=data.traj,
    oversampling=SpatialDimension(1, 1, 1),
)
(xcoils,) = ft_op.H(data.data)
idata = IData.from_tensor_and_kheader(xcoils, data.header)

smoothing_width = SpatialDimension(z=1, y=5, x=5)
csm = CsmData.from_idata_walsh(idata, smoothing_width)
sensitivity_op = SensitivityOp(csm)
(x,) = sensitivity_op.H(xcoils)

# %% [markdown]
# # See the results
# %%
image = x.squeeze(0)
for i in image:
    print(i.shape)
    plt.matshow(torch.abs(i[0]))
    break
# %% [markdown]
# # Save reconstructed images
# In case you would like to save the reconstructed images,
# set the NIFTI flag at the top to True
# %%
if NIFTI:
    image = image.swapaxes(0, 2)
    rotated = np.rot90(image.numpy(), 3)
    ni_img = nib.Nifti1Image(np.abs(image.numpy()), affine=np.eye(4))
    nib.save(
        ni_img,
        f'{filepath}/{h5_filename.split(".")[0]}/.nii',
    )
