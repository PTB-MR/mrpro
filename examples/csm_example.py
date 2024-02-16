# %%
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from mrpro.data import SpatialDimension
from mrpro.data._CsmData._CsmData import CsmData
from mrpro.data._IData import IData
from mrpro.data._KData import KData
from mrpro.data.traj_calculators._KTrajectoryCartesian import KTrajectoryCartesian
from mrpro.operators import SensitivityOp
from mrpro.operators._FourierOp import FourierOp

# %%
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

# %%
data = KData.from_file(
    ktrajectory=KTrajectoryCartesian(),
    filename=f'{filepath}/{h5_filename}',
)
# %%
# create operator
op = FourierOp(
    recon_shape=SpatialDimension(1, 256, 256),
    encoding_shape=SpatialDimension(1, 256, 256),
    traj=data.traj,
    oversampling=SpatialDimension(1, 1, 1),
)

# %%
(xcoils,) = op.H(data.data)
idata = IData.from_tensor_and_kheader(xcoils, data.header)

smoothing_width = SpatialDimension(z=1, y=5, x=5)
# csm_walsh = CsmData.from_idata_walsh(idata, smoothing_width)
csm = CsmData.coil_map_study_2d_Inati(data=idata.data.squeeze(), ks=1, power=1)
sensitivity_op = SensitivityOp(csm)
(x,) = sensitivity_op.H(xcoils)
# %%
image = x.abs().square().sum(1).sqrt()
image = image.squeeze()
for i in image:
    print(i.shape)
    plt.matshow(torch.abs(i))
    break
