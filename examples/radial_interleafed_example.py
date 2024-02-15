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
from mrpro.data._DcfData import DcfData
from mrpro.data._KData import KData
from mrpro.data.traj_calculators._KTrajectoryRadial2D import KTrajectoryRadial2D
from mrpro.operators import SensitivityOp
from mrpro.operators._FourierOp import FourierOp

# %%

# flag for conversion and saving nifty files
# If you wish to save the resulting reconstructed image as a nifty fily, set  NIFTI = "True"
NIFTI = False
# %%

data_folder = Path(tempfile.mkdtemp())
zenodo_cmd = f'zenodo_get 10.5281/zenodo.10664974 -o {str(data_folder)}'
out = subprocess.call(zenodo_cmd, shell=True)

if out != 0:
    raise ConnectionError('Zenodo donload failed!')

# %%
# for single file
filepath = data_folder
seq_filename = '20240130_CEST_interleafed_radial_256px_fov256_8mm_50spokes_100ms.seq'
h5_filename = 'meas_MID00044_FID00370_20240130_CEST_interleafed_radial_256px_fov256_8mm_50spokes_100m.h5'


# %%
data = KData.from_file(
    ktrajectory=KTrajectoryRadial2D(),
    filename=f'{filepath}/{h5_filename}',
)
# %%
# Densitiy compensation
dcf = DcfData.from_traj_voronoi(traj=data.traj)

# %%
# perform FT and CSM
ft_op = FourierOp(
    # im_shape=SpatialDimension(1, 256, 256),
    recon_shape=SpatialDimension(1, 256, 256),
    encoding_shape=SpatialDimension(1, 256, 256),
    traj=data.traj,
    oversampling=SpatialDimension(1, 1, 1),
)
xcoils = ft_op.H(data.data * dcf.data)
idata = IData.from_tensor_and_kheader(xcoils, data.header)

smoothing_width = SpatialDimension(z=1, y=5, x=5)
csm = CsmData.from_idata_walsh(idata, smoothing_width)
sensitivity_op = SensitivityOp(csm)
x = sensitivity_op.H(xcoils)

# %%
image = x.abs().square().sum(1).sqrt()
image = image.squeeze()
for i in image:
    print(i.shape)
    plt.matshow(torch.abs(i))
    break

# idata = IData.from_tensor_and_kheader(im, data.header)
# %%
if NIFTI:
    image = image.swapaxes(0, 2)
    rotated = np.rot90(image.numpy(), 3)
    ni_img = nib.Nifti1Image(np.abs(image.numpy()), affine=np.eye(4))
    nib.save(
        ni_img,
        f'{filepath}/{h5_filename.split(".")[0]}/.nii',
    )
# %%

# %%
# test_load = nib.load(
#    "/home/hammac01/CEST_Data/2024-01-25_JOHANNES_Interleafed_CEST/Transversal/20240123_CEST_interleafed_radial_256px_fov256_8mm_200spokes_golden_angle_56offsets_0.035saturation_dummy_spokes/meas_MID00058_FID00269_0_035saturation.nii"
# ).get_fdata()
# test_load = test_load.swapaxes(0, 2)

# %%
# for i in test_load:
#    print(i.shape)
#    plt.matshow(i)
#    break

# %%
