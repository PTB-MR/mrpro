# %%
import torch

from mrpro.data import SpatialDimension
from mrpro.data._KData import KData
from mrpro.data.traj_calculators._KTrajectoryPulseq import KTrajectoryPulseq
from mrpro.operators._FourierOp import FourierOp

# %%
filepath = R'Z:/_allgemein/projects/8_13/MRPro/hackathon2_qmri/pulseq_WASABITI/'
seq_filename = '20231127_WASABITI_adjusted_fov192.seq'
h5_filename = 'meas_MID00021_FID05730_20231127_WASABITI_adjusted_fov192.h5'

data = KData.from_file(
    ktrajectory=KTrajectoryPulseq(seq_path=filepath + seq_filename),
    filename=filepath + h5_filename,
)
# %%
# manually set kz
data.traj.kz = torch.zeros(data.traj.kz.shape[0], 1, 1, 1)
data.traj.ky = data.traj.ky.to(torch.float32)
data.traj.kx = data.traj.kx.to(torch.float32)

# %%
# create operator
op = FourierOp(im_shape=SpatialDimension(1, 192, 192), traj=data.traj, oversampling=SpatialDimension(1, 1, 1))

# apply adjoint operator
reco = torch.fft.fftshift(op.H(data.data))

# %%
