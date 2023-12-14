# %%
import matplotlib.pyplot as plt
import torch

from mrpro.data import SpatialDimension
from mrpro.data._KData import KData
from mrpro.data.traj_calculators._KTrajectoryPulseq import KTrajectoryPulseq
from mrpro.operators._FourierOp import FourierOp
from mrpro.operators.models.WASABI import WASABI
from mrpro.operators.models.WASABITI import WASABITI

# %%
filepath = R'/home/hammac01/CEST_Data/'
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
offsets = torch.linspace(-250, 250, 101)
wasabi_model = WASABI(offsets=offsets)

qdata = torch.ones(4, 1, 5, 1, 1, 1)
qdata[0, ...] = 0  # b0_shift
qdata[1, ...] = 1.0
qdata[2, ...] = 1.0
qdata[3, ...] = 2.0

sig = wasabi_model.forward(qdata)

# %%
trec = torch.ones_like(offsets)

wasabiti_model = WASABITI(offsets=offsets, trec=trec)

qdata = torch.ones(4, 1, 1, 1, 1, 1)
qdata[0, ...] = 0  # b0_shift
qdata[1, ...] = 1.0
qdata[2, ...] = 10.0

sig2 = wasabiti_model.forward(qdata=qdata)

# %%
# fig, ax = plt.subplots()
# ax.plot(offsets, torch.abs(sig[0:101].squeeze()))
# plt.show()

fig, ax = plt.subplots()
ax.plot(torch.abs(sig2.squeeze()))
plt.show()

# %%
