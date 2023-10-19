# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchkbnufft as tkbn
from einops import rearrange

from mrpro.data._DcfData import DcfData
from mrpro.data._KData import KData
from mrpro.data.traj_calculators._KTrajectorySeq import KTrajectorySeq

filepath = 'C:/Users/hammac01/Desktop/PythonCode/mrpro_test_data/'

data = KData.from_file(
    ktrajectory=KTrajectorySeq(path=filepath + '20231010_tse_golden_radial_fov192_128px_etl1_128ang.seq'),
    filename=filepath + 'meas_MID00073_FID03746_20231010_tse_golden_radial_fov192_128px_etl1_128ang.h5',
)

# %%
for i in range(0, len(data.traj.kx[0, 0, :, 0])):
    plt.plot(np.array(data.traj.kx[0, 0, i, :]), np.array(data.traj.ky[0, 0, i, :]))
plt.show()

# %%
im_size = (data.header.recon_matrix.y, data.header.recon_matrix.y)
kbnufft_adj = tkbn.KbNufftAdjoint(im_size=im_size)

kdcf = DcfData.from_traj_voronoi(data.traj)
dcomp = kdcf.data
dcomp = rearrange(dcomp, 'other other2 Nt dc->Nt other other2 dc')
dcomp = dcomp.squeeze(1)

kdata = data.data * dcomp

kdata_kbnufft = rearrange(kdata, 'other coils k2 k1 k0->other coils (k2 k1 k0)').to(torch.complex64)
ktraj_kbnufft = rearrange(data.traj.as_tensor()[1:, ...], 'dim other k2 k1 k0->other dim (k2 k1 k0)').to(torch.float32)

im = kbnufft_adj(kdata_kbnufft, ktraj_kbnufft)
cmap = plt.cm.Greys_r
plt.imshow(np.abs(im[0, -1, :, :]), cmap=cmap)
plt.show()
# %%
