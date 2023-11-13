# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchkbnufft as tkbn
from einops import rearrange

from mrpro.data._DcfData import DcfData
from mrpro.data._KData import KData
from mrpro.data.traj_calculators._KTrajectorySeq import KTrajectorySeq

filepath = (
    R"../../CEST_Data/2023-10-30_CEST_JOHANNES/meas_MID00072_FID04741_20231026_glycoCEST_fov192_offsets_44_b1_2p80/"
)

data = KData.from_file(
    ktrajectory=KTrajectorySeq(path=filepath + "20231026_glycoCEST_fov192_offsets_44_b1_2p80.seq"),
    filename=filepath + "meas_MID00072_FID04741_20231026_glycoCEST_fov192_offsets_44_b1_2p80.h5",
)
# %%
sortidx = torch.argsort(data.traj.ky, dim=-2, stable=True)
reshaped = torch.broadcast_to(sortidx.unsqueeze(1), data.data.shape)
sorted = torch.gather(data.data, -2, reshaped)
# sorted = data.data[...,reshaped,:]

coilwise = torch.fft.fftshift(torch.fft.ifft2(sorted), dim=(-1, -2))
image = coilwise.abs().square().sum(1).sqrt()

image = image.squeeze()
for im in image:
    plt.matshow(im)

# %%
import nibabel as nib

image = image.swapaxes(0, 2)
rotated = np.rot90(image.numpy(), 3)
ni_img = nib.Nifti1Image(np.abs(image.numpy()), affine=np.eye(4))
nib.save(
    ni_img,
    filepath + "meas_MID00072_FID04741_20231026_glycoCEST_fov192_offsets_44_b1_2p80_2.nii",
)

# %%
loaded_img = nib.load(
    filepath + "meas_MID00072_FID04741_20231026_glycoCEST_fov192_offsets_44_b1_2p80_2.nii"
).get_fdata()

# %%
data2 = KData.from_file(
    ktrajectory=KTrajectorySeq(path=filepath + "20231010_tse_golden_radial_fov192_128px_etl1_128ang.seq"),
    filename=filepath + "meas_MID00073_FID03746_20231010_tse_golden_radial_fov192_128px_etl1_128ang.h5",
)

# %%
for i in range(0, len(data.traj.kx[0, 0, :, 0])):
    plt.plot(np.array(data.traj.kx[0, 0, i, :]), np.array(data.traj.ky[0, 0, i, :]))
plt.show()

# %%
im_size = (data.header.recon_matrix.x, data.header.recon_matrix.x)
kbnufft_adj = tkbn.KbNufftAdjoint(im_size=im_size)

kdcf = DcfData.from_traj_voronoi(data.traj)
dcomp = kdcf.data
dcomp = rearrange(dcomp, "other other2 Nt dc->Nt other other2 dc")
dcomp = dcomp.squeeze(1)

kdata = data.data * dcomp

kdata_kbnufft = rearrange(kdata, "other coils k2 k1 k0->other coils (k2 k1 k0)").to(torch.complex64)
ktraj_kbnufft = rearrange(data.traj.as_tensor()[1:, ...], "dim other k2 k1 k0->other dim (k2 k1 k0)").to(torch.float32)

im = kbnufft_adj(kdata_kbnufft, ktraj_kbnufft)
cmap = plt.cm.Greys_r
plt.imshow(np.abs(im[0, -1, :, :]), cmap=cmap)
plt.show()
# %%
