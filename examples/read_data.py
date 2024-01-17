# %%
import matplotlib.pyplot as plt
import torch

from mrpro.data._KData import KData
from mrpro.data.traj_calculators._KTrajectoryCartesian import KTrajectoryCartesian

# %%

h5_filename = (
    R'C:\Users\hammac01\Desktop\PythonCode\mrpro_test_data\meas_MID296_ssm_CVB1R_1sl_sag_trig400_FID39837_ismrmrd.h5'
)

data = KData.from_file(
    ktrajectory=KTrajectoryCartesian(),
    filename=h5_filename,
)


# %%
sortidx = torch.argsort(data.traj.ky, dim=-2, stable=True)
reshaped = torch.broadcast_to(sortidx.unsqueeze(1), data.data.shape)
sorted = torch.gather(data.data, -2, reshaped)

coilwise = torch.fft.fftshift(torch.fft.ifft2(sorted), dim=(-1, -2))
image = coilwise.abs().square().sum(1).sqrt()

image = image.squeeze()

# %%
for im in image:
    plt.matshow(im)
# %%
