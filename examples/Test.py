# %%
# Import
import matplotlib.pyplot as plt
from mrpro.algorithms.reconstruction import DirectReconstruction
from mrpro.data import KData  # Import the KData class
from mrpro.data._SpatialDimension import SpatialDimension
from mrpro.data.traj_calculators import KTrajectoryPulseq

# Base path
base_path = '/data/bouill01/PTBSequences/PTBSequences/seqfiles/'

# Local paths
paths = {
    'US_26': {
        'h5_path': f'{base_path}newlabel/meas_MID00054_FID06138_cartesian_2d_nolabel_poiss_0_5_us_145_20240704_154645.h5',
        'seq_path': f'{base_path}newlabel/cartesian_2d_nolabel_poiss_0.5_us_145_20240704_154645.seq',
    },
    'US_71': {
        'h5_path': f'{base_path}newlabel/meas_MID00048_FID06132_cartesian_2d_nolabel_poiss_0_2_us_72_20240704_144429.h5',
        'seq_path': f'{base_path}newlabel/cartesian_2d_nolabel_poiss_0.2_us_72_20240704_144429.seq',
    },
    'US_49': {
        'h5_path': f'{base_path}newlabel/meas_MID00049_FID06133_cartesian_2d_nolabel_poiss_0_5_us_144_20240704_144327.h5',
        'seq_path': f'{base_path}newlabel/cartesian_2d_nolabel_poiss_0.5_us_144_20240704_144327.seq',
    },
    'US_90': {
        'h5_path': f'{base_path}newlabel/meas_MID00050_FID06134_cartesian_2d_nolabel_poiss_0_us_256_20240704_144404.h5',
        'seq_path': f'{base_path}newlabel/cartesian_2d_nolabel_poiss_0_us_256_20240704_144404.seq',
    },
    'US_141': {
        'h5_path': f'{base_path}meas_MID00033_FID06117_cartesian_2d_poisson_0_5_us_141_20240704_114326.h5',
        'seq_path': f'{base_path}cartesian_2d_poisson_0.5_us_141_20240704_114326.seq',
    },
    'US_143': {
        'h5_path': f'{base_path}meas_MID00034_FID06118_cartesian_2d_poisson_0_5_us_143_20240704_114322.h5',
        'seq_path': f'{base_path}cartesian_2d_poisson_0.5_us_143_20240704_114322.seq',
    },
    'US_49_no_label': {
        'h5_path': f'{base_path}meas_MID00046_FID06130_cartesian_2d_nolabel_poiss_0_5_us_49_20240704_142956.h5',
        'seq_path': f'{base_path}cartesian_2d_nolabel_poiss_0.5_us_49_20240704_142956.seq',
    },
}

# %%
# Load in the Data from the ISMRMRD file
kdatapuls = KData.from_file(paths['US_26']['h5_path'], KTrajectoryPulseq(seq_path=paths['US_26']['seq_path']))
kdatapuls.header.recon_matrix = SpatialDimension(z=1, y=256, x=256)

# Perform the reconstruction
reconstructionpuls = DirectReconstruction.from_kdata(kdatapuls)
# Use this to run on gpu: kdata = kdata.cuda()
img = reconstructionpuls(kdatapuls)

# Display the reconstructed image
# If there are multiple slices, ..., only the first one is selected
first_img = img.rss().cpu()[0, 0, :, :]  #  images, z, y, x
plt.matshow(first_img, cmap='gray')
plt.show()
