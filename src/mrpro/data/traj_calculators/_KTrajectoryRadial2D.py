# %%
# Imports
from mrpro.data import KTrajectory
from mrpro.data import KHeader
from mrpro.data.traj_calculators import KTrajectoryCalculator

import numpy as np

# %%
class KTrajectoryRadial2D(KTrajectoryCalculator):
    
    def __init__(
        self,
        angle: float = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2)),
    ) -> None:
        super().__init__()
        self.angle: float = angle
    
    def __call__(self, header: KHeader) -> KTrajectory:
        """Calculate dummy trajectory for given KHeader."""
        
        
        num_samples = header.acq_info.number_of_samples
        
        if not (torch.all(num_samples == num_samples[0])):
            raise ValueError("Different number of readout sample points are not supported.")
        
        # num_spokes = header.encoding_limits.k1.max - header.encoding_limits.k1.min
        
        # center_sample = header.acq_info.center_sample
        
        # Calculate points along readout

        # k0 = torch.linspace(0, nk0 - 1, nk0, dtype=torch.float32) - center_sample[0, 0, 0]
        # k0 *= 2 * torch.pi / nk0

        
        # kx = torch.zeros(1, 1, 1, header.encoding_limits.k0.length)
        # ky = kz = torch.zeros(1, 1, 1, 1)
        
        # n_spokes, n_readout = self._get_shape(header)[-2], self._get_shape(header)[-1]

        # # length of each spoke;
        # spoke_length = n_readout

        # # golden angle radial increment
        # if self.golden_angle is True:
        #     del_theta = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))

        # # create radial coordinates
        # ktheta = torch.arange(0, del_theta*n_spokes, del_theta)
        # kradius = torch.linspace(-np.pi, np.pi, spoke_length)  # format of trajectories kbNUFFT package expects to be

        # # construct transformation matrices
        # ktraj = torch.zeros(self._get_shape(header))
        # ktraj_init = torch.stack((kradius, torch.zeros(n_readout), torch.zeros(n_readout)))
        # for i in range(n_spokes):
        #     theta = ktheta[i]
        #     rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
        #                             [torch.sin(theta), torch.cos(theta), 0],
        #                             [0, 0, 1]])
        #     ktraj[..., i, :] = torch.matmul(rot_mat, ktraj_init).unsqueeze(1).unsqueeze(0)

        # # skip if all optional trajectory information are missing else do something
        # if torch.all(header.acq_info.trajectory_dimensions == 0):
        #     pass
        # else:
        #     pass  # do something: replace the calculated trajectory information with the already present one

        # return ktraj
        
        
        
        return KTrajectory(kz, ky, kx)
    
# %%

test = KTrajectoryRadial2D()

out = test()
# %%
