import time

import torch
from mrpro.data import DcfData, KTrajectory

a = KTrajectory(kz=torch.ones(1, 1, 1, 1), ky=torch.rand(1, 1, 20, 200), kx=torch.rand(1, 1, 20, 200))
start = time.time()

d = DcfData.from_traj_voronoi(a)
print(time.time() - start)

# import numpy as np


# def flip_binary_sort(N):
#     # Generate array from 0 to N-1
#     arr = np.arange(N)

#     # Convert array elements to binary strings, reverse, and convert back to integers
#     return np.argsort([int(bin(i)[2:].zfill(int(np.ceil(np.log2(N))))[::-1], 2) for i in arr])

#     return sorted_indices


# # Example usage:
# N = 7
# indices = flip_binary_sort(N)
# print(indices)
