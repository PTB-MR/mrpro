# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from __future__ import annotations

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import show_2D
from utils import show_3D

from mrpro.operators._SuperResOp import Slice_prof
from mrpro.operators._SuperResOp import SuperResOp

# load simulated data
path_XCAT = '/../../echo/allgemein/projects/8_13/MRPro/example_data/raw_data/SuperRes/Simul/XCAT.npy'
vol_HR = torch.Tensor(np.load(path_XCAT))

# prepare simulated data to match in vivo data
shape_HR = torch.IntTensor([vol_HR.shape[0], vol_HR.shape[1], vol_HR.shape[2]])
vol_HR = torch.swapaxes(vol_HR, 0, 2)
vol_HR = torch.swapaxes(vol_HR, 2, 1)
vol_HR = torch.flip(vol_HR, dims=[0])
vol_HR = torch.flip(vol_HR, dims=[2])

# show HR simulated data
flag_show_orig = True
if flag_show_orig:
    for idx_proj in range(3):
        show_3D(vol_HR, axs_proj=idx_proj, title='vol_HR', flag_T1map=True)

# simulate SuperRes geometry
num_slices_per_stack = 5
gap_slices_mm = 14
thickness_slice_LR_mm = 4
thickness_slice_HR_mm = 1.3
offsets_stack_mm = torch.Tensor([-6, 0, 0, 0, 0, 6, 6, -6, 6, -6, 6, -6])
rot_per_stack = torch.Tensor(
    [
        [90, 0, 0],
        [90, 0, 135],
        [90, 0, 90],
        [90, 0, 45],
        [90, 0, 0],
        [90, 0, 0],
        [90, 0, 135],
        [90, 0, 135],
        [90, 0, 90],
        [90, 0, 90],
        [90, 0, 45],
        [90, 0, 45],
    ]
)

num_stacks = rot_per_stack.shape[0]
slice_profile = Slice_prof(thickness_slice=thickness_slice_LR_mm)

# parametrize SuperResOp
srr_op = SuperResOp(
    shape_HR=shape_HR,
    num_slices_per_stack=num_slices_per_stack,
    gap_slices_inHR=gap_slices_mm / thickness_slice_HR_mm,
    num_stacks=num_stacks,
    thickness_slice_inHR=thickness_slice_LR_mm / thickness_slice_HR_mm,
    offsets_stack_HR=offsets_stack_mm / thickness_slice_HR_mm,
    rot_per_stack=rot_per_stack,
    w=3 * slice_profile.sigma,
    slice_prof=slice_profile.rect,
)

# simulate LR slices
vol_HR = vol_HR[None, None]
other, coils, z, y, x = vol_HR.shape
slices_LR = torch.tensor(srr_op.forward(vol_HR))
slices_LR = einops.rearrange(
    slices_LR,
    '(num_stacks num_slices_per_stack other coils 1 y x) -> num_stacks num_slices_per_stack other coils 1 y x',
    coils=coils,
    y=y,
    x=x,
    num_stacks=num_stacks,
    other=other,
    num_slices_per_stack=num_slices_per_stack,
)

# show LR slices
flag_show_LR_stacks = False
if flag_show_LR_stacks:
    for idx_stack in range(num_stacks):
        for idx_slice in range(1, 4):
            show_2D(
                slices_LR[idx_stack][idx_slice][0, 0, 0, ...],
                'stack' + str(idx_stack) + '_slice' + str(idx_slice),
                flag_T1map=True,
            )

# combine LR slices to HR volume
vol_HR_adjoint = srr_op.adjoint(slices_LR)[0]

# show HR volume
for idx_proj in range(3):
    show_3D(vol_HR_adjoint[0, 0], axs_proj=idx_proj, title='vol_HR_adjoint', flag_T1map=True)
plt.show()
