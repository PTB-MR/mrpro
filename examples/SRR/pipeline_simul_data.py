"""Class for Super Resolution Operator."""

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

import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import show_2D
from utils import show_3D

from mrpro.operators._SuperResOp import Slice_profile
from mrpro.operators._SuperResOp import SuperResOp

path_XCAT = '/../../echo/allgemein/projects/hufnag01/forMrPro/forEcho.npy'
vol_HR = torch.Tensor(np.load(path_XCAT))
vol_HR_zeros = torch.zeros([256, 256, 256])
size = 128

vol_HR_zeros[
    128 - size // 2 : 128 + size // 2,
    128 - size // 2 : 128 + size // 2,
    128 - size // 2 : 128 + size // 2,
] = vol_HR
vol_HR = torch.clone(vol_HR_zeros)

pos_center = torch.IntTensor([128, 128, 128])

vol_HR = vol_HR[
    pos_center[0] - size // 2 : pos_center[0] + size // 2,
    pos_center[1] - size // 2 : pos_center[1] + size // 2,
    pos_center[2] - size // 2 : pos_center[2] + size // 2,
]

shape_HR = vol_HR.shape

vol_HR = torch.swapaxes(vol_HR, 0, 2)
vol_HR = torch.swapaxes(vol_HR, 2, 1)

vol_HR = torch.flip(vol_HR, dims=[0])
vol_HR = torch.flip(vol_HR, dims=[2])

flag_show_orig = True
if flag_show_orig:
    for idx_proj in range(3):
        show_3D(vol_HR, axs_proj=idx_proj, title='vol_HR')


# vol_HR = vol_HR.unsqueeze(0).unsqueeze(0).expand(-1, 2, -1, -1, -1)
vol_HR = vol_HR[:, :, None, None, None]

num_slices_per_stack = 5
gap_slices = 14
thickness_slice = 4
# stacks are rotated around the septum (using the septum as rotation axis)
rot_per_stack = np.array(
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

offsets_stack = np.array([-6, 0, 0, 0, 0, 6, 6, -6, 6, -6, 6, -6])

num_stacks = rot_per_stack.shape[0]

slice_profile = Slice_profile(thickness_slice=thickness_slice)


srr_op = SuperResOp(
    shape_HR=shape_HR,
    num_slices_per_stack=num_slices_per_stack,
    gap_slices_inHR=gap_slices,
    num_stacks=num_stacks,
    thickness_slice_inHR=thickness_slice,
    offsets_stack_HR=offsets_stack,
    rot_per_stack=rot_per_stack,
    w=3 * slice_profile.sigma,
    slice_profile=slice_profile.rect,
)

slices_LR = srr_op.forward(vol_HR)

flag_show_LR_stacks = False
if flag_show_LR_stacks:
    for idx_stack in range(num_stacks):
        for idx_slice in range(1, 4):
            show_2D(slices_LR[idx_stack][idx_slice][0, 0, 0, ...], 'stack' + str(idx_stack) + '_slice' + str(idx_slice))


vol_HR_adjoint = srr_op.adjoint(slices_LR)[0]

for idx_proj in range(3):
    show_3D(vol_HR_adjoint[0, 0], axs_proj=idx_proj, title='vol_HR_adjoint_0')
    show_3D(vol_HR_adjoint[0, 1], axs_proj=idx_proj, title='vol_HR_adjoint_1')
plt.show()
