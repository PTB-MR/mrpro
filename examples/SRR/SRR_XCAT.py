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

from mrpro.operators._SuperResOp import Slice_profile
from mrpro.operators._SuperResOp import SuperResOp


def show_3D(vol, axs_proj, title='', idx_slice=None, cmap=None):
    if cmap is None:
        cmap = plt.get_cmap('gray')
    shape_vol = vol.shape
    if axs_proj == 0:
        idx_slice = shape_vol[0] // 2 if idx_slice is None else idx_slice
        img = vol[idx_slice]
    elif axs_proj == 1:
        idx_slice = shape_vol[1] // 2 if idx_slice is None else idx_slice

        img = vol[:, idx_slice]
    elif axs_proj == 2:
        idx_slice = shape_vol[2] // 2 if idx_slice is None else idx_slice
        img = vol[:, :, idx_slice]
    else:
        raise Exception()

    plt.matshow(img)
    if title == '':
        title = 'img'
    plt.title(title + '_proj' + str(axs_proj))


def show_2D(img, title=''):
    plt.matshow(img, cmap=plt.get_cmap('grey'))
    if title == '':
        title = 'img'
    plt.title(title)
    plt.show()


def simulHR(flag_show=True):
    img_square = torch.zeros(64, 64)
    img_square[22:42, 22:42] = 1

    square_vol = img_square.unsqueeze(0).repeat(64, 1, 1)

    if flag_show:
        plt.matshow(img_square)
        plt.title('img_square')
    return square_vol


# vol_HR = simulHR(flag_show = True)
# vol_HR_exp = vol_HR.unsqueeze(0).unsqueeze(0).expand(-1, 2, -1, -1, -1)

vol_HR_orig = torch.tensor(np.load('/../../echo/allgemein/projects/hufnag01/forMrPro/xcatRef.npy')[16:112, :, :96])
shape_HR = vol_HR_orig.shape

for idx_proj in range(3):
    show_3D(vol_HR_orig, axs_proj=idx_proj, title='vol_HR_orig')

vol_HR_orig = vol_HR_orig.unsqueeze(0).unsqueeze(0).expand(-1, 2, -1, -1, -1)

num_slices_per_stack = 5
gap_slices = 4
thickness_slice = 4
# stacks are rotated around the septum (using the septum as rotation axis)
rot_per_stack = [[90, 0, 0], [90, 0, 45], [90, 0, 90], [90, 0, 135]]
num_stacks = len(rot_per_stack)

slice_profile = Slice_profile(thickness_slice=thickness_slice)

srr_op = SuperResOp(
    shape_HR=shape_HR,
    num_slices_per_stack=num_slices_per_stack,
    gap_slices=gap_slices,
    num_stacks=num_stacks,
    thickness_slice=thickness_slice,
    list_offsets_stack=np.zeros([num_stacks, 3]),
    rot_per_stack=rot_per_stack,
    w=3 * slice_profile.sigma,
    slice_profile=slice_profile.rect,
)

slices_LR = srr_op.forward(vol_HR_orig)
vol_HR_adjoint = srr_op.adjoint(slices_LR)


for idx_proj in range(3):
    show_3D(vol_HR_adjoint[0, 0], axs_proj=idx_proj, title='vol_HR_adjoint_0')
    show_3D(vol_HR_adjoint[0, 1], axs_proj=idx_proj, title='vol_HR_adjoint_1')


flag_showLRstacks = False
if flag_showLRstacks:
    for idx_stack in range(num_stacks):
        for idx_slice in range(num_slices_per_stack):
            show_2D(slices_LR[idx_stack][idx_slice][0, 0, 0, ...], 'stack' + str(idx_stack) + '_slice' + str(idx_slice))

plt.show()
