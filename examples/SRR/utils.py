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

import math
import os
import pickle
from typing import Any
from typing import Union

import torch
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap

from mrpro.data import SpatialDimension


def save_object(obj: Any, filename: str) -> None:
    with open(filename, 'wb') as output:
        type_protocol = 4
        pickle.dump(obj, output, type_protocol)
    print('object saved in ' + str(filename))


def load_object(filename: str) -> Any:
    filename = os.path.abspath(filename)
    with open(filename, 'rb') as pickle_file:
        storedConfig = pickle.load(pickle_file)
    print('loaded object ' + str(filename))
    return storedConfig


def check_if_path_exists(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(os.path.abspath(path))
        print('created path ' + str(path))


def normed_vec(vector: torch.Tensor) -> torch.Tensor:
    if vectorLength(vector) == 0.0:
        return vector
    return vector / vectorLength(vector)


class ScanInfo:
    def __init__(self, path_scanInfo: str, num_slices=Union[int, None], num_stacks=Union[int, None]):

        fid = open(path_scanInfo)

        self.pathes_h5 = []
        self.pathes_ecg = []
        self.num_slices = num_slices
        self.thickness_slice = 8
        self.flag_invertEcg = False
        self.flip_angle = 9
        self.res_in_plane = 1.3
        self.num_stacks = num_stacks
        self.flag_one_slice_per_stack = False

        flag_line_is_folder_path = False
        flag_line_is_slice_thickness = False
        flag_line_is_num_slices = False
        flag_line_is_num_stacks = False
        flag_line_is_dataset_path = False
        flag_line_is_ecg_path = False
        flag_line_is_invert_ecg = False
        flag_line_is_1_slice_per_stack = False
        flag_line_is_flip_angle = False
        flag_line_is_res_in_plane = False

        while True:
            cline = fid.readline()
            if not cline:
                break
            if '#' in cline:
                continue
            if flag_line_is_folder_path:
                self.path_folder = cline.split('\n')[0]
                flag_line_is_folder_path = False
            elif flag_line_is_slice_thickness:
                self.thickness_slice = int(cline.split('\n')[0])
                flag_line_is_slice_thickness = False
            elif flag_line_is_res_in_plane:
                self.res_in_plane = int(cline.split('\n')[0]) / 10.0
                flag_line_is_res_in_plane = False
            elif flag_line_is_invert_ecg:
                bool_str = cline.split('\n')[0]
                self.flag_invertEcg = True if bool_str == 'True' else False
                flag_line_is_invert_ecg = False
            elif flag_line_is_1_slice_per_stack:
                bool_str = cline.split('\n')[0]
                self.flag_one_slice_per_stack = True if bool_str == 'True' else False
                flag_line_is_1_slice_per_stack = False
            elif flag_line_is_flip_angle:
                self.flip_angle = int(cline.split('\n')[0])
                flag_line_is_flip_angle = False
            elif flag_line_is_num_slices:
                self.num_slices = int(cline)
                flag_line_is_num_slices = False
            elif flag_line_is_num_stacks:
                self.num_stacks = int(cline)
                flag_line_is_num_stacks = False
            elif flag_line_is_dataset_path:
                without_line_break = cline.split('\n')[0]
                if without_line_break == '':
                    flag_line_is_dataset_path = False
                    continue
                if '#' not in without_line_break:
                    self.pathes_h5.append(without_line_break.split(': ')[1])
            elif flag_line_is_ecg_path:
                without_line_break = cline.split('\n')[0]
                if without_line_break == '':
                    flag_line_is_ecg_path = False
                    continue
                if '#' not in without_line_break:
                    self.pathes_ecg.append(without_line_break.split(': ')[1])
            elif 'folderpath' in cline or 'folderPath' in cline:
                flag_line_is_folder_path = True
            elif 'sliceThickness' in cline:
                flag_line_is_slice_thickness = True
            elif 'res_inPlane' in cline:
                flag_line_is_res_in_plane = True
            elif 'datasetPathes' in cline:
                flag_line_is_dataset_path = True
            elif 'ecgFilePathes' in cline:
                flag_line_is_ecg_path = True
            elif 'numberOfSlices' in cline or 'num_slices' in cline:
                flag_line_is_num_slices = True
            elif 'num_stacks' in cline:
                flag_line_is_num_stacks = True
            elif 'flag_invertEcg' in cline:
                flag_line_is_invert_ecg = True
            elif 'flag_oneSlicePerStack' in cline:
                flag_line_is_1_slice_per_stack = True
            elif 'flipAngle' in cline:
                flag_line_is_flip_angle = True


def dist_3D(pointOne: torch.Tensor, pointTwo: torch.Tensor) -> float:
    return math.sqrt(
        (pointOne[0] - pointTwo[0]) ** 2 + (pointOne[1] - pointTwo[1]) ** 2 + (pointOne[2] - pointTwo[2]) ** 2
    )


def toTensor(spaDim: SpatialDimension[torch.Tensor]) -> torch.Tensor:
    return torch.Tensor([spaDim.x[0, 0, 0, 0], spaDim.y[0, 0, 0, 0], spaDim.z[0, 0, 0, 0]])


def angleBetweenVec(
    vectorA: None | torch.Tensor, vectorB: None | torch.Tensor, normal: None | torch.Tensor = None
) -> float:

    assert type(vectorA) is torch.Tensor and type(vectorB) is torch.Tensor

    vectorA_norm = vectorA / vectorLength(vectorA)
    vectorB_norm = vectorB / vectorLength(vectorB)

    if torch.equal(vectorA, vectorB):
        return 0.0
    dotProduct = torch.dot(vectorA_norm, vectorB_norm)
    if dotProduct < -1.0:
        return math.degrees(math.acos(-1.0))
    if dotProduct > 1.0:
        return math.degrees(math.acos(1.0))

    angle_rad = math.acos(torch.dot(vectorA_norm, vectorB_norm))
    angle_deg = math.degrees(angle_rad)

    if normal is not None:
        if torch.dot(normal, torch.cross(vectorA_norm, vectorB_norm)) < 0:
            angle_deg *= -1.0
    return angle_deg


def vectorLength(vector: torch.Tensor) -> float:
    return math.sqrt(sum(i * i for i in vector))


def show_3D(
    vol: torch.Tensor,
    axs_proj: int,
    title: str = 'img',
    idx_slice: None | int = None,
    cmap: None | Colormap = None,
    vmin: None | float = None,
    vmax: None | float = None,
) -> None:

    flag_isT1map = False
    if flag_isT1map:
        cmap = plt.get_cmap('jet')
        vmin = 0
        vmax = 2.5
    else:
        cmap = plt.get_cmap('gray')
        vmin = vmin
        vmax = vmax
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

    plt.matshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title + '_proj' + str(axs_proj))


def show_2D(img: torch.Tensor, title: str = 'img', vmin: None | float = None, vmax: None | float = None) -> None:
    flag_isT1map = False
    if flag_isT1map:
        cmap = plt.get_cmap('jet')
        vmin = 0
        vmax = 2.5
    else:
        cmap = plt.get_cmap('gray')
        vmin = vmin
        vmax = vmax
    plt.matshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.show()
