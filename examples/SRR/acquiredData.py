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

import os
from collections.abc import Callable

import ismrmrd
import numpy as np
import torch
from matplotlib import pyplot as plt
from utils import ScanInfo
from utils import angleBetweenVec
from utils import check_if_path_exists
from utils import dist_3D
from utils import load_object
from utils import normed_vec
from utils import save_object
from utils import show_2D
from utils import show_3D
from utils import toTensor

from mrpro.data import CsmData
from mrpro.data import DcfData
from mrpro.data import IData
from mrpro.data import KData
from mrpro.data.traj_calculators import KTrajectoryIsmrmrd
from mrpro.data.traj_calculators import KTrajectoryRadial2D
from mrpro.operators import FourierOp
from mrpro.operators import SensitivityOp
from mrpro.operators._SuperResOp import Slice_profile
from mrpro.operators._SuperResOp import SuperResOp


def showImgsOfStack(im: torch.Tensor, idx_stack: int) -> None:
    # Visualize results
    num_slices = im.shape[0]
    for idx_slice in range(num_slices):
        show_2D(
            torch.abs(im[idx_slice]),
            title='stack ' + str(idx_stack) + '_slice' + str(idx_slice),
        )


def calcDistToStack0(list_sGeometry: list[SGeometry], rots: torch.Tensor) -> torch.Tensor:
    num_slicesPerStack = list_sGeometry[0].pos_slices.shape[0]
    array_dist = torch.zeros(len(rots))
    for rot_unique in list(torch.unique(rots)):
        idx_withSameRot = list(torch.where(rots == rot_unique))[0]
        if list(rots).count(rot_unique) == 1:
            array_dist[idx_withSameRot[0]] = 0.0
        else:
            idx_sliceToCompare = 2
            pos_stackGroup_0 = list_sGeometry[idx_withSameRot[0]].pos_slices[idx_sliceToCompare]
            dist_withinRotGroup = torch.zeros(len(idx_withSameRot))
            # check if all slice dirs within grouop are the same
            sliceDirOfGroup = torch.zeros([len(idx_withSameRot), 3])
            for idx_inGroup in range(0, len(idx_withSameRot)):
                sliceDir = (
                    list_sGeometry[idx_withSameRot[idx_inGroup]].pos_slices[num_slicesPerStack - 1]
                    - list_sGeometry[idx_withSameRot[idx_inGroup]].pos_slices[0]
                )
                sliceDirOfGroup[idx_inGroup] = normed_vec(sliceDir)

            for idx_inGroup in range(0, len(idx_withSameRot)):
                pos_stack = torch.tensor(list_sGeometry[idx_withSameRot[idx_inGroup]].pos_slices[idx_sliceToCompare])
                stackDir = pos_stack - pos_stackGroup_0
                dist_withinRotGroup[idx_inGroup] = dist_3D(
                    pos_stackGroup_0, list_sGeometry[idx_withSameRot[idx_inGroup]].pos_slices[idx_sliceToCompare]
                )

                if torch.dot(normed_vec(stackDir), normed_vec(sliceDirOfGroup[idx_inGroup])) < 0:
                    dist_withinRotGroup[idx_inGroup] *= -1.0

            dist_withinRotGroup -= torch.mean(dist_withinRotGroup)
            for idx_inGroup in range(len(idx_withSameRot)):
                array_dist[idx_withSameRot[idx_inGroup]] = dist_withinRotGroup[idx_inGroup]
    return array_dist


def calcGapBetweenLR(list_sGeometries: list[SGeometry], thickness_slice: float) -> float:
    num_stacks = len(list_sGeometries)
    slicesPerStack = list_sGeometries[0].pos_slices.shape[0]
    if slicesPerStack <= 1:
        return 0
    else:
        gap = torch.zeros([num_stacks, slicesPerStack - 1])
        for idx_stack in range(num_stacks):
            for idx_slice in range(slicesPerStack - 1):
                dist = dist_3D(
                    list_sGeometries[idx_stack].pos_slices[idx_slice],
                    list_sGeometries[idx_stack].pos_slices[idx_slice + 1],
                )
                gap[idx_stack, idx_slice] = round(dist - thickness_slice, 2)
    return float(torch.median(gap))


def calc_rotToStack0(list_sGeometries: list[SGeometry], img_stacks: torch.Tensor) -> torch.Tensor:
    num_stacks = len(list_sGeometries)

    # search for sign change in angles
    dir_slice_stack0 = list_sGeometries[0].dir.slice
    dir_phase_stack0 = list_sGeometries[0].dir.phase

    # check if read and phase dir need to be swapped
    for idx_stack in range(1, num_stacks):
        dir_phase_stack = list_sGeometries[idx_stack].dir.phase
        angle_phase = round(number=angleBetweenVec(vectorA=dir_phase_stack0, vectorB=dir_phase_stack), ndigits=0)

        if abs(angle_phase - 90.0) < 10.0:
            dir_read_old = torch.clone(list_sGeometries[idx_stack].dir.read)
            list_sGeometries[idx_stack].dir.read = list_sGeometries[idx_stack].dir.phase
            list_sGeometries[idx_stack].dir.phase = dir_read_old
            img_stacks[idx_stack] = torch.swapaxes(img_stacks[idx_stack], 1, 2)
            print('... swap read and phase dir ')

    rots_toStack0 = torch.zeros([num_stacks, 3])

    for idx_stack in range(num_stacks):
        angle_slice = round(
            angleBetweenVec(
                dir_slice_stack0,
                list_sGeometries[idx_stack].dir.slice,
                normal=list_sGeometries[idx_stack].dir.phase,
            ),
            0,
        )
        angle_phase = round(angleBetweenVec(vectorA=dir_phase_stack0, vectorB=list_sGeometries[idx_stack].dir.phase), 1)

        rot = torch.tensor([angle_slice, 0, 0])
        rots_toStack0[idx_stack] = rot
    return rots_toStack0


def getSliceOrder(num_slices: int) -> list[int]:
    if num_slices == 6:
        order_slices = [3, 0, 4, 1, 5, 2]
    elif num_slices == 5:
        order_slices = [0, 3, 1, 4, 2]
    elif num_slices == 12:
        order_slices = [6, 0, 7, 1, 8, 2, 9, 3, 10, 4, 11, 5]
    else:
        raise Exception

    return order_slices


def resortSlices(imgs: torch.Tensor, list_sGeometries: list[SGeometry]) -> None:
    num_slices = imgs.shape[1]
    order_slices = getSliceOrder(num_slices=num_slices)

    imgs_before = torch.clone(imgs)
    for slice in range(len(order_slices)):
        imgs[:, slice] = imgs_before[:, int(order_slices[slice])]

    for idx_stack in range(imgs.shape[0]):
        pos_before = torch.clone(list_sGeometries[idx_stack].pos_slices)
        for slice in range(len(order_slices)):
            list_sGeometries[idx_stack].pos_slices[slice] = pos_before[int(order_slices[slice])]

    print('resorted slices with order ' + str(order_slices))


class Directions:
    def __init__(self) -> None:
        self.read: torch.Tensor
        self.slice: torch.Tensor
        self.phase: torch.Tensor

    def makeSureExists(self):
        assert type(self.read) is torch.Tensor
        assert type(self.slice) is torch.Tensor
        assert type(self.phase) is torch.Tensor


class SGeometry:
    def __init__(self) -> None:
        self.pos_slices: torch.Tensor
        self.dir = Directions()


# Golden angle radial acquisition of Kirstens T1 mapping sequence
def traj_rad_kirsten_t1mapping(nkrad: int, nkang: int) -> np.ndarray:
    krad = np.linspace(-nkrad // 2, nkrad // 2, nkrad)
    ky = np.linspace(0, nkang - 1, nkang)

    # Check if angle is already specified:
    angRad = ky * (np.pi / 180) * (180 * 0.618034)

    # Calculate trajectory
    rad_traj = np.zeros((nkrad, nkang, 2), dtype=np.float32)
    rad_traj[:, :, 0] = krad.reshape(-1, 1) * np.cos(angRad)
    rad_traj[:, :, 1] = krad.reshape(-1, 1) * np.sin(angRad)
    rad_traj = np.moveaxis(rad_traj, 0, 1)
    return rad_traj


def prep_filename(full_filename_in: str, full_filename_out: str | None = None, suffix: str = 'traj') -> str:
    if full_filename_out is None:
        full_filename_out = full_filename_in.replace('.h5', f'_{suffix}.h5')

    if full_filename_in == full_filename_out:
        raise ValueError('Input and output filename are the same. This would overwrite the original data.')

    # Set trajectory and save
    if os.path.exists(full_filename_out) == 1:
        os.remove(full_filename_out)
        print(f'{full_filename_out} deleted')

    return full_filename_out


# Add 2D non-Cartesian trajectory to ISMRMD data file
def add_2D_noncart_traj_to_ismrmrd(
    fun_calc_traj: Callable[[int, int], np.ndarray],
    full_filename_in: str,
    full_filename_out: str,
    sGeometry: SGeometry,
    nrad: int | None = None,
) -> str:

    # Get info and acquisitions from original data
    data_orig = KData.from_file(full_filename_in, KTrajectoryRadial2D())
    sGeometry.dir.read = toTensor(data_orig.header.acq_info.read_dir)
    sGeometry.dir.slice = toTensor(data_orig.header.acq_info.slice_dir)
    sGeometry.dir.phase = toTensor(data_orig.header.acq_info.phase_dir)

    with ismrmrd.File(full_filename_in, 'r') as file:
        ds = file[list(file.keys())[0]]
        ismrmrd_header = ds.header
        acquisitions = ds.acquisitions[:]

        # sliceThickness = ismrmrd_header.encoding[0].reconSpace.fieldOfView_mm.z
        # fligAngle = ismrmrd_header.sequenceParameters.flipAngle_deg

    # Create new file
    ds = ismrmrd.Dataset(full_filename_out)
    ds.write_xml_header(ismrmrd_header.toXML())

    # Calculate trajectory
    grad_traj = fun_calc_traj(data_orig.data.shape[-1], data_orig.data.shape[-2])

    # Acquisition time
    acqt = torch.tensor([acq.acquisition_time_stamp for acq in acquisitions], dtype=torch.float32)

    # Slice index
    sl = torch.tensor([acq.idx.slice for acq in acquisitions], dtype=torch.float32)

    num_slices = len(torch.unique(sl))
    idx_slice = 0
    sGeometry.pos_slices = torch.zeros([num_slices, 3])
    # Go through all slices
    for sl_idx in torch.unique(sl):
        cidx = torch.where(sl == sl_idx)[0]

        # Sort based on acquisition time step
        acqt_slice_idx = torch.argsort(acqt[cidx])

        # Remove noise sample
        if len(acqt_slice_idx) == data_orig.data.shape[-2] + 1:
            ds.append_acquisition(acquisitions[acqt_slice_idx[0]])
            acqt_slice_idx = acqt_slice_idx[1:]

        # Make sure all acquisitions are for imaging and not e.g. noise samples are present
        if len(acqt_slice_idx) != data_orig.data.shape[-2]:
            raise ValueError(
                f'Expected number of radial lines: {data_orig.data.shape[-2]} but found {len(acqt_slice_idx)}.'
            )

        # Select only nrad lines
        if nrad is None:
            nrad = len(acqt_slice_idx)

        flag_posSet = False
        for idx, acq_idx in enumerate(acqt_slice_idx[:nrad]):
            acq = acquisitions[cidx[acq_idx]]
            if not flag_posSet:
                pos = acq.position
                sGeometry.pos_slices[idx_slice] = torch.Tensor([pos[0], pos[1], pos[2]])
                flag_posSet = True
                idx_slice += 1
            acq.resize(
                number_of_samples=acq.number_of_samples, active_channels=acq.active_channels, trajectory_dimensions=2
            )
            acq.traj[:] = grad_traj[idx, :, :]
            ds.append_acquisition(acq)
    ds.close()

    return full_filename_out


def writeTraj(filename_h5: str):
    sGeometry = SGeometry()
    fname_traj = add_2D_noncart_traj_to_ismrmrd(
        traj_rad_kirsten_t1mapping, filename_h5, filename_h5.replace('.h5', '_traj_2s.h5'), sGeometry=sGeometry
    )

    return [fname_traj, sGeometry]


def recoStack(path_h5: str) -> torch.Tensor:
    kdata = KData.from_file(path_h5, KTrajectoryIsmrmrd())

    # Calculate dcf
    kdcf = DcfData.from_traj_voronoi(kdata.traj)

    # Reconstruct average image for coil map estimation
    FOp = FourierOp(
        recon_shape=kdata.header.recon_matrix,
        encoding_shape=kdata.header.encoding_matrix,
        traj=kdata.traj,
    )
    (im,) = FOp.adjoint(kdata.data * kdcf.data[:, None, ...])

    # Calculate coilmaps
    idat = IData.from_tensor_and_kheader(im, kdata.header)
    csm = CsmData.from_idata_walsh(idat)
    csm_op = SensitivityOp(csm)

    # Coil combination
    (im,) = csm_op.adjoint(im)

    return im


def calcDsize_HR(dsize_LR: torch.Size, voxelSize_HR: torch.Tensor, voxelSize_LR: torch.Tensor) -> torch.Tensor:
    return torch.round(
        torch.Tensor(
            [
                dsize_LR[0] * voxelSize_LR[0] / voxelSize_HR[0],
                dsize_LR[1] * voxelSize_LR[1] / voxelSize_HR[1],
                dsize_LR[1] * voxelSize_LR[1] / voxelSize_HR[2],
            ]
        ).to(torch.int32)
    )


# Read raw data and trajectory
path_folder_inVivo = '/../../echo/allgemein/projects/8_13/MRPro/example_data/raw_data/SuperRes_inVivo/'
path_folder_phantom = '/../../echo/allgemein/projects/8_13/MRPro/example_data/raw_data/SuperRes_Phantom/'
path_folder = path_folder_inVivo
scanInfo = ScanInfo(path_folder + 'scanInfo')
pathes_orig = scanInfo.pathes_h5
num_stacks = 12  # len(scanInfo.pathes_h5)
list_sGeometries = []
list_imgs_stacks = []
path_save = path_folder + 'results/'
check_if_path_exists(path_save)

flag_useStored = True
for idx_stack in range(num_stacks):  #
    if not flag_useStored:
        [path_new, sGeometry] = writeTraj(path_folder + scanInfo.pathes_h5[idx_stack])
        img_stack = recoStack(path_new)
        if 'Phantom' in path_folder:
            path_maps = '/../../echo/allgemein/projects/hufnag01/forMrPro/maps/stack'
            list_im = []
            for idx_slice in range(scanInfo.num_slices):
                loaded = np.load(path_maps + str(idx_stack) + '/slice' + str(idx_slice) + '.npy')[:, :, 2]
                loaded = np.swapaxes(loaded, 0, 1)
                list_im.append(torch.from_numpy(loaded))
                show_2D(torch.abs(list_im[idx_slice]), 'map')
                show_2D(torch.abs(img_stack[idx_slice, 0, 0]) * 10000.0, 'reco')
            img_stack = torch.stack(list_im)
            img_stack = img_stack[:, None, None]
            plt.show()
        save_object(filename=path_save + 'sGeometry_' + str(idx_stack), obj=sGeometry)
        np.save(path_save + 'img_stack_' + str(idx_stack), arr=img_stack)
    else:
        sGeometry = load_object(path_save + 'sGeometry_' + str(idx_stack))
        img_stack = np.load(path_save + 'img_stack_' + str(idx_stack) + '.npy')

    img_stack = torch.from_numpy(img_stack[:, 0, 0])
    # img_stack = img_stack[:, 0, 0]

    cutoff = 71
    img_stack = img_stack[:, cutoff : 240 - cutoff, cutoff : 240 - cutoff]
    list_imgs_stacks.append(torch.abs(img_stack))
    sGeometry.dir.makeSureExists()
    list_sGeometries.append(sGeometry)

    # showImgsOfStack(img_stack, idx_stack = idx_stack, num_slices=img_stack.shape[0])

imgs_stacks = torch.stack(list_imgs_stacks)

voxelSize_HR = torch.Tensor([1.3, 1.3, 1.3])
voxelSize_LR = torch.Tensor([1.3, 1.3, scanInfo.thickness_slice])
shape_HR = calcDsize_HR(dsize_LR=imgs_stacks.shape[2:4], voxelSize_HR=voxelSize_HR, voxelSize_LR=voxelSize_LR)
slice_profile = Slice_profile(thickness_slice=scanInfo.thickness_slice)

resortSlices(imgs=imgs_stacks, list_sGeometries=list_sGeometries)
rots = calc_rotToStack0(list_sGeometries=list_sGeometries, img_stacks=imgs_stacks)
distTo0_mm = calcDistToStack0(rots=rots[:, 0], list_sGeometry=list_sGeometries)
gap_slices_mm = calcGapBetweenLR(list_sGeometries=list_sGeometries, thickness_slice=scanInfo.thickness_slice)

rots_before = torch.clone(rots)
rots = torch.zeros_like(rots_before)
rots[:, 0] = 90
rots[:, 2] = rots_before[:, 0]

srr_op = SuperResOp(
    shape_HR=shape_HR,
    num_slices_per_stack=scanInfo.num_slices,
    gap_slices_inHR=gap_slices_mm / voxelSize_HR[2],
    num_stacks=len(list_sGeometries),
    thickness_slice_inHR=scanInfo.thickness_slice / voxelSize_HR[2],
    offsets_stack_HR=distTo0_mm / voxelSize_HR[2],
    rot_per_stack=rots,
    w=3 * slice_profile.sigma,
    slice_profile=slice_profile.rect,
)


flag_showLRstacks = False
if flag_showLRstacks:
    for idx_stack in range(num_stacks):
        for idx_slice in range(1, 4):  # ,scanInfo.num_slices):
            show_2D(
                imgs_stacks[idx_stack][idx_slice],
                'stack' + str(idx_stack) + '_slice' + str(idx_slice),
                vmin=0,
                vmax=0.0003,
            )

vol_HR = srr_op.adjoint(imgs_stacks[:, :, None, None, None])[0]

shape_vol_HR_adjoint = vol_HR.shape
for idx_proj in range(3):
    show_3D(vol_HR[0, 0], axs_proj=idx_proj, title='vol_HR_adjoint_0', vmin=0, vmax=0.00017)

flag_loopOverSlices = False
if flag_loopOverSlices:
    for slice_1 in range(int(0.4 * shape_vol_HR_adjoint[3]), int(0.6 * shape_vol_HR_adjoint[3])):
        show_3D(
            vol_HR[0, 0],
            axs_proj=0,
            idx_slice=slice_1,
            title='vol_HR_adjoint_1_slice' + str(slice_1),
            vmin=0,
            vmax=0.00002 * len(list_sGeometries),
        )
