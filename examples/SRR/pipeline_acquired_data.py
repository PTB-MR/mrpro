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

from collections.abc import Callable

import ismrmrd
import numpy as np
import torch
from utils import ScanInfo
from utils import angle_between_vec
from utils import check_if_path_exists
from utils import dist_3D
from utils import load_object
from utils import normed_vec
from utils import save_object
from utils import show_2D
from utils import show_3D
from utils import spa_dim_to_tensor

from mrpro.data import CsmData
from mrpro.data import DcfData
from mrpro.data import IData
from mrpro.data import KData
from mrpro.data.traj_calculators import KTrajectoryIsmrmrd
from mrpro.data.traj_calculators import KTrajectoryRadial2D
from mrpro.operators import FourierOp
from mrpro.operators import SensitivityOp
from mrpro.operators._SuperResOp import Slice_prof
from mrpro.operators._SuperResOp import SuperResOp


def show_imgs_of_stack(im: torch.Tensor, idx_stack: int) -> None:
    # Visualize results
    num_slices = im.shape[0]
    for idx_slice in range(num_slices):
        show_2D(
            torch.abs(im[idx_slice]),
            title='stack ' + str(idx_stack) + '_slice' + str(idx_slice),
        )


def calc_dist_to_stack0(list_sGeometry: list[SGeometry], rots: torch.Tensor) -> torch.Tensor:
    num_slices_per_stack = list_sGeometry[0].pos_slices.shape[0]
    array_dist = torch.zeros(len(rots))

    # loop over all unique rotations
    for rot_unique in list(torch.unique(rots)):
        idx_with_same_rot = list(torch.where(rots == rot_unique))[0]
        if list(rots).count(rot_unique) == 1:
            array_dist[idx_with_same_rot[0]] = 0.0
        else:
            idx_slice_to_compare = 2
            pos_stack_group_0 = list_sGeometry[idx_with_same_rot[0]].pos_slices[idx_slice_to_compare]
            dist_within_rot_group = torch.zeros(len(idx_with_same_rot))
            # check if all slice dirs within grouop are the same
            dir_slice_of_group = torch.zeros([len(idx_with_same_rot), 3])
            for idx_in_group in range(0, len(idx_with_same_rot)):
                dir_slice = (
                    list_sGeometry[idx_with_same_rot[idx_in_group]].pos_slices[num_slices_per_stack - 1]
                    - list_sGeometry[idx_with_same_rot[idx_in_group]].pos_slices[0]
                )
                dir_slice_of_group[idx_in_group] = normed_vec(dir_slice)

            for idx_in_group in range(0, len(idx_with_same_rot)):
                pos_stack = torch.tensor(
                    list_sGeometry[idx_with_same_rot[idx_in_group]].pos_slices[idx_slice_to_compare]
                )
                dir_stack = pos_stack - pos_stack_group_0
                dist_within_rot_group[idx_in_group] = dist_3D(
                    pos_stack_group_0, list_sGeometry[idx_with_same_rot[idx_in_group]].pos_slices[idx_slice_to_compare]
                )

                if torch.dot(normed_vec(dir_stack), normed_vec(dir_slice_of_group[idx_in_group])) < 0:
                    dist_within_rot_group[idx_in_group] *= -1.0

            dist_within_rot_group -= torch.mean(dist_within_rot_group)
            for idx_in_group in range(len(idx_with_same_rot)):
                array_dist[idx_with_same_rot[idx_in_group]] = dist_within_rot_group[idx_in_group]
    return array_dist


def calc_gap_between_LRslices(list_sGeometries: list[SGeometry], thickness_slice: float) -> float:
    num_stacks = len(list_sGeometries)
    num_slices_per_stack = list_sGeometries[0].pos_slices.shape[0]
    if num_slices_per_stack <= 1:
        return 0
    else:
        gap = torch.zeros([num_stacks, num_slices_per_stack - 1])
        for idx_stack in range(num_stacks):
            for idx_slice in range(num_slices_per_stack - 1):
                dist = dist_3D(
                    list_sGeometries[idx_stack].pos_slices[idx_slice],
                    list_sGeometries[idx_stack].pos_slices[idx_slice + 1],
                )
                gap[idx_stack, idx_slice] = round(dist - thickness_slice, 2)
    return float(torch.median(gap))


def calc_rot_to_stack_0(list_sGeometries: list[SGeometry], img_stacks: torch.Tensor) -> torch.Tensor:
    num_stacks = len(list_sGeometries)

    # search for sign change in angles
    dir_slice_stack0 = list_sGeometries[0].dir.slice
    dir_phase_stack0 = list_sGeometries[0].dir.phase

    # check if read and phase dir need to be swapped
    for idx_stack in range(1, num_stacks):
        dir_phase_stack = list_sGeometries[idx_stack].dir.phase
        angle_phase = round(number=angle_between_vec(vector_A=dir_phase_stack0, vector_B=dir_phase_stack), ndigits=0)

        if abs(angle_phase - 90.0) < 10.0:
            dir_read_old = torch.clone(list_sGeometries[idx_stack].dir.read)
            list_sGeometries[idx_stack].dir.read = list_sGeometries[idx_stack].dir.phase
            list_sGeometries[idx_stack].dir.phase = dir_read_old
            img_stacks[idx_stack] = torch.swapaxes(img_stacks[idx_stack], 1, 2)
            print('... swap read and phase dir ')

    rots_toStack0 = torch.zeros([num_stacks, 3])

    for idx_stack in range(num_stacks):
        angle_slice = round(
            angle_between_vec(
                dir_slice_stack0,
                list_sGeometries[idx_stack].dir.slice,
                normal=list_sGeometries[idx_stack].dir.phase,
            ),
            0,
        )
        angle_phase = round(
            angle_between_vec(vector_A=dir_phase_stack0, vector_B=list_sGeometries[idx_stack].dir.phase), 1
        )

        rot = torch.tensor([0, 0, angle_slice])
        rots_toStack0[idx_stack] = rot
    return rots_toStack0


def get_slice_order(num_slices: int) -> list[int]:
    if num_slices == 6:
        order_slices = [3, 0, 4, 1, 5, 2]
    elif num_slices == 5:
        order_slices = [0, 3, 1, 4, 2]
    elif num_slices == 12:
        order_slices = [6, 0, 7, 1, 8, 2, 9, 3, 10, 4, 11, 5]
    else:
        raise Exception
    return order_slices


def resort_slices(imgs: torch.Tensor, list_sGeometries: list[SGeometry]) -> None:
    num_slices = imgs.shape[1]
    order_slices = get_slice_order(num_slices=num_slices)

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


# Add 2D non-Cartesian trajectory to ISMRMD data file
def add_2D_noncart_traj_to_ismrmrd(
    fun_calc_traj: Callable[[int, int], np.ndarray],
    full_filename_in: str,
    full_filename_out: str,
    sGeometry: SGeometry,
    nrad: int | None = None,
) -> None:

    # Get info and acquisitions from original data
    data_orig = KData.from_file(full_filename_in, KTrajectoryRadial2D())
    sGeometry.dir.read = spa_dim_to_tensor(data_orig.header.acq_info.read_dir)
    sGeometry.dir.slice = spa_dim_to_tensor(data_orig.header.acq_info.slice_dir)
    sGeometry.dir.phase = spa_dim_to_tensor(data_orig.header.acq_info.phase_dir)

    pos_slices_x = data_orig.header.acq_info.position.x[:, 0, 0, 0]
    pos_slices_y = data_orig.header.acq_info.position.y[:, 0, 0, 0]
    pos_slices_z = data_orig.header.acq_info.position.z[:, 0, 0, 0]
    pos_slices = torch.stack([pos_slices_x, pos_slices_y, pos_slices_z])
    sGeometry.pos_slices = torch.swapaxes(pos_slices, 0, 1)

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

        for idx, acq_idx in enumerate(acqt_slice_idx[:nrad]):
            acq = acquisitions[cidx[acq_idx]]

            acq.resize(
                number_of_samples=acq.number_of_samples, active_channels=acq.active_channels, trajectory_dimensions=2
            )
            acq.traj[:] = grad_traj[idx, :, :]
            ds.append_acquisition(acq)
    ds.close()


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


def calc_dsize_HR(dsize_LR: torch.Size, voxel_size_HR: torch.Tensor, voxel_size_LR: torch.Tensor) -> torch.Tensor:
    return torch.round(
        torch.Tensor(
            [
                dsize_LR[0] * voxel_size_LR[0] / voxel_size_HR[0],
                dsize_LR[1] * voxel_size_LR[1] / voxel_size_HR[1],
                dsize_LR[1] * voxel_size_LR[1] / voxel_size_HR[2],
            ]
        ).to(torch.int32)
    )


path_invivo_rot = '/../../echo/allgemein/projects/8_13/MRPro/example_data/raw_data/SuperRes/inVivo_rot/'
path_invivo_transl = '/../../echo/allgemein/projects/8_13/MRPro/example_data/raw_data/SuperRes/inVivo_transl/'
path_phantom_rot = '/../../echo/allgemein/projects/8_13/MRPro/example_data/raw_data/SuperRes/Phantom_rot/'
path_phantom_rot_T1 = '/../../echo/allgemein/projects/8_13/MRPro/example_data/raw_data/SuperRes/Phantom_rot_T1/'
path_phantom_transl = '/../../echo/allgemein/projects/8_13/MRPro/example_data/raw_data/SuperRes/Phantom_transl/'

path_folder = path_phantom_transl

# read through scanInfo of dataset and gather information as e.g. the h5 file pathes
scan_info = ScanInfo(path_folder + 'scanInfo')

num_stacks = len(scan_info.pathes_h5)
list_sGeometries = []
list_imgs_stacks = []
path_save = path_folder + 'reconstructed_files/'
check_if_path_exists(path_save)

flag_use_stored_recon = False
for idx_stack in range(num_stacks):
    if not flag_use_stored_recon:
        sGeometry = SGeometry()
        filename_h5 = path_folder + scan_info.pathes_h5[idx_stack]
        filename_h5_new = filename_h5.replace('.h5', '_traj_2s.h5')

        # modify tratectory
        add_2D_noncart_traj_to_ismrmrd(traj_rad_kirsten_t1mapping, filename_h5, filename_h5_new, sGeometry=sGeometry)
        # save gathered information about geometry of stacks
        save_object(filename=path_save + 'sGeometry_' + str(idx_stack), obj=sGeometry)

        # if t1 maps have been reconstructed externally, use these as data_LR_loaded
        if path_folder == path_phantom_rot_T1:
            list_im = []
            for idx_slice in range(scan_info.num_slices):
                loaded = np.load(
                    path_phantom_rot_T1 + 'maps/stack' + str(idx_stack) + '/slice' + str(idx_slice) + '.npy'
                )[:, :, 2]
                loaded = np.swapaxes(loaded, 0, 1)
                list_im.append(loaded)
            data_LR_loaded = np.array(list_im)
            data_LR_loaded = data_LR_loaded[:, np.newaxis, np.newaxis]
            np.save(path_save + 'img_stack_' + str(idx_stack), arr=data_LR_loaded)

        # if not, reconstruct data_LR_loaded
        else:
            data_LR_loaded = recoStack(filename_h5_new).numpy()
            np.save(path_save + 'img_stack_' + str(idx_stack), arr=data_LR_loaded)
    else:
        sGeometry = load_object(path_save + 'sGeometry_' + str(idx_stack))
        data_LR_loaded = np.load(path_save + 'img_stack_' + str(idx_stack) + '.npy')

    data_LR = torch.from_numpy(data_LR_loaded[:, 0, 0])

    # crop data_LR to center to accelerate computation
    cutoff = 71
    data_LR = data_LR[:, cutoff : 240 - cutoff, cutoff : 240 - cutoff]

    # combine information of all stacks
    list_imgs_stacks.append(torch.abs(data_LR))
    sGeometry.dir.makeSureExists()
    list_sGeometries.append(sGeometry)

    # showImgsOfStack(img_stack, idx_stack = idx_stack, num_slices=img_stack.shape[0])

imgs_stacks = torch.stack(list_imgs_stacks)

voxel_size_HR = torch.Tensor([1.3, 1.3, 1.3])
voxel_size_LR = torch.Tensor([1.3, 1.3, scan_info.thickness_slice])
shape_HR = calc_dsize_HR(dsize_LR=imgs_stacks.shape[2:4], voxel_size_HR=voxel_size_HR, voxel_size_LR=voxel_size_LR)
slice_prof = Slice_prof(thickness_slice=scan_info.thickness_slice)

# resort slices due to their interleaved acquisition scheme
resort_slices(imgs=imgs_stacks, list_sGeometries=list_sGeometries)

# calc rotation of all stacks with respect to stack 0
rots = calc_rot_to_stack_0(list_sGeometries=list_sGeometries, img_stacks=imgs_stacks)

# calc distance of all stacks in mm with respect to stack 0
dist_To_0_mm = calc_dist_to_stack0(rots=rots[:, 2], list_sGeometry=list_sGeometries)

# calc distance between slices within a single stack in mm
gap_slices_mm = calc_gap_between_LRslices(list_sGeometries=list_sGeometries, thickness_slice=scan_info.thickness_slice)

# set rotation to 90 degree in 0 dim to rotate around phase encoding direction (depends on the acquired data)
rots[:, 0] = 90

# pass calculcated data into super res op
srr_op = SuperResOp(
    shape_HR=shape_HR,
    num_slices_per_stack=scan_info.num_slices,
    gap_slices_inHR=float(gap_slices_mm / voxel_size_HR[2]),
    num_stacks=len(list_sGeometries),
    thickness_slice_inHR=scan_info.thickness_slice / voxel_size_HR[2],
    offsets_stack_HR=dist_To_0_mm / voxel_size_HR[2],
    rot_per_stack=rots,
    w=3 * slice_prof.sigma,
    slice_prof=slice_prof.rect,
)


flag_show_LR_stacks = False
if flag_show_LR_stacks:
    for idx_stack in range(num_stacks):
        for idx_slice in range(1, 4):  # ,scanInfo.num_slices):
            show_2D(
                imgs_stacks[idx_stack][idx_slice],
                'stack' + str(idx_stack) + '_slice' + str(idx_slice),
                vmin=0,
                vmax=0.0003,
                flag_T1map=path_folder == path_phantom_rot_T1,
            )

vol_HR = srr_op.adjoint(imgs_stacks[:, :, None, None, None])[0]
vol_HR = torch.nan_to_num(vol_HR, nan=0.0)
shape_vol_HR_adjoint = vol_HR.shape
for idx_proj in range(3):
    show_3D(
        vol_HR[0, 0],
        axs_proj=idx_proj,
        title='vol_HR_adjoint_0',
        vmin=0,
        vmax=0.00017,
        flag_T1map=path_folder == path_phantom_rot_T1,
    )

flag_loop_over_slices = False
if flag_loop_over_slices:
    for slice in range(int(0.4 * shape_vol_HR_adjoint[3]), int(0.6 * shape_vol_HR_adjoint[3])):
        show_3D(
            vol_HR[0, 0],
            axs_proj=0,
            idx_slice=slice,
            title='vol_HR_adjoint_1_slice' + str(slice),
            vmin=0,
            vmax=0.00002 * len(list_sGeometries),
            flag_T1map=path_folder == path_phantom_rot_T1,
        )
