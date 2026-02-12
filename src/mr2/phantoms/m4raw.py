"""FastMRI dataset."""

import re
from collections import defaultdict
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from warnings import warn
from xml.etree import ElementTree as ET

import h5py
import numpy as np
import torch
from einops import rearrange

from mr2.data.AcqInfo import AcqInfo
from mr2.data.enums import TrajectoryType
from mr2.data.KData import KData
from mr2.data.KHeader import KHeader
from mr2.data.SpatialDimension import SpatialDimension
from mr2.data.traj_calculators.KTrajectoryCartesian import KTrajectoryCartesian
from mr2.utils.reshape import unsqueeze_left
from mr2.utils.unit_conversion import deg_to_rad, magnetic_field_to_lamor_frequency, ms_to_s


class M4RawDataset(torch.utils.data.Dataset):
    """M4Raw KData Dataset.

    M4Raw [M4RAW]_ low-field (0.3T) brain dataset. The data can be downloaded from zenodo [M4RAW_DATA]_.
    The data consists of Cartsian GRE, T1w, T2w, and FLAIR images with multiple repetitions.
    It will be returned as `mr2.data.KData` objects of either a single slice or a stack of
    slices with multiple repetitions.

    References
    ----------
    .. [M4RAW_DATA] Lyu, M., Mei, L., Huang, S., et al.(2023). M4Raw: A multi-contrast, multi-repetition,
       multi-channel MRI k-space dataset for low-field MRI research [Data set] https://doi.org/10.5281/zenodo.8056074
    .. [M4RAW] Lyu, M., Mei, L., Huang, S. et al. M4Raw: A multi-contrast, multi-repetition, multi-channel MRI k-space
       dataset for low-field MRI research. Sci Data 10, 264 (2023). https://doi.org/10.1038/s41597-023-02181-4
    """

    def __init__(self, path: PathLike | str | Sequence[PathLike | str], single_slice: bool = False):
        """Initialize the dataset.

        Parameters
        ----------
        path : PathLike
            Either a path to a directory containing the M4Raw data as .h5 files or a sequence of paths of
            individual files.
        single_slice : bool
            Whether to return single slices or stacks of slices.
        """
        pattern = re.compile(r'(\d+_(?:FLAIR|T1|T2|GRE))(\d+)')
        grouped: dict[str, list] = defaultdict(list)
        for fn in Path(path).rglob('*.h5') if isinstance(path, str | Path | PathLike) else [Path(p) for p in path]:
            m = pattern.fullmatch(fn.stem)
            if m is None:
                warn(f'Ignoring file {fn} because it does not match pattern.', stacklevel=2)
                continue
            grouped[m[1]].append(fn)
        self._filenames = list(grouped.values())
        self.single_slice = single_slice

    def __len__(self) -> int:
        """Get length (number of slices or stacks of slices) of the dataset."""
        if self.single_slice:
            return len(self._filenames) * 18
        else:
            return len(self._filenames)

    def __getitem__(self, idx: int) -> KData:
        """Get a single slice or stack of slices."""
        if not -len(self) <= idx < len(self):
            raise IndexError(f'Index {idx} is out of bounds for the dataset of size {len(self)}')
        if idx < 0:
            idx += len(self)

        if self.single_slice:
            slice_idx: int | slice
            file_idx, slice_idx = divmod(idx, 18)
        else:
            file_idx = idx
            slice_idx = slice(None)

        with h5py.File(self._filenames[file_idx][0], 'r') as file:
            xml_root = ET.fromstring(file['ismrmrd_header'][()].decode('utf-8'))  # noqa: S314

        info = AcqInfo()
        info.idx.k1 = torch.arange(30, 225)[None, None, None, :, None]
        # The dataset includes a header with an ISMRMRD namespace. But the XML is not a valid ISMRMRD header.
        # So we need to parse it manually.
        recon_matrix = encoding_matrix = SpatialDimension(1, 256, 256)  # Fix for all files
        recon_fov = encoding_fov = SpatialDimension(0.005, 0.24, 0.24)  # Fix for all files

        def get(name: str) -> list[str]:
            return [
                e.text for e in xml_root.findall(name, {'ns0': 'http://www.ismrm.org/ISMRMRD'}) if e.text is not None
            ]

        te = [ms_to_s(float(e)) for e in get('ns0:sequenceParameters/ns0:TE')]
        ti = [ms_to_s(float(e)) for e in get('ns0:sequenceParameters/ns0:TI')]
        fa = [deg_to_rad(float(e)) for e in get('ns0:sequenceParameters/ns0:flipAngle_deg')]
        tr = [ms_to_s(float(e)) for e in get('ns0:sequenceParameters/ns0:TR')]
        echo_spacing = [
            ms_to_s(float(e.replace('ms', ''))) for e in get('ns0:sequenceParameters/ns0:echo_spacing') if e != 'N/A'
        ]
        echo_train_length = int(get('ns0:encoding/ns0:echoTrainLength')[0])
        sequence_type = get('ns0:sequenceParameters/ns0:sequence_type')[0]
        model = get('ns0:acquisitionSystemInformation/ns0:systemModel')[0]
        vendor = get('ns0:acquisitionSystemInformation/ns0:systemVendor')[0]
        protocol_name = get('ns0:measurementInformation/ns0:protocolName')[0]
        measurement_id = get('ns0:measurementInformation/ns0:measurementID')[0]

        header = KHeader(
            recon_matrix=recon_matrix,
            encoding_matrix=encoding_matrix,
            recon_fov=recon_fov,
            encoding_fov=encoding_fov,
            te=te,
            ti=ti,
            fa=fa,
            tr=tr,
            echo_spacing=echo_spacing,
            echo_train_length=echo_train_length,
            sequence_type=sequence_type,
            model=model,
            vendor=vendor,
            protocol_name=protocol_name,
            measurement_id=measurement_id,
            trajectory_type=TrajectoryType.CARTESIAN,
            acq_info=info,
            lamor_frequency_proton=int(magnetic_field_to_lamor_frequency(0.3)),
        )

        reps = []
        for filename in self._filenames[file_idx]:
            with h5py.File(filename, 'r') as file:
                reps.append(torch.as_tensor(np.array(file['kspace'][slice_idx, ..., 30:225])))
        data = rearrange(torch.stack(reps), 'reps ... coils k0 k1 -> ... reps coils 1 k1 k0')
        info = AcqInfo()
        info.idx.k1 = unsqueeze_left(torch.arange(30, 225)[:, None], data.ndim - 2)
        info.idx.repetition = unsqueeze_left(torch.arange(data.shape[-5])[:, None, None, None, None], data.ndim - 5)

        traj = KTrajectoryCartesian()(
            n_k0=256,
            k0_center=128,
            k1_idx=info.idx.k1,
            k1_center=128,
            k2_idx=torch.tensor(0),
            k2_center=0,
        )
        return KData(data=data, header=header, traj=traj)
