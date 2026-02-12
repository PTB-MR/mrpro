"""MD-CNN radial cardiac cine dataset."""

import concurrent.futures
import io
import re
import urllib.request
from collections.abc import Sequence
from os import PathLike
from pathlib import Path

import numpy as np
import platformdirs
import scipy.io
import torch
from einops import rearrange
from torch.utils.data import Dataset
from tqdm import tqdm

from mr2.data.AcqInfo import AcqInfo
from mr2.data.enums import TrajectoryType
from mr2.data.KData import KData
from mr2.data.KHeader import KHeader
from mr2.data.SpatialDimension import SpatialDimension
from mr2.data.traj_calculators.KTrajectoryRadial2D import KTrajectoryRadial2D

# ~/.cache/mr2/mdcnn on Linux, %AppData%\Local\mr2\mdcnn on Windows
CACHE_DIR_MDCNN = Path(platformdirs.user_cache_dir('mr2')) / 'mdcnn'


def download_mdcnn(
    output_directory: str | PathLike = CACHE_DIR_MDCNN,
    n_files: int = 108,
    workers: int = 4,
    progress: bool = False,
) -> None:
    """
    Download MD-CNN [MDCNN]_ dataset from Harvard Dataverse and save it as numpy files.

    References
    ----------
    .. [MDCNN] H. El-Rewaidy, “Replication Data for: Multi-Domain Convolutional Neural Network (MD-CNN) For Radial
       Reconstruction of Dynamic Cardiac MRI.” Harvard Dataverse, 2020. https://doi.org/10.7910/DVN/CI3WB6.

    Parameters
    ----------
    output_directory : Path
        Directory to save files.
    n_files : int
        Number of files to download. Maximum is 108.
        If lower than 108, the first `n_files` based on the harvard dataverse id will be downloaded.
    workers : int
        Number of parallel downloads.
    progress : bool
        Show progress bar.
    """
    progressbar = tqdm(total=n_files, desc='Downloading') if progress else None
    if not 0 < n_files <= 108:
        raise ValueError('n_files must be positive and less than 108')
    try:
        output_directory_ = Path(output_directory)
        output_directory_.mkdir(parents=True, exist_ok=True)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f'Failed to create directory {output_directory}: {e}') from None

    def fetch(file_id: int) -> None:
        nonlocal n_files
        url = f'https://dataverse.harvard.edu/api/access/datafile/{file_id}'
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'mr2'})) as resp:  # noqa: S310
            if (match := re.search('(P[0-9]+)_', resp.headers['Content-Disposition'])) is None:
                return
            if n_files < 1:
                return
            n_files -= 1
            subject = match.groups()[0]
            data = torch.as_tensor(scipy.io.loadmat(io.BytesIO(resp.read()))['data'], dtype=torch.float32)
            data = torch.view_as_complex(
                rearrange(data, '1 phase k0 k1 coils complex -> phase coils 1 k1 k0 complex').contiguous()
            )
            if data[0, 0, 0, 0, 208].abs() > 1e-12:
                first = 208
            else:
                first = 296
            data = data[..., first:624]
            # we save as numpy, as loadmat is more than 10x slower than np.load
            filename = output_directory_ / f'{subject}.npy'
            np.save(filename, data)
            if progressbar is not None:
                progressbar.update(1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for file_id in range(4000844, 4000978):
            executor.submit(fetch, file_id)


class MDCNNDataset(Dataset):
    """MD-CNN radial cardiac cine dataset.

     The MD-CNN [MDCNN]_ dataset is a radial cardiac cine dataset with 108 subjects.
     See `mr2.phantoms.mdcnn.download_mdcnn` for downloading and conversion to numpy format of the dataset.

     The dataset returns `mr2.data.KData` objects.

    References
    ----------
    .. [MDCNN] El-Rewaidy H., Replication Data for: Multi-Domain Convolutional Neural Network (MD-CNN) For Radial
       Reconstruction of Dynamic Cardiac MRI. Harvard Dataverse, 2020. https://doi.org/10.7910/DVN/CI3WB6.
    .. [REWAIDY] El-Rewaidy H, Fahmy AS, Pashakhanloo F, et al. Multi-domain convolutional neural network (MD-CNN) for
       radial reconstruction of dynamic cardiac MRI. Magn Reson Med. 2020; 85: 1195-1208. https://doi.org/10.1002/mrm.28485
    """

    def __init__(self, path: str | PathLike | Sequence[str | PathLike] = CACHE_DIR_MDCNN):
        """Initialize the MD-CNN dataset.

        Parameters
        ----------
        path : str | PathLike
            Path to the directory containing the MD-CNN dataset converted to numpy format or sequence of npy files.
        """
        files = list(Path(path).rglob('P*.npy')) if isinstance(path, str | PathLike) else [Path(p) for p in path]
        self.files = sorted(files, key=lambda x: int(x.stem[1:]))

    def __len__(self) -> int:
        """Get the number of CINES in the dataset."""
        return len(self.files)

    def __getitem__(self, idx: int) -> KData:
        """Get one CINE from the dataset."""
        filename = self.files[idx]
        data = torch.as_tensor(np.load(filename))
        n_phases, *_, n_k1, n_k0 = data.shape

        if n_k0 == 416:  # different partial fourier settings
            k0_center = 417 - 208
        else:
            k0_center = 417 - 296

        info = AcqInfo()
        info.idx.k1 = torch.arange(n_k1)[None, None, None, :, None] + n_k1 // 2
        info.idx.phase = torch.arange(n_phases)[:, None, None, None, None]
        traj = KTrajectoryRadial2D(torch.pi / n_k1)(
            n_k0=n_k0,
            k0_center=k0_center,
            k1_idx=info.idx.k1,
        )

        header = KHeader(
            encoding_matrix=SpatialDimension(1, 624, 624),
            recon_matrix=SpatialDimension(1, 208, 208),
            acq_info=info,
            recon_fov=SpatialDimension(1, 0.38, 0.38),
            encoding_fov=SpatialDimension(1, 1.52, 1.52),
            vendor='Siemens',
            model='MAGNETOM Vida',
            protocol_name='bSSFP Cine',
            tr=torch.tensor(3.06e-3)[None, None, None, None, None],
            te=torch.tensor(1.4e-3)[None, None, None, None, None],
            fa=torch.deg2rad(torch.tensor(48))[None, None, None, None, None],
            lamor_frequency_proton=123e6,
            trajectory_type=TrajectoryType.RADIAL,
            patient_name=filename.stem,
        )
        return KData(data=data, header=header, traj=traj)
