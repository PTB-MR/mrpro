import concurrent.futures
import io
import re
import urllib.request
from os import PathLike
from pathlib import Path

import numpy as np
import platformdirs
import scipy.io
import torch
from einops import rearrange
from torch.utils.data import Dataset
from tqdm import tqdm

from mrpro.data.AcqInfo import AcqInfo
from mrpro.data.enums import TrajectoryType
from mrpro.data.KData import KData
from mrpro.data.KHeader import KHeader
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.data.traj_calculators.KTrajectoryRadial2D import KTrajectoryRadial2D

CACHE_DIR = platformdirs.user_cache_dir('mrpro')  #  ~/.cache/mrpro on Linux, %AppData%\Local\mrpro on Windows


def download_mdcnn(output_directory: str | PathLike = CACHE_DIR / 'mdcnn', n_files: int = 108) -> None:
    """
    Download MD-CNN [MDCNN]_ dataset from Harvard Dataverse.

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

    """
    progress = tqdm(total=n_files, desc='Downloading')
    if n_files > 108:
        raise ValueError('n_files must be less than 108')
    if n_files < 0:
        raise ValueError('n_files must be positive')
    try:
        output_directory_ = Path(output_directory)
        output_directory_.mkdir(parents=True, exist_ok=True)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f'Failed to create directory {output_directory}: {e}') from None

    def fetch(file_id: int) -> None:
        url = f'https://dataverse.harvard.edu/api/access/datafile/{file_id}'
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'mrpro'})) as resp:  # noqa: S310
            if (match := re.search('(P[0-9]+)_', resp.headers['Content-Disposition'])) is None:
                return
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
            filename = output_directory_ / f'{subject}.npy'
            np.save(filename, data)
            progress.update(1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for file_id in range(4000844, 4000978):
            executor.submit(fetch, file_id)


class MDCNNDataset(Dataset):
    """MD-CNN radial cardiac cine dataset.

     The MD-CNN [MDCNN]_ dataset is a radial cardiac cine dataset with 108 subjects.
     See `mrpro.phantoms.mdcnn.download_mdcnn` for downloading and conversion to numpy format of the dataset.

     The dataset returns `mrpro.data.KData` objects.

    References
    ----------
    .. [MDCNN] El-Rewaidy H., Replication Data for: Multi-Domain Convolutional Neural Network (MD-CNN) For Radial
       Reconstruction of Dynamic Cardiac MRI. Harvard Dataverse, 2020. https://doi.org/10.7910/DVN/CI3WB6.
    .. [REWAIDY] El-Rewaidy H, Fahmy AS, Pashakhanloo F, et al. Multi-domain convolutional neural network (MD-CNN) for
       radial reconstruction of dynamic cardiac MRI. Magn Reson Med. 2020; 85: 1195–1208. https://doi.org/10.1002/mrm.28485
    """

    def __init__(self, path: str | PathLike):
        """Initialize the MD-CNN dataset.

        Parameters
        ----------
        path : str | PathLike
            Path to the directory containing the MD-CNN dataset converted to numpy format.
        """
        files = list(Path(path).glob('**/*.npy'))
        self.files = sorted(files, key=lambda x: int(x.stem[1:]))

    def __len__(self):
        """Get the number of CINES in the dataset."""
        return len(self.files)

    def __getitem__(self, idx):
        """Get one CINE from the dataset."""
        filename = self.files[idx]
        data = torch.as_tensor(np.load(filename))
        n_k0 = data.shape[-1]
        if n_k0 == 416:
            k0_center = 417 - 208
        else:
            k0_center = 417 - 296
        n_k1 = data.shape[-2]
        k1_idx = torch.arange(n_k1)[None, None, None, :, None] + n_k1 // 2
        traj = KTrajectoryRadial2D(torch.pi / n_k1)(
            n_k0=n_k0,
            k0_center=k0_center,
            k1_idx=k1_idx,
        )
        info = AcqInfo()
        info.idx.k1 = k1_idx
        header = KHeader(
            encoding_matrix=SpatialDimension(1, 623, 623),
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
