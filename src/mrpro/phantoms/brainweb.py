"""Brainweb Phantom."""

import concurrent.futures
import gzip
import hashlib
import io
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import platformdirs
import requests
import torch
import torchvision
from tqdm import tqdm
from typing_extensions import TypeVar

HASHES = {
    '04': '5da73dc2efe2fbd92ad1394400db3a2e',
    '05': 'e9f5d422e4ecb99af9be405ec21cfac8',
    '06': 'b6c0299e8f61f5f023c1d83c0342f5a0',
    '18': 'a23d9bce000a356eaf11726f5bc596be',
    '20': 'c6a9b7435a8d468a4ec8cbffef3f13a8',
    '38': 'bbac67921fce764abf619e15ff897e12',
    '41': 'e6374ab62f416e911c6b50bfdb3f1063',
    '42': 'ec0059117da02bd5374607944b4cb732',
    '43': '7a7f68b6c265f77dcbf42adde67697e4',
    '44': '7813c8f9ef5e6409bfcf321b04208346',
    '45': '68cf6c1f1cbfd94cbd503b74c417ea89',
    '46': '719f460c7db9a1fd9ad2dd14998083de',
    '47': 'c337253f85233a78313859e8871c5a22',
    '48': '60b8c46cb18cb8f400771e28f31dea9d',
    '49': 'c5961edd86de2a669020464ceaeab703',
    '50': '71a392559f425017dd3142a5bce08a45',
    '51': '86516195671a32df211af9c624777791',
    '52': '8f70d209f208ce7b4294acfdb3522c62',
    '53': '5bd8d025379772b87dd225b6bb3c63d5',
    '54': '1d241f9a1fcb7e4af5e9a3c76dcdbf2a',
}
OVERVIEW_URL = 'http://brainweb.bic.mni.mcgill.ca/brainweb/anatomic_normal_20.html'
URL_TEMPLATE = (
    'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?'
    'do_download_alias=subject{subject}_{c}'
    '&format_value=raw_short'
    '&zip_value=gnuzip'
    '&download_for_real=%5BStart+download%21%5D'
)


CLASSES = ['skl', 'gry', 'wht', 'csf', 'mrw', 'dura', 'fat', 'fat2', 'mus', 'm-s', 'ves']  # noqa: typos

CACHE_DIR = platformdirs.user_cache_dir('mrpro')
K = TypeVar('K')


def load_file(url: str, timeout: float = 60) -> bytes:
    """Load url content."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def unpack(data: bytes, dtype: np.typing.DTypeLike, shape: Sequence[int]) -> np.ndarray:
    """Unpack gzipped data."""
    return np.frombuffer(gzip.open(io.BytesIO(data)).read(), dtype=dtype).reshape(shape)


def norm_(values: list[np.ndarray]) -> np.ndarray:
    """Normalize values to sum to 1 and convert to uint8."""
    for i, x in enumerate(values):
        values[i] = np.clip(x - np.min(x[50], (0, 1)), 0, 4096)
    sum_values = sum(values)
    for i, x in enumerate(values):
        x = np.divide(x, sum_values, where=sum_values != 0)
        x[sum_values == 0] = i == 0
        x = (x * (2**8 - 1)).astype(np.uint8)
        values[i] = x
    return np.stack(values, -1)


def download_subject(subject: str, outfilename: Path, workers: int, progressbar: tqdm) -> None:
    """Download and process all class files for a single subject asynchronously."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(load_file, URL_TEMPLATE.format(subject=subject, c=c)): c for c in CLASSES}

        downloaded_data = {}
        for future in concurrent.futures.as_completed(futures):
            c = futures[future]
            downloaded_data[c] = future.result()
            progressbar.update(1)

    values = norm_([unpack(downloaded_data[c], shape=(362, 434, 362), dtype=np.uint16) for c in CLASSES])

    with h5py.File(outfilename, 'w') as f:
        f.create_dataset(
            'classes',
            values.shape,
            dtype=values.dtype,
            data=values,
            chunks=(4, 4, 4, values.shape[-1]),
            compression='lzf',
        )
        f.attrs['classnames'] = list(CLASSES)
        f.attrs['subject'] = int(subject)


def download(output_directory: str | PathLike, workers: int = 4, progress: bool = False) -> None:
    """Download Brainweb data with subjects in series and class files in parallel."""
    page = requests.get(OVERVIEW_URL, timeout=5)
    subjects = re.findall(r'option value=(\d*)>', page.text)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    totalsteps = len(subjects) * len(CLASSES)
    with tqdm(total=totalsteps, desc='Downloading Brainweb data', disable=not progress) as progressbar:
        for subject in subjects:
            outfilename = output_directory / f's{subject}.h5'
            if outfilename.exists():
                md5 = hashlib.file_digest(outfilename.open('rb'), 'md5').hexdigest()
                # if md5 == HASHES[subject]:
                #     progressbar.update(len(CLASSES))
                #     continue
            download_subject(subject, outfilename, workers, progressbar)
            md5 = hashlib.file_digest(outfilename.open('rb'), 'md5').hexdigest()
            HASHES[subject] = md5


@dataclass
class T1T2PD:
    """Container for Parameters of a single tissue."""

    t1_min: float
    t1_max: float
    t2_min: float
    t2_max: float
    m0_min: float
    m0_max: float
    phase_min: float = -0.01
    phase_max: float = 0.01

    @property
    def random_r1(self) -> torch.Tensor:
        """Get randomized r1 value."""
        return 1 / torch.empty(1).uniform_(self.t1_min, self.t1_max)

    @property
    def random_r2(self) -> torch.Tensor:
        """Get randomized r2 value."""
        return 1 / torch.empty(1).uniform_(self.t2_min, self.t2_max)

    @property
    def random_m0(self) -> torch.Tensor:
        """Get renadomized complex m0 value."""
        return torch.polar(
            1 / torch.empty(1).uniform_(self.m0_min, self.m0_max),
            torch.empty(1).uniform_(self.phase_min, self.phase_max),
        )


VALUES_3T = {
    'gry': T1T2PD(1.200, 2.000, 0.080, 0.120, 0.7, 1.0),
    'wht': T1T2PD(0.800, 1.500, 0.060, 0.100, 0.50, 0.9),  # noqa:typos
    'csf': T1T2PD(2.000, 4.000, 1.300, 2.000, 0.9, 1.0),
    'mrw': T1T2PD(0.400, 0.600, 0.060, 0.100, 0.7, 1.0),
    'dura': T1T2PD(2.000, 2.800, 0.200, 0.500, 0.9, 1.0),
    'fat': T1T2PD(0.300, 0.500, 0.060, 0.100, 0.9, 1.0),
    'fat2': T1T2PD(0.400, 0.600, 0.060, 0.100, 0.6, 0.9),
    'mus': T1T2PD(1.200, 1.500, 0.040, 0.060, 0.9, 1.0),
    'm-s': T1T2PD(0.500, 0.900, 0.300, 0.500, 0.9, 1),
    'ves': T1T2PD(1.700, 2.100, 0.200, 0.400, 0.8, 1),
}

DEFAULT_VALUES = {'r1': 0.0, 'pd': 1.0, 'r2': 1, 'mask': 0, 'classes': -1}


def trim_slices(mask: torch.Tensor) -> tuple[slice, slice]:
    """Get slices that remove outer masked out values."""
    mask = mask.any(dim=tuple(range(mask.ndim - 2)))
    row_mask, col_mask = mask.any(1), mask.any(0)
    row_min = int(torch.argmax(row_mask))
    row_max = int(mask.size(0) - torch.argmax(row_mask.flip(0)))
    col_min = int(torch.argmax(col_mask))
    col_max = int(mask.size(1) - torch.argmax(col_mask.flip(0)))
    return slice(row_min, row_max), slice(col_min, col_max)


def apply_transform(
    data: Mapping[K, torch.Tensor], transform: Callable[[torch.Tensor], torch.Tensor], mask: torch.Tensor
) -> tuple[dict[K, torch.Tensor], torch.Tensor]:
    """Apply a transformation."""
    x = torch.stack(list(data.values()), 0)
    x[:, ~mask] = torch.nan
    x = transform(x)
    data = dict(zip(data, x, strict=True))
    newmask = mask & ~x.isnan().any(0)
    return data, newmask


DEFAULT_TRANSFORMS_256 = (
    torchvision.transforms.RandomAffine(
        degrees=10,
        translate=(0.05, 0.05),
        scale=(0.7, 0.8),
        fill=0.0,
        shear=(0, 5, 0, 5),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    ),
    torchvision.transforms.CenterCrop((256, 256)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomHorizontalFlip(),
)


class BrainwebSlices(torch.utils.data.Dataset):
    """Dataset of 2D qMRI parameter slices based on Brainweb dataset."""

    def __init__(
        self,
        folder: str | Path,
        parameters: Mapping[str, T1T2PD] = VALUES_3T,
        cuts: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
        axis: int = 0,
        step: int = 1,
        transforms: Sequence[Callable[[torch.Tensor], torch.Tensor]] = DEFAULT_TRANSFORMS_256,
        what: Sequence[Literal['r1', 'r2', 'pd', 't1', 't2', 'mask', 'classes']] = ('pd', 'r1', 'r2'),
        mask_values: Mapping[str, float | None] = DEFAULT_VALUES,
    ) -> None:
        """Initialize Brainweb qMRI slice phantom."""
        self.parameters = parameters
        self._cuts = cuts
        self._axis = axis
        self.step = step
        files = []
        ns = [0]
        for fn in Path(folder).glob('s++.h5'):
            with h5py.File(fn) as f:
                ns.append(
                    (f['classes'].shape[self._axis]) - (self._cuts[self._axis * 2] + self._cuts[self._axis * 2 + 1])
                )
                files.append(fn)
        self._files = tuple(files)
        self._ns = np.cumsum(ns)
        self.transforms = torchvision.transforms.Compose(transforms)
        self.what = what
        self.mask_values = mask_values

    def __len__(self) -> int:
        """Get number of slices."""
        return self._ns[-1] // self.step

    def __getitem__(self, index: int) -> dict[Literal['r1', 'r2', 'pd', 't1', 't2', 'mask', 'classes'], torch.Tensor]:
        """Get a single slice."""
        if index * self.step >= self._ns[-1]:
            raise IndexError
        elif index < 0:
            index = self._ns[-1] + index * self.step
        else:
            index = index * self.step

        file_id = np.searchsorted(self._ns, index, 'right') - 1
        slice_id = index - self._ns[file_id] + self._cuts[self._axis * 2]

        with h5py.File(
            self._files[file_id],
        ) as file:
            where = [slice(self._cuts[2 * i], file['classes'].shape[i] - self._cuts[2 * i + 1]) for i in range(3)] + [
                slice(None)
            ]
            where[self._axis] = slice_id
            data = torch.as_tensor(np.array(file['classes'][tuple(where)], dtype=np.uint8))
            classnames = tuple(file.attrs['classnames'])
            mask = data.sum(-1) > 150
            slices = trim_slices(mask)
            mask = mask[..., slices]
            data = data[..., slices]

            result: dict[Literal['r1', 'r2', 'pd', 't1', 't2', 'mask', 'classes'], torch.Tensor] = {}
            for el in self.what:
                if el == 'r1':
                    values = torch.stack([self.parameters[k].random_r1 for k in classnames]) / 255
                    result[el] = torch.dot(data, values)
                elif el == 'r2':
                    values = torch.stack([self.parameters[k].random_r2 for k in classnames]) / 255
                    result[el] = torch.dot(data, values)
                elif el == 'pd':
                    values = torch.stack([self.parameters[k].random_m0 for k in classnames]) / 255
                    result[el] = torch.dot(data, values)
                elif el == 't1':
                    values = torch.stack([self.parameters[k].random_r1 for k in classnames]) / 255
                    result[el] = torch.dot(data, values).reciprocal()
                elif el == 't2':
                    values = torch.stack([self.parameters[k].random_r2 for k in classnames]) / 255
                    result[el] = torch.dot(data, values).reciprocal()
                elif el == 'classes':
                    result[el] = data.argmax(-1)
                else:
                    raise NotImplementedError(f'what=({el},) is not implemented.')
        result, mask = apply_transform(result, self.transforms[0], mask)

        for key, value in result.items():
            if mask_value := self.mask_values.get(key, None) is not None:
                value[~mask] = mask_value
        if 'classes' in result:
            uncertain_class = (result['classes'] - result['classes'].round()).abs() > 0.1
            class_value = self.mask_values.get('classes', None)
            result['classes'][uncertain_class] = torch.nan if class_value is None else class_value
        if 'mask' in self.what:
            result['mask'] = ~(
                torch.nn.functional.conv2d(~mask[None, None].float(), torch.ones(1, 1, 3, 3), padding=1)[0, 0] < 1
            )
        return result


if __name__ == '__main__':
    import platformdirs

    download(CACHE_DIR, workers=4, progress=True)
    print(HASHES)
