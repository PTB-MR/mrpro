"""Brainweb Phantom."""

import concurrent.futures
import gzip
import io
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import h5py
import numpy as np
import platformdirs
import requests
import torch
import torchvision.transforms.functional
from tqdm import tqdm
from typing_extensions import TypeVar

OVERVIEW_URL = 'http://brainweb.bic.mni.mcgill.ca/brainweb/anatomic_normal_20.html'
URL_TEMPLATE = (
    'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?'
    'do_download_alias=subject{subject}_{c}'
    '&format_value=raw_short'
    '&zip_value=gnuzip'
    '&download_for_real=%5BStart+download%21%5D'
)


CLASSES = ('bck', 'skl', 'gry', 'wht', 'csf', 'mrw', 'dura', 'fat', 'fat2', 'mus', 'm-s', 'ves')  # noqa: typos
VERSION = 1
CACHE_DIR = platformdirs.user_cache_dir('mrpro')  # ~/.cache/mrpro on Linux, %AppData%\Local\mrpro on Windows
K = TypeVar('K')
TClassNames = Literal['skl', 'gry', 'wht', 'csf', 'mrw', 'dura', 'fat', 'fat2', 'mus', 'm-s', 'ves']  # noqa: typos


@dataclass
class BrainwebTissue:
    """Container for Parameters of a single tissue."""

    t1: float | tuple[float, float]
    t2: float | tuple[float, float]
    m0_abs: float | tuple[float, float]
    m0_phase: float | tuple[float, float] = 0.0

    def sample_r1(self, rng: None | torch.Generator = None) -> torch.Tensor:
        """Get randomized r1 value.

        Parameters
        ----------
        rng
            Random number generator. `None` uses the default generator.
        """
        if isinstance(self.t1, tuple):
            return 1 / torch.empty(1).uniform_(*self.t1, generator=rng)
        return 1 / torch.tensor(self.t1)

    def sample_r2(self, rng: None | torch.Generator = None) -> torch.Tensor:
        """Get randomized r2 value.

        Parameters
        ----------
        rng
            Random number generator. `None` uses the default generator.
        """
        if isinstance(self.t2, tuple):
            return 1 / torch.empty(1).uniform_(*self.t2, generator=rng)
        return 1 / torch.tensor(self.t2)

    def sample_m0(self, rng: None | torch.Generator = None) -> torch.Tensor:
        """Get renadomized complex m0 value.

        Parameters
        ----------
        rng
            Random number generator. `None` uses the default generator.
        """
        if isinstance(self.m0_abs, tuple):
            magnitude = torch.empty(1).uniform_(*self.m0_abs, generator=rng)
        else:
            magnitude = torch.tensor(self.m0_abs)
        if isinstance(self.m0_phase, tuple):
            phase = torch.empty(1).uniform_(*self.m0_phase, generator=rng)
        else:
            phase = torch.tensor(self.m0_phase)
        return torch.polar(magnitude, phase)


def affine_augment(data: torch.Tensor, size: int = 256, rng: torch.Generator | None = None) -> torch.Tensor:
    """Apply random affine augmentation.

    Parameters
    ----------
    data
        2D data to augment.
    size
        resulting image will be (size x size) pixels.
    rng
        Random number generator. `None` uses the default generator.
    """
    rand = torch.empty(6).uniform_(-1, 1, generator=rng).tolist()

    shear_x = rand[0] * 5
    shear_y = rand[1] * 5
    angle = rand[2] * 10
    scale = size / max(data.shape[1:])
    scale *= 1 + 0.1 * rand[3]
    translate = rand[4:6]

    data = torchvision.transforms.functional.affine(
        data,
        angle=angle,
        scale=scale,
        shear=[shear_x, shear_y],
        translate=translate,
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        fill=0.0,
    )
    data = torchvision.transforms.functional.center_crop(data, [size, size])
    return data


def resize(data: torch.Tensor, size: int = 256) -> torch.Tensor:
    """Resize and crop tensor.

    Parameters
    ----------
    data
        2D input tensor.
    size
        Size of the output tensor.

    Returns
    -------
    resized data
    """
    scale = size / max(data.shape[1:])
    data = torchvision.transforms.functional.resize(data, [int(scale * data.shape[1]), int(scale * data.shape[2])])
    data = torchvision.transforms.functional.center_crop(data, [size, size])
    return data


def trim_indices(mask: torch.Tensor) -> tuple[slice, slice]:
    """Get slices that remove outer masked out values.

    Parameters
    ----------
    mask
        Mask indicating valid data.

    Returns
    -------
        slices to index data
    """
    mask = mask.any(dim=tuple(range(mask.ndim - 2)))
    row_mask, col_mask = mask.any(1).short(), mask.any(0).short()
    row_min = int(torch.argmax(row_mask))
    row_max = int(mask.size(0) - torch.argmax(row_mask.flip(0)))
    col_min = int(torch.argmax(col_mask))
    col_max = int(mask.size(1) - torch.argmax(col_mask.flip(0)))
    return slice(row_min, row_max), slice(col_min, col_max)


VALUES_3T: Mapping[TClassNames, BrainwebTissue] = MappingProxyType(
    {
        'skl': BrainwebTissue((0.000, 2.000), (0.000, 0.010), (0.00, 0.05), (-0.2, 0.2)),
        'gry': BrainwebTissue((1.200, 2.000), (0.080, 0.120), (0.70, 1.00), (-0.2, 0.2)),
        'wht': BrainwebTissue((0.800, 1.500), (0.060, 0.100), (0.50, 0.90), (-0.2, 0.2)),  # noqa:typos
        'csf': BrainwebTissue((2.000, 4.000), (1.300, 2.000), (0.90, 1.00), (-0.2, 0.2)),
        'mrw': BrainwebTissue((0.400, 0.600), (0.060, 0.100), (0.70, 1.00), (-0.2, 0.2)),
        'dura': BrainwebTissue((2.000, 2.800), (0.200, 0.500), (0.90, 1.00), (-0.2, 0.2)),
        'fat': BrainwebTissue((0.300, 0.500), (0.060, 0.100), (0.90, 1.00), (-0.2, 0.2)),
        'fat2': BrainwebTissue((0.400, 0.600), (0.060, 0.100), (0.60, 0.90), (-0.2, 0.2)),
        'mus': BrainwebTissue((1.200, 1.500), (0.040, 0.060), (0.90, 1.00), (-0.2, 0.2)),
        'm-s': BrainwebTissue((0.500, 0.900), (0.300, 0.500), (0.90, 1.00), (-0.2, 0.2)),
        'ves': BrainwebTissue((1.700, 2.100), (0.200, 0.400), (0.80, 1.00), (-0.2, 0.2)),
    }
)
"""Tissue values at 3T."""

DEFAULT_VALUES = {'r1': 0.0, 'm0': 0.0, 'r2': 0.0, 'mask': 0, 'tissueclass': -1}
"""Default values for masked out regions."""


def download_brainweb(
    output_directory: str | PathLike = CACHE_DIR, workers: int = 4, progress: bool = False, compress: bool = False
) -> None:
    """Download Brainweb data.

    Parameters
    ----------
    output_directory
        Directory to save the data.
    workers
        Number of parallel downloads.
    progress
        Show progress bar.
    compress
        Use compression for HDF5 files. Saves disk space but might slow down (or speed up) access,
        depending on the system and access pattern.
    """

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
        values.pop(CLASSES.index('bck'))  # noqa: typos
        for i, x in enumerate(values):
            x = np.divide(x, sum_values, where=sum_values != 0)
            x[sum_values == 0] = 0
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

        values = norm_([unpack(downloaded_data.pop(c), shape=(362, 434, 362), dtype=np.uint16) for c in CLASSES])

        with h5py.File(outfilename, 'w') as f:
            f.create_dataset(
                'classes',
                values.shape,
                dtype=values.dtype,
                data=values,
                chunks=(4, 4, 4, values.shape[-1]) if compress else None,
                compression='lzf' if compress else None,
            )
            f.attrs['classnames'] = [c for c in CLASSES if c != 'bck']  # noqa: typos
            f.attrs['subject'] = int(subject)
            f.attrs['version'] = VERSION

    page = requests.get(OVERVIEW_URL, timeout=5)
    subjects = re.findall(r'option value=(\d*)>', page.text)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    totalsteps = len(subjects) * len(CLASSES)
    with tqdm(total=totalsteps, desc='Downloading Brainweb data', disable=not progress) as progressbar:
        for subject in subjects:
            outfilename = output_directory / f's{subject}.h5'
            if outfilename.exists():
                with h5py.File(outfilename, 'r') as f:
                    if (
                        f.attrs.get('version', 0) == VERSION
                        and f.attrs.get('subject', 0) == int(subject)
                        and f['classes'].compression == 'lzf'
                        if compress
                        else None
                    ):
                        # file is already downloaded and up to date
                        progressbar.update(len(CLASSES))
                        continue
            download_subject(subject, outfilename, workers, progressbar)


class BrainwebVolumes(torch.utils.data.Dataset):
    """3D Brainweb Dataset.

    This dataset provides 1mm isotropic 3D brain data of various quantitative MRI (qMRI) parameters.
    """

    @staticmethod
    def download(
        output_directory: str | PathLike = CACHE_DIR, workers: int = 4, progress: bool = False, compress: bool = False
    ) -> None:
        """Download Brainweb data.

        Parameters
        ----------
        output_directory
            Directory to save the data.
        workers
            Number of parallel downloads.
        progress
            Show progress bar.
        compress
            Use compression for HDF5 files. Saves disk space but might slow down or speed up access,
            depending on the file system and access pattern.

        """
        download_brainweb(output_directory, workers, progress, compress)

    def __init__(
        self,
        folder: str | Path = CACHE_DIR,
        what: Sequence[Literal['r1', 'r2', 'm0', 't1', 't2', 'mask', 'tissueclass'] | TClassNames] = ('m0', 'r1', 'r2'),
        parameters: Mapping[TClassNames, BrainwebTissue] = VALUES_3T,
        mask_values: Mapping[str, float | None] = DEFAULT_VALUES,
        seed: int | Literal['index', 'random'] = 'random',
    ) -> None:
        """Initialize Dataset.

        Parameters
        ----------
        folder:
            The directory containing Brainweb HDF5 files
        what
            What to return for each subject:
                - r1: R1 relaxation rate.
                - r2: R2 relaxation rate.
                - m0: M0 magnetization.
                - t1: T1 relaxation time.
                - t2: T2 relaxation time.
                - mask: Mask indicating valid data.
                - tissueclass: (Majority) Class index.
                - Brainweb class name: raw percentage for a specific tissue class.
        parameters
            Parameters for each tissue class.
        mask_values
            Values to use for masked out regions.
        seed
            Determines how the random number generator is initialized:
            - If ``random``, uses torch.default_generator.
            - If an integer, creates a new torch.Generator seeded with the provided value.
            - If ``index``, uses the index of the subject as seed.
        """
        self.files = list(Path(folder).glob('s??.h5'))

        if not self.files:
            raise FileNotFoundError(f'No files found in {folder}.')
        self.parameters = parameters
        self.what = what

        if seed == 'random':
            self._rng: torch.Generator | None = torch.default_generator
        elif isinstance(seed, int):
            self._rng = torch.Generator().manual_seed(seed)
        elif seed == 'index':
            self._rng = None

        self.mask_values = mask_values

    def __len__(self) -> int:
        """Get number of subjects."""
        return len(self.files)

    def __getitem__(
        self, index: int
    ) -> dict[Literal['r1', 'r2', 'm0', 't1', 't2', 'mask', 'tissueclass'] | TClassNames, torch.Tensor]:
        """Get data from one subject."""
        with h5py.File(self.files[index]) as file:
            data = torch.as_tensor(np.array(file['classes'], dtype=np.uint8))
            classnames = tuple(file.attrs['classnames'])
        result: dict[Literal['r1', 'r2', 'm0', 't1', 't2', 'mask', 'tissueclass'] | TClassNames, torch.Tensor] = {}
        for el in self.what:
            if el == 'r1':
                values = torch.stack([self.parameters[k].sample_r1() for k in classnames]) / 255
                result[el] = (data.to(values) @ values)[..., 0]
            elif el == 'r2':
                values = torch.stack([self.parameters[k].sample_r2() for k in classnames]) / 255
                result[el] = (data.to(values) @ values)[..., 0]
            elif el == 'm0':
                values = torch.stack([self.parameters[k].sample_m0() for k in classnames]) / 255
                result[el] = (data.to(values) @ values)[..., 0]
            elif el == 't1':
                values = torch.stack([self.parameters[k].sample_r1() for k in classnames]) / 255
                result[el] = (data.to(values) @ values)[..., 0].reciprocal()
            elif el == 't2':
                values = torch.stack([self.parameters[k].sample_r2() for k in classnames]) / 255
                result[el] = (data.to(values) @ values)[..., 0].reciprocal()
            elif el == 'tissueclass':
                result[el] = data.argmax(-1)
            elif el in classnames:
                result[el] = data[..., classnames.index(el)] / 255
            elif el == 'mask':
                mask = data.sum(-1) < 150
                mask = (
                    torch.nn.functional.conv3d(mask[None, None].float(), torch.ones(1, 1, 3, 3, 3), padding=1)[0, 0] < 1
                )
                result[el] = ~mask
            else:
                raise NotImplementedError(f'what=({el},) is not implemented.')

        for key, value in result.items():
            if key == 'mask':
                continue
            if (mask_value := self.mask_values.get(key, None)) is not None:
                value[~mask] = mask_value
            elif key not in classnames:
                value[~mask] = torch.nan

        return result


class BrainwebSlices(torch.utils.data.Dataset):
    """Dataset of 2D qMRI parameter slices based on Brainweb dataset."""

    @staticmethod
    def download(
        output_directory: str | PathLike = CACHE_DIR, workers: int = 4, progress: bool = False, compress: bool = False
    ) -> None:
        """Download Brainweb data.

        Parameters
        ----------
        output_directory
            Directory to save the data.
        workers
            Number of parallel downloads.
        progress
            Show progress bar.
        compress
            Use compression for HDF5 files. Saves disk space but might slow down or speed up access,
            depending on the file system and access pattern.

        """
        download_brainweb(output_directory, workers, progress, compress)

    def __init__(
        self,
        folder: str | Path = CACHE_DIR,
        what: Sequence[Literal['r1', 'r2', 'm0', 't1', 't2', 'mask', 'tissueclass'] | TClassNames] = ('m0', 'r1', 'r2'),
        parameters: Mapping[TClassNames, BrainwebTissue] = VALUES_3T,
        orientation: Literal['axial', 'coronal', 'sagittal'] = 'axial',
        skip_slices: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = ((80, 80), (100, 100), (100, 100)),
        step: int = 1,
        matrix_size: int = 256,
        augmentations: bool = True,
        mask_values: Mapping[str, float | None] = DEFAULT_VALUES,
        seed: int | Literal['index', 'random'] = 'random',
    ) -> None:
        """Initialize Brainweb qMRI slice phantom.

        Parameters
        ----------
        folder
            Folder with Brainweb data as HDF5 files.
        what
            What to return for each slice:
                - r1: R1 relaxation rate.
                - r2: R2 relaxation rate.
                - m0: M0 magnetization.
                - t1: T1 relaxation time.
                - t2: T2 relaxation time.
                - mask: Mask indicating valid data.
                - tissueclass: Class index.
        parameters
            Parameters for each tissue class.
        orientation
            Orientation of slices (axial, coronal, or sagittal).
        skip_slices
            specifies how much many slices to skip from the beginning and end for each of axial, coronal,
            or sagittal orientation.
        step
            Step size between slices, in voxel units (1mm).
        matrix_size
            Size of output slice (pixels).
        augmentations
            Whether to apply random augmentations to slices.
        mask_values
            Values to use for masked out regions.
        seed
            Random seed - can be an int, ``index`` to use slice index as seed, or ``random`` for random seed.
        """
        self.parameters = parameters
        self.step = step
        self.what = what
        self.mask_values = mask_values

        try:
            self._axis = {'axial': 0, 'coronal': 1, 'sagittal': 2}[orientation]
        except KeyError:
            raise ValueError(f'Invalid axis: {orientation}.') from None
        self._skip_slices = skip_slices[self._axis]

        files = []
        ns_slices = [0]
        for fn in Path(folder).glob('s??.h5'):
            with h5py.File(fn) as f:
                n_slices = f['classes'].shape[self._axis] - self._skip_slices[0] - self._skip_slices[1]
                ns_slices.append(n_slices)
                files.append(fn)
        if not files:
            raise FileNotFoundError(f'No files found in {folder}.')
        self._files = tuple(files)
        self._ns_slices = np.cumsum(ns_slices)

        if augmentations:
            self.transforms = partial(affine_augment, size=matrix_size)
        else:
            self.transforms = partial(resize, size=matrix_size)

        if seed == 'random':
            self._rng: torch.Generator | None = torch.default_generator
        elif isinstance(seed, int):
            self._rng = torch.Generator().manual_seed(seed)
        elif seed == 'index':
            self._rng = None

    def __len__(self) -> int:
        """Get number of slices."""
        return self._ns_slices[-1] // self.step

    def __getitem__(
        self, index: int
    ) -> dict[Literal['r1', 'r2', 'm0', 't1', 't2', 'mask', 'tissueclass'] | TClassNames, torch.Tensor]:
        """Get a single slice."""
        if index * self.step >= self._ns_slices[-1]:
            raise IndexError
        elif index < 0:
            index = self._ns_slices[-1] + index * self.step
        else:
            index = index * self.step

        file_id = np.searchsorted(self._ns_slices, index, 'right') - 1
        slice_id = index - self._ns_slices[file_id] + self._skip_slices[0]

        with h5py.File(
            self._files[file_id],
        ) as file:
            where = [slice(self._skip_slices[0], file['classes'].shape[i] - self._skip_slices[1]) for i in range(3)] + [
                slice(None)
            ]
            where[self._axis] = slice_id
            data = torch.as_tensor(np.array(file['classes'][tuple(where)], dtype=np.uint8))
            classnames = tuple(file.attrs['classnames'])
        rng = torch.Generator().manual_seed(index) if self._rng is None else self._rng
        data = data[trim_indices(data.sum(-1) > 0.5)]
        data = self.transforms(data.moveaxis(-1, 0) / 255, rng=rng).moveaxis(0, -1)
        mask = data.sum(-1) > 0.5
        result: dict[Literal['r1', 'r2', 'm0', 't1', 't2', 'mask', 'tissueclass'] | TClassNames, torch.Tensor] = {}

        for el in self.what:
            if el == 'r1':
                values = torch.stack([self.parameters[k].sample_r1(rng) for k in classnames])
                result[el] = (data.to(values) @ values)[..., 0]
            elif el == 'r2':
                values = torch.stack([self.parameters[k].sample_r2(rng) for k in classnames])
                result[el] = (data.to(values) @ values)[..., 0]
            elif el == 'm0':
                values = torch.stack([self.parameters[k].sample_m0(rng) for k in classnames])
                result[el] = (data.to(values) @ values)[..., 0]
            elif el == 't1':
                values = torch.stack([self.parameters[k].sample_r1(rng) for k in classnames])
                result[el] = (data.to(values) @ values)[..., 0].reciprocal()
            elif el == 't2':
                values = torch.stack([self.parameters[k].sample_r2(rng) for k in classnames])
                result[el] = (data.to(values) @ values)[..., 0].reciprocal()
            elif el == 'tissueclass':
                result[el] = data.argmax(-1)
            elif el in classnames:
                result[el] = data[..., classnames.index(el)]
            elif el == 'mask':
                result[el] = ~(
                    torch.nn.functional.conv2d((~mask)[None, None].float(), torch.ones(1, 1, 3, 3), padding=1)[0, 0] < 1
                )
            else:
                raise NotImplementedError(f'what=({el},) is not implemented.')

        for key, value in result.items():
            if key == 'mask':
                continue
            if (mask_value := self.mask_values.get(key, None)) is not None:
                value[~mask] = mask_value
            elif key not in classnames:
                value[~mask] = torch.nan

        return result
