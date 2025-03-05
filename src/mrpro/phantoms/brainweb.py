"""Brainweb Phantom."""

import concurrent.futures
import gzip
import io
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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

# includes background
ALL_CLASSES = ('bck', 'skl', 'gry', 'wht', 'csf', 'mrw', 'dura', 'fat', 'fat2', 'mus', 'm-s', 'ves')  # noqa: typos
VERSION = 1
CACHE_DIR = platformdirs.user_cache_dir('mrpro')  #  ~/.cache/mrpro on Linux, %AppData%\Local\mrpro on Windows
K = TypeVar('K')
TClassNames = Literal['skl', 'gry', 'wht', 'csf', 'mrw', 'dura', 'fat', 'fat2', 'mus', 'm-s', 'ves']  # noqa: typos


@dataclass
class BrainwebTissue:
    """Container for parameters of a single tissue.

    Attributes are either single values or ranges.
    If ranges, values are sampled uniformly from within this range by the `sample_r1`, `sample_r2`, and `sample_m0`
    methods. If a single value is given, this value is returned by the respective method.
    """

    t1: float | tuple[float, float]
    """T1 value or range (T1_min, T1_max) in seconds."""

    t2: float | tuple[float, float]
    """T2 value or range (T2_min, T2_max) in seconds."""

    m0_abs: float | tuple[float, float]
    """Absolute value or range (M0_min, M0_max) of the complex M0."""

    m0_phase: float | tuple[float, float] = 0.0
    """Phase value or range (Phase_min, Phase_max) of the complex M0 in radians."""

    def sample_r1(self, rng: None | torch.Generator = None) -> torch.Tensor:
        """Get (possibly randomized) r1=1/t1 value.

        Parameters
        ----------
        rng
            Random number generator. `None` uses the default generator.
        """
        if isinstance(self.t1, tuple):
            return 1 / torch.empty(1).uniform_(*self.t1, generator=rng)
        return 1 / torch.tensor(self.t1)

    def sample_r2(self, rng: None | torch.Generator = None) -> torch.Tensor:
        """Get (possibly randomized) r2=1/t2 value.

        Parameters
        ----------
        rng
            Random number generator. `None` uses the default generator.
        """
        if isinstance(self.t2, tuple):
            return 1 / torch.empty(1).uniform_(*self.t2, generator=rng)
        return 1 / torch.tensor(self.t2)

    def sample_m0(self, rng: None | torch.Generator = None) -> torch.Tensor:
        """Get (possibly randomized) complex m0 value.

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


def augment(
    size: int = 256,
    trim: bool = True,
    max_random_shear: float = 5,
    max_random_rotation: float = 10,
    max_random_scaling_factor: float = 0.1,
    p_horizontal_flip: float = 0.5,
    p_vertical_flip: float = 0.5,
) -> Callable[[torch.Tensor, torch.Generator | None], torch.Tensor]:
    """Get an augmentation function.

    Creates a function that applies augmentation to the input tensor consisting of
    rotation, shearing, scaling and horizontal/vertical flipping.

    The image is scaled such that the largest dimension is in
    [size * (1 - max_random_scaling), size * (1 + max_random_scaling)], then padded/cropped to size `size x size`.
    In scaling, the aspect ratio is preserved.
    Random horizontal and vertical flips are applied with probability `p_horizontal_flip` and `p_vertical_flip`.

    Parameters
    ----------
    size
        resulting image will be (size x size) pixels.
    trim
        If True, remove fully zero outer rows and columns before scaling
    max_random_shear
        Maximum random shear in degrees, shear is in [-max_shear, max_shear] in x and y direction.
    max_random_rotation
        Maximum random rotation in degrees, rotation is in [-max_rotation, max_rotation].
    max_random_scaling_factor
        Strength of the scaling randomization (see above).
    p_horizontal_flip
        Probability of horizontal flip.
    p_vertical_flip
        Probability of vertical flip.

    Returns
    -------
        Callable that applies augmentation to the input tensor
    """

    def augment_fn(data: torch.Tensor, rng: torch.Generator | None = None) -> torch.Tensor:
        """Apply augmentation to the input tensor."""
        rand = torch.empty(6).uniform_(-1, 1, generator=rng).tolist()

        shear_x = rand[0] * max_random_shear
        shear_y = rand[1] * max_random_shear
        angle = rand[2] * max_random_rotation
        scale = size / max(data.shape[-2:])
        scale *= 1 + max_random_scaling_factor * rand[3]
        translate = rand[4:6]  # subpixel translation for edge aliasing
        if trim:
            data = data[trim_indices(data.sum(-1) > 0.1 * data.amax())]

        data = torchvision.transforms.functional.affine(
            data,
            angle=angle,
            scale=scale,
            shear=[shear_x, shear_y],
            translate=translate,
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            fill=0.0,
        )
        rand = torch.empty(2).uniform_(0, 1, generator=rng).tolist()
        data = torchvision.transforms.functional.hflip(data) if rand[0] < p_horizontal_flip else data
        data = torchvision.transforms.functional.vflip(data) if rand[1] < p_vertical_flip else data

        data = torchvision.transforms.functional.center_crop(data, [size, size])

        return data

    return augment_fn


def resize(size: int = 256) -> Callable[[torch.Tensor, torch.Generator | None], torch.Tensor]:
    """Get a resizing and cropping function.

    Parameters
    ----------
    size
        Size of the output tensor.

    Returns
    -------
        Callable that resizes and crops the input tensor.
    """

    def resize_fn(data: torch.Tensor, rng: torch.Generator | None = None) -> torch.Tensor:  # noqa: ARG001
        """Apply resizing and cropping to the input tensor."""
        scale = size / max(data.shape[-2:])
        data = torchvision.transforms.functional.resize(data, [int(scale * data.shape[1]), int(scale * data.shape[2])])
        data = torchvision.transforms.functional.center_crop(data, [size, size])
        return data

    return resize_fn


DEFAULT_AUGMENT_256 = augment(256)
"""Default random augmentation for 256x256 images."""


def trim_indices(mask: torch.Tensor) -> tuple[slice, slice]:
    """Get slices that remove fully masked out outer rows and columns.

    Parameters
    ----------
    mask
        Mask indicating valid data.

    Returns
    -------
        Two `slice` objects, that can be used to index the data
        to remove fully masked out outer rows and columns.
    """
    mask = mask.any(dim=tuple(range(mask.ndim - 2)))
    row_mask, col_mask = mask.any(1).short(), mask.any(0).short()
    row_min = int(torch.argmax(row_mask))
    row_max = int(mask.size(0) - torch.argmax(row_mask.flip(0)))
    col_min = int(torch.argmax(col_mask))
    col_max = int(mask.size(1) - torch.argmax(col_mask.flip(0)))
    return slice(row_min, row_max), slice(col_min, col_max)


VALUES_3T_RANDOMIZED: Mapping[TClassNames, BrainwebTissue] = MappingProxyType(
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
"""Tissue values for 3T with wide randomization ranges."""

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
        values.pop(ALL_CLASSES.index('bck'))  # noqa: typos
        for i, x in enumerate(values):
            x = np.divide(x, sum_values, where=sum_values != 0)
            x[sum_values == 0] = 0
            x = (x * (2**8 - 1)).astype(np.uint8)
            values[i] = x
        return np.stack(values, -1)

    def download_subject(subject: str, outfilename: Path, workers: int, progressbar: tqdm) -> None:
        """Download and process all class files for a single subject asynchronously."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(load_file, URL_TEMPLATE.format(subject=subject, c=c)): c for c in ALL_CLASSES}

            downloaded_data = {}
            for future in concurrent.futures.as_completed(futures):
                c = futures[future]
                downloaded_data[c] = future.result()
                progressbar.update(1)

        values = norm_([unpack(downloaded_data.pop(c), shape=(362, 434, 362), dtype=np.uint16) for c in ALL_CLASSES])

        with h5py.File(outfilename, 'w') as f:
            f.create_dataset(
                'classes',
                values.shape,
                dtype=values.dtype,
                data=values,
                chunks=(4, 4, 4, values.shape[-1]) if compress else None,
                compression='lzf' if compress else None,
            )
            f.attrs['classnames'] = [c for c in ALL_CLASSES if c != 'bck']  # noqa: typos
            f.attrs['subject'] = int(subject)
            f.attrs['version'] = VERSION

    page = requests.get(OVERVIEW_URL, timeout=5)
    subjects = re.findall(r'option value=(\d*)>', page.text)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    totalsteps = len(subjects) * len(ALL_CLASSES)
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
                        progressbar.update(len(ALL_CLASSES))
                        continue
            download_subject(subject, outfilename, workers, progressbar)


class BrainwebVolumes(torch.utils.data.Dataset):
    """Dataset of 3D qMRI parameters based on Brainweb dataset.

    This dataset provides 1 mm isotropic 3D brain data of various quantitative MRI (qMRI)
    parameters based on the segmentations provided by the Brainweb [AubertBroche2006]_ dataset.

    References
    ----------
    .. [AubertBroche2006] Aubert-Broche, B., Griffin, M., Pike, G.B., Evans, A.C., & Collins, D.L. (2006).
       Twenty New Digital Brain Phantoms for Creation of Validation Image Data Bases.
       IEEE Transactions on Medical Imaging, 25 (11), 1410-1416. https://doi.org/10.1109/TMI.2006.883453
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
        parameters: Mapping[TClassNames, BrainwebTissue] = VALUES_3T_RANDOMIZED,
        mask_values: Mapping[str, float | None] = DEFAULT_VALUES,
        seed: int | Literal['index', 'random'] = 'random',
    ) -> None:
        """Initialize Dataset.

        Parameters
        ----------
        folder:
            The directory containing Brainweb HDF5 files
        what
            What to return for each subject.
            Possible values are:
                - r1: R1 relaxation rate.
                - r2: R2 relaxation rate.
                - m0: M0 magnetization.
                - t1: T1 relaxation time.
                - t2: T2 relaxation time.
                - mask: mask indicating valid data.
                - tissueclass: (majority) class index.
                - Brainweb class name: raw percentage of the specific tissue class (see below for possible values).
        parameters
            Parameters for each tissue class.
            The Brainweb tissue classes are:
                - skl: skull
                - gry: gray matter
                - wht: white matter
                - csf: cerebrospinal fluid
                - mrw: bone marrow
                - dura: dura
                - fat: fat
                - fat2: fat and tissue
                - mus: muscle
                - m-s: skin
                - ves: vessels
        mask_values
            Default values to use for masked out regions.
        seed
            Random seed. Can be an int, the strings ``index`` to use subject index as seed,
            or ``random`` for random seed.
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
        """Get 3D data from one subject."""
        with h5py.File(self.files[index]) as file:
            data = torch.as_tensor(np.array(file['classes'], dtype=np.uint8))
            classnames = tuple(file.attrs['classnames'])
        result: dict[Literal['r1', 'r2', 'm0', 't1', 't2', 'mask', 'tissueclass'] | TClassNames, torch.Tensor] = {}
        for el in self.what:
            if el == 'r1':
                # / 255 to convert from uint8 to 0...1
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
    """Dataset of 2D qMRI parameter slices based on Brainweb dataset.

    Dataset of agmented 2D qMRI parameter slices based on the segmentations of
    the Brainweb dataset [AubertBroche2006]_.


    References
    ----------
    .. [AubertBroche2006] Aubert-Broche, B., Griffin, M., Pike, G.B., Evans, A.C., & Collins, D.L. (2006).
       Twenty New Digital Brain Phantoms for Creation of Validation Image Data Bases.
       IEEE Transactions on Medical Imaging, 25 (11), 1410-1416. https://doi.org/10.1109/TMI.2006.883453
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
        parameters: Mapping[TClassNames, BrainwebTissue] = VALUES_3T_RANDOMIZED,
        orientation: Literal['axial', 'coronal', 'sagittal'] = 'axial',
        skip_slices: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = ((80, 80), (100, 100), (100, 100)),
        step: int = 1,
        slice_preparation: Callable[[torch.Tensor, torch.Generator | None], torch.Tensor] = DEFAULT_AUGMENT_256,
        mask_values: Mapping[str, float | None] = DEFAULT_VALUES,
        seed: int | Literal['index', 'random'] = 'random',
    ) -> None:
        """Initialize Brainweb qMRI slice phantom.

        Parameters
        ----------
        folder
            Folder with Brainweb data as HDF5 files.
        what
            What to return for each slice.
            Possible values are:
                - r1: R1 relaxation rate.
                - r2: R2 relaxation rate.
                - m0: M0 magnetization.
                - t1: T1 relaxation time.
                - t2: T2 relaxation time.
                - mask: mask indicating valid data.
                - tissueclass: class index.
        parameters
            Parameters for each tissue class.
            The Brainweb tissue classes are:
                - skl: skull
                - gry: gray matter
                - wht: white matter
                - csf: cerebrospinal fluid
                - mrw: bone marrow
                - dura: dura
                - fat: fat
                - fat2: fat and Tissue
                - mus: muscle
                - m-s: skin
                - ves: vessels
        orientation
            Orientation of slices (axial, coronal, or sagittal).
        skip_slices
            Specifies how much many slices to skip from the beginning and end for each of axial, coronal,
            or sagittal orientation.
        step
            Step size between slices, in voxel units (1 mm).
        slice_preparation
            Callable that performs slice augmentation and resizing, see `resize` or `augment` for examples.
            The default applies slight random rotation, shear, scaling, and flips, and scales to 256x256 images.
        mask_values
            Default values to use for masked out regions.
        seed
            Random seed. Can be an int, the strings ``index`` to use slice index as seed, or ``random`` for random seed.
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

        self.slice_preparation = slice_preparation

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
        data = self.slice_preparation(data.moveaxis(-1, 0) / 255, rng).moveaxis(
            0, -1
        )  # / 255 to convert from uint8 to 0...1
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
