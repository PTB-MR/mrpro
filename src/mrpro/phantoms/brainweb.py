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
from types import MappingProxyType
from typing import Literal, cast

import h5py
import numpy as np
import platformdirs
import requests
import torch
import torchvision
from tqdm import tqdm
from typing_extensions import TypeVar

HASHES = MappingProxyType(
    {
        '04': '84408e7f1f283722de0555cdb8dc014b',
        '05': 'ea316cef36e4575a0410f033c9dae532',
        '06': 'ad695379f4d0a8209483959c6d143538',
        '18': 'fe847558357f8164800c652a146a1c77',
        '20': '467d52fdf4d7b6668fc61ecb329e9b8b',
        '38': '2a3dcce823e8e8ae60805d24a96f73c8',
        '41': 'f8c57277328e7c8ec0fd047b78922b71',
        '42': '944c128c019a4b5610030cd846a0f25a',
        '43': 'a9910dea3d9db6afaaeaae47bbe15c22',
        '44': 'd380d9700c84f48462c820c8f7f2be4a',
        '45': '4d82f7cf02f47bccc9b39e00b3874042',
        '46': '8df1985c9c613f5799065f648768c8df',
        '47': '8c5ac60ce5ae496917f9fe6fdbf4df49',
        '48': '590a3dac01de32f6f17c40a542402e43',
        '49': '985ab9752355de81a16ae08cfedbafaf',
        '50': 'ce7205b609f265729693b5f4d6983ee8',
        '51': 'be78055171f5c19a999655ba2abaab30',
        '52': 'e017d263c2fe62ad441f072bc8ee2d85',
        '53': '0d65a61628bb0fba84816a054d13cbf4',
        '54': 'abd56a191d00fab9fec1d7849705303e',
    }
)
OVERVIEW_URL = 'http://brainweb.bic.mni.mcgill.ca/brainweb/anatomic_normal_20.html'
URL_TEMPLATE = (
    'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?'
    'do_download_alias=subject{subject}_{c}'
    '&format_value=raw_short'
    '&zip_value=gnuzip'
    '&download_for_real=%5BStart+download%21%5D'
)


CLASSES = ('bck', 'skl', 'gry', 'wht', 'csf', 'mrw', 'dura', 'fat', 'fat2', 'mus', 'm-s', 'ves')  # noqa: typos

CACHE_DIR = platformdirs.user_cache_dir('mrpro')
K = TypeVar('K')


@dataclass
class T1T2M0:
    """Container for Parameters of a single tissue."""

    t1_min: float
    t1_max: float
    t2_min: float
    t2_max: float
    m0_min: float
    m0_max: float
    phase_min: float = -0.01
    phase_max: float = 0.01

    def r1(self, rng: None | torch.Generator = None) -> torch.Tensor:
        """Get randomized r1 value."""
        return 1 / torch.empty(1).uniform_(self.t1_min, self.t1_max, generator=rng)

    def r2(self, rng: None | torch.Generator = None) -> torch.Tensor:
        """Get randomized r2 value."""
        return 1 / torch.empty(1).uniform_(self.t2_min, self.t2_max, generator=rng)

    def m0(self, rng: None | torch.Generator = None) -> torch.Tensor:
        """Get renadomized complex m0 value."""
        return torch.polar(
            torch.empty(1).uniform_(self.m0_min, self.m0_max, generator=rng),
            torch.empty(1).uniform_(self.phase_min, self.phase_max, generator=rng),
        )


VALUES_3T = MappingProxyType(
    {
        'skl': T1T2M0(0.000, 2.000, 0.000, 0.010, 0.0, 0.05),
        'gry': T1T2M0(1.200, 2.000, 0.080, 0.120, 0.7, 1.0),
        'wht': T1T2M0(0.800, 1.500, 0.060, 0.100, 0.50, 0.9),  # noqa:typos
        'csf': T1T2M0(2.000, 4.000, 1.300, 2.000, 0.9, 1.0),
        'mrw': T1T2M0(0.400, 0.600, 0.060, 0.100, 0.7, 1.0),
        'dura': T1T2M0(2.000, 2.800, 0.200, 0.500, 0.9, 1.0),
        'fat': T1T2M0(0.300, 0.500, 0.060, 0.100, 0.9, 1.0),
        'fat2': T1T2M0(0.400, 0.600, 0.060, 0.100, 0.6, 0.9),
        'mus': T1T2M0(1.200, 1.500, 0.040, 0.060, 0.9, 1.0),
        'm-s': T1T2M0(0.500, 0.900, 0.300, 0.500, 0.9, 1),
        'ves': T1T2M0(1.700, 2.100, 0.200, 0.400, 0.8, 1),
    }
)
"""Tissue values at 3T."""

DEFAULT_VALUES = {'r1': 0.0, 'm0': 0.0, 'r2': 0.0, 'mask': 0, 'classes': -1}
"""Default values for masked out regions."""


def _trim_slices(mask: torch.Tensor) -> tuple[slice, slice]:
    """Get slices that remove outer masked out values."""
    mask = mask.any(dim=tuple(range(mask.ndim - 2)))
    row_mask, col_mask = mask.any(1).short(), mask.any(0).short()
    row_min = int(torch.argmax(row_mask))
    row_max = int(mask.size(0) - torch.argmax(row_mask.flip(0)))
    col_min = int(torch.argmax(col_mask))
    col_max = int(mask.size(1) - torch.argmax(col_mask.flip(0)))
    return slice(row_min, row_max), slice(col_min, col_max)


def _apply_transform(
    data: Mapping[K, torch.Tensor],
    transform: Callable[[torch.Tensor], torch.Tensor],
    mask: torch.Tensor,
    default_values: Mapping[str, float | None] = DEFAULT_VALUES,
) -> tuple[dict[K, torch.Tensor], torch.Tensor]:
    """Apply a transformation."""
    # We need to stack the data into a single tensor, otherwise random transformations will be different for each
    # data element.
    tmp = []
    for name, x in data.items():
        x = x.clone()
        default_value = default_values.get(cast(str, name), None)
        x[~mask] = 0.0 if default_value is None else default_value
        if torch.is_complex(x):
            tmp.append(x.real)
            tmp.append(x.imag)
        else:
            tmp.append(x)

    tensor = torch.stack(tmp, 0)
    tensor = transform(tensor)
    newmask = ~(tensor.isnan().any(0) | (tensor == 0).all(0))
    newdata = {}

    i = 0
    for name, x in data.items():
        if torch.is_complex(x):
            newdata[name] = tensor[i] + 1j * tensor[i + 1]
            i += 2
        else:
            newdata[name] = tensor[i]
            i += 1
    return newdata, newmask


TRANSFORMS_RANDOM_256 = (
    torchvision.transforms.Resize(256),
    torchvision.transforms.RandomAffine(
        degrees=5,
        translate=(0.01, 0.01),
        scale=(0.95, 1.05),
        fill=0.0,
        shear=(0, 5, 0, 5),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    ),
    torchvision.transforms.CenterCrop((256, 256)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomVerticalFlip(),
)
"""Random transformations for 256x256 images."""

TRANSFORMS_RESIZE_256 = (torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop((256, 256)))
"""Non-random transformations for 256x256 images."""

DEFAULT_CUTS = {'axial': (80, 80, 0, 0, 0, 0), 'coronal': (0, 0, 100, 100, 0, 0), 'sagittal': (0, 0, 0, 0, 100, 100)}
"""Default cuts for axial, coronal, and sagittal slices."""


class BrainwebSlices(torch.utils.data.Dataset):
    """Dataset of 2D qMRI parameter slices based on Brainweb dataset."""

    @staticmethod
    def download(output_directory: str | PathLike = CACHE_DIR, workers: int = 4, progress: bool = False) -> None:
        """Download Brainweb data with subjects in series and class files in parallel."""

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
            values.pop(CLASSES.index('back'))
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
                    chunks=(4, 4, 4, values.shape[-1]),
                    compression='lzf',
                )
                f.attrs['classnames'] = [c for c in CLASSES if c != 'back']
                f.attrs['subject'] = int(subject)

        page = requests.get(OVERVIEW_URL, timeout=5)
        subjects = re.findall(r'option value=(\d*)>', page.text)
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        totalsteps = len(subjects) * len(CLASSES)
        with tqdm(total=totalsteps, desc='Downloading Brainweb data', disable=not progress) as progressbar:
            for subject in subjects:
                outfilename = output_directory / f's{subject}.h5'
                if (
                    outfilename.exists()
                    and hashlib.file_digest(outfilename.open('rb'), 'md5').hexdigest() == HASHES[subject]
                ):
                    progressbar.update(len(CLASSES))
                    continue
                download_subject(subject, outfilename, workers, progressbar)
                if not hashlib.file_digest(outfilename.open('rb'), 'md5').hexdigest() == HASHES[subject]:
                    raise RuntimeError(f'Hash mismatch for subject {subject}')

    def __init__(
        self,
        folder: str | Path,
        parameters: Mapping[str, T1T2M0] = VALUES_3T,
        orientation: Literal['axial', 'coronal', 'sagittal'] = 'axial',
        cuts: tuple[int, int, int, int, int, int] | Mapping[str, tuple[int, int, int, int, int, int]] = DEFAULT_CUTS,
        step: int = 1,
        transforms: Sequence[Callable[[torch.Tensor], torch.Tensor]] = TRANSFORMS_RANDOM_256,
        what: Sequence[Literal['r1', 'r2', 'm0', 't1', 't2', 'mask', 'classes']] = ('m0', 'r1', 'r2'),
        mask_values: Mapping[str, float | None] = DEFAULT_VALUES,
        value_randomization: Literal['index', 'random'] | int = 'random',
    ) -> None:
        """Initialize Brainweb qMRI slice phantom.

        Parameters
        ----------
        folder
            Folder with Brainweb data as HDF5 files.
        parameters
            Parameters for each tissue class.
        orientation
            Orientation of slices.
        cuts
            If a tuple of 6 integers, specifies how much to cut away from the 3D volume to the left and right.
            If a mapping, the cut values matching the `orientation` are used. In voxel units (1mm)
        step
            Step size between slices, in voxel units (1mm).
        transforms
            Transforms/augmentations to apply to 2D slices.
        what
            What to return for each slice:
                - 'r1': R1 relaxation rate.
                - 'r2': R2 relaxation rate.
                - 'm0': M0 magnetization.
                - 't1': T1 relaxation time.
                - 't2': T2 relaxation time.
                - 'mask': Mask indicating valid data.
                - 'classes': Class index.
        mask_values
            Values to use for masked out regions.
        value_randomization
            How to randomize values:
                - 'index': Randomize based on slice index.
                - 'random': Randomize based on a random seed.
                - int: Randomize based on a fixed seed.

        Returns
        -------
        Dictionary with keys corresponding to `what` and values being 2D tensors.
        """
        self.parameters = parameters
        if isinstance(cuts, Mapping):
            try:
                self._cuts = cuts[orientation]
            except KeyError:
                raise KeyError(f'Axis {orientation} not found in cuts.') from None
        else:
            self._cuts = cuts
        try:
            self._axis = {'axial': 0, 'coronal': 1, 'sagittal': 2}[orientation]
        except KeyError:
            raise ValueError(f'Invalid axis: {orientation}.') from None
        self.step = step
        files = []
        ns = [0]
        for fn in Path(folder).glob('s??.h5'):
            with h5py.File(fn) as f:
                ns.append(
                    (f['classes'].shape[self._axis]) - (self._cuts[self._axis * 2] + self._cuts[self._axis * 2 + 1])
                )
                files.append(fn)
        if not files:
            raise FileNotFoundError(f'No files found in {folder}.')
        self._files = tuple(files)
        self._ns = np.cumsum(ns)
        self.transforms = torchvision.transforms.Compose(transforms)
        self.what = what
        self.mask_values = mask_values
        self.value_randomization = value_randomization

    def __len__(self) -> int:
        """Get number of slices."""
        return self._ns[-1] // self.step

    def __getitem__(self, index: int) -> dict[Literal['r1', 'r2', 'm0', 't1', 't2', 'mask', 'classes'], torch.Tensor]:
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
            slices = _trim_slices(mask)
            mask = mask[slices]
            data = data[slices]

            result: dict[Literal['r1', 'r2', 'm0', 't1', 't2', 'mask', 'classes'], torch.Tensor] = {}
            for el in self.what:
                if self.value_randomization == 'index':
                    rng = torch.Generator().manual_seed(index)
                elif self.value_randomization == 'random':
                    rng = None
                else:
                    rng = torch.Generator().manual_seed(self.value_randomization)

                if el == 'r1':
                    values = torch.stack([self.parameters[k].r1(rng) for k in classnames]) / 255
                    result[el] = (data.to(values) @ values)[..., 0]
                elif el == 'r2':
                    values = torch.stack([self.parameters[k].r2(rng) for k in classnames]) / 255
                    result[el] = (data.to(values) @ values)[..., 0]
                elif el == 'm0':
                    values = torch.stack([self.parameters[k].m0(rng) for k in classnames]) / 255
                    result[el] = (data.to(values) @ values)[..., 0]
                elif el == 't1':
                    values = torch.stack([self.parameters[k].r1(rng) for k in classnames]) / 255
                    result[el] = (data.to(values) @ values)[..., 0].reciprocal()
                elif el == 't2':
                    values = torch.stack([self.parameters[k].r2(rng) for k in classnames]) / 255
                    result[el] = (data.to(values) @ values)[..., 0].reciprocal()
                elif el == 'classes':
                    result[el] = data.argmax(-1)
                else:
                    raise NotImplementedError(f'what=({el},) is not implemented.')
        result, mask = _apply_transform(result, self.transforms, mask)

        for key, value in result.items():
            if (mask_value := self.mask_values.get(key, None)) is not None:
                value[~mask] = mask_value
            else:
                value[~mask] = torch.nan
        if 'classes' in result:
            uncertain_class = (result['classes'] - result['classes'].round()).abs() > 0.1
            class_value = self.mask_values.get('classes', None)
            result['classes'][uncertain_class] = torch.nan if class_value is None else class_value
        if 'mask' in self.what:
            result['mask'] = ~(
                torch.nn.functional.conv2d(~mask[None, None].float(), torch.ones(1, 1, 3, 3), padding=1)[0, 0] < 1
            )
        return result
