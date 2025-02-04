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
OVERVIEW_URL = 'http://brainweb.bic.mni.mcgill.ca/brainweb/anatomic_normal_20.html'
URL_TEMPLATE = (
    'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?'
    'do_download_alias=subject{subject}_{c}'
    '&format_value=raw_short'
    '&zip_value=gnuzip'
    '&download_for_real=%5BStart+download%21%5D'
)


CLASSES = ['bck', 'skl', 'gry', 'wht', 'csf', 'mrw', 'dura', 'fat', 'fat2', 'mus', 'm-s', 'ves']  # noqa: typos

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
            if (
                outfilename.exists()
                and hashlib.file_digest(outfilename.open('rb'), 'md5').hexdigest() == HASHES[subject]
            ):
                progressbar.update(len(CLASSES))
                continue
            download_subject(subject, outfilename, workers, progressbar)
            # HASHES[subject] = hashlib.file_digest(outfilename.open('rb'), 'md5').hexdigest()
            # if not hashlib.file_digest(outfilename.open('rb'), 'md5').hexdigest() == HASHES[subject]:
            #     raise RuntimeError(f'Hash mismatch for subject {subject}.')


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
            torch.empty(1).uniform_(self.m0_min, self.m0_max),
            torch.empty(1).uniform_(self.phase_min, self.phase_max),
        )


VALUES_3T = {
    'skl': T1T2PD(0.000, 2.000, 0.000, 0.010, 0.0, 0.05),
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

DEFAULT_VALUES = {'r1': 0.0, 'pd': 0.0, 'r2': 0.0, 'mask': 0, 'classes': -1}


def trim_slices(mask: torch.Tensor) -> tuple[slice, slice]:
    """Get slices that remove outer masked out values."""
    mask = mask.any(dim=tuple(range(mask.ndim - 2)))
    row_mask, col_mask = mask.any(1).short(), mask.any(0).short()
    row_min = int(torch.argmax(row_mask))
    row_max = int(mask.size(0) - torch.argmax(row_mask.flip(0)))
    col_min = int(torch.argmax(col_mask))
    col_max = int(mask.size(1) - torch.argmax(col_mask.flip(0)))
    return slice(row_min, row_max), slice(col_min, col_max)


def apply_transform(
    data: Mapping[K, torch.Tensor],
    transform: Callable[[torch.Tensor], torch.Tensor],
    mask: torch.Tensor,
    default_values: Mapping[K, float | None] = DEFAULT_VALUES,
) -> tuple[dict[K, torch.Tensor], torch.Tensor]:
    """Apply a transformation."""
    # We need to stack the data into a single tensor, otherwise random transformations will be different for each
    # data element.
    tmp = []
    for name, x in data.items():
        x = x.clone()
        default_value = default_values.get(name, None)
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


DEFAULT_TRANSFORMS_256 = (
    torchvision.transforms.RandomAffine(
        degrees=5,
        translate=(0.05, 0.05),
        scale=(0.5, 0.6),
        fill=0.0,
        shear=(0, 5, 0, 5),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    ),
    torchvision.transforms.CenterCrop((256, 256)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomHorizontalFlip(),
)

DEFAULT_CUTS = {'axial': (80, 80, 0, 0, 0, 0), 'coronal': (0, 0, 100, 100, 0, 0), 'sagittal': (0, 0, 0, 0, 100, 100)}


class BrainwebSlices(torch.utils.data.Dataset):
    """Dataset of 2D qMRI parameter slices based on Brainweb dataset."""

    def __init__(
        self,
        folder: str | Path,
        parameters: Mapping[str, T1T2PD] = VALUES_3T,
        cuts: tuple[int, int, int, int, int, int] | Mapping[str, tuple[int, int, int, int, int, int]] = DEFAULT_CUTS,
        axis: Literal['axial', 'coronal', 'sagittal'] = 'axial',
        step: int = 1,
        transforms: Sequence[Callable[[torch.Tensor], torch.Tensor]] = DEFAULT_TRANSFORMS_256,
        what: Sequence[Literal['r1', 'r2', 'pd', 't1', 't2', 'mask', 'classes']] = ('pd', 'r1', 'r2'),
        mask_values: Mapping[str, float | None] = DEFAULT_VALUES,
    ) -> None:
        """Initialize Brainweb qMRI slice phantom."""
        self.parameters = parameters
        if isinstance(cuts, Mapping):
            try:
                self._cuts = cuts[axis]
            except KeyError:
                raise KeyError(f'Axis {axis} not found in cuts.') from None
        else:
            self._cuts = cuts
        try:
            self._axis = {'axial': 0, 'coronal': 1, 'sagittal': 2}[axis]
        except KeyError:
            raise ValueError(f'Invalid axis: {axis}.') from None
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
            mask = mask[slices]
            data = data[slices]

            result: dict[Literal['r1', 'r2', 'pd', 't1', 't2', 'mask', 'classes'], torch.Tensor] = {}
            for el in self.what:
                if el == 'r1':
                    values = torch.stack([self.parameters[k].random_r1 for k in classnames]) / 255
                    result[el] = (data.to(values) @ values)[..., 0]
                elif el == 'r2':
                    values = torch.stack([self.parameters[k].random_r2 for k in classnames]) / 255
                    result[el] = (data.to(values) @ values)[..., 0]
                elif el == 'pd':
                    values = torch.stack([self.parameters[k].random_m0 for k in classnames]) / 255
                    result[el] = (data.to(values) @ values)[..., 0]
                elif el == 't1':
                    values = torch.stack([self.parameters[k].random_r1 for k in classnames]) / 255
                    result[el] = (data.to(values) @ values)[..., 0].reciprocal()
                elif el == 't2':
                    values = torch.stack([self.parameters[k].random_r2 for k in classnames]) / 255
                    result[el] = (data.to(values) @ values)[..., 0].reciprocal()
                elif el == 'classes':
                    result[el] = data.argmax(-1)
                else:
                    raise NotImplementedError(f'what=({el},) is not implemented.')
        result, mask = apply_transform(result, self.transforms, mask)

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


def visualize_dataset(dataset) -> None:
    """
    Visualize elements from a PyTorch dataset using a slider and radio buttons.

    The dataset is expected to return a dict of 2D tensors on each access.
    The function displays an image corresponding to a selected key (via radio buttons)
    and dataset element index (via a slider).

    Parameters
    ----------
    dataset : Dataset
        A PyTorch dataset with __getitem__ and __len__ implemented. Each item should be a dict
        mapping string keys to 2D tensors.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from matplotlib.widgets import RadioButtons, Slider

    # Retrieve the keys from the first element and the total number of items.
    first_item = dataset[0]
    keys = list(first_item.keys())
    dataset_length = len(dataset)

    # Initial display settings.
    initial_index = 0
    initial_key = keys[0]

    # Create a figure and an axis for the image display.
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Get the initial image data.
    image_data = dataset[initial_index][initial_key]
    if isinstance(image_data, torch.Tensor):
        image_data = image_data.numpy()
    if np.iscomplexobj(image_data):
        image_data = np.abs(image_data)
    # Display the image; assuming a grayscale image.
    img = ax.imshow(image_data, cmap='gray')
    ax.set_title(f'Index: {initial_index}, Key: {initial_key}')

    # Create a slider axis for selecting the dataset index.
    ax_index = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider_index = Slider(
        ax=ax_index, label='Index', valmin=0, valmax=dataset_length - 1, valinit=initial_index, valstep=1
    )

    # Create a radio button axis for selecting the key.
    ax_radio = plt.axes([0.05, 0.4, 0.15, 0.15])
    radio = RadioButtons(ax_radio, keys, active=0)

    def update(_: float) -> None:
        """
        Update the displayed image based on the current slider and radio button values.
        """
        idx = int(slider_index.val)
        selected_key = radio.value_selected
        # Get the new image data from the dataset.
        data = dataset[idx][selected_key]
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        if np.iscomplexobj(data):
            data = np.abs(data)
        # Update the image display and title.
        img.set_data(data)
        ax.set_title(f'Index: {idx}, Key: {selected_key}')
        fig.canvas.draw_idle()

    # Connect the slider and radio button events to the update function.
    slider_index.on_changed(update)
    radio.on_clicked(lambda label: update(0))

    plt.show()


if __name__ == '__main__':
    # import platformdirs

    # download(CACHE_DIR, workers=4, progress=True)
    # print(HASHES)
    # print(HASHES)

    b = BrainwebSlices(CACHE_DIR)
    # import matplotlib.pyplot as plt

    # plt.imshow(b[20]['pd'].abs())
    # plt.show()
    # print()
    visualize_dataset(b)
