"""Brainweb Phantom."""

import concurrent.futures
import gzip
import hashlib
import io
import re
from collections.abc import Sequence
from os import PathLike
from pathlib import Path

import h5py
import numpy as np
import requests
from tqdm import tqdm

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
URL = 'http://brainweb.bic.mni.mcgill.ca/brainweb/anatomic_normal_20.html'

CLASSES = ['gry', 'what', 'csf', 'mrw', 'dura', 'fat', 'fat2', 'mus', 'm-s', 'ves', 'back', 'skl']


def load_url(url: str, timeout: float = 60) -> bytes:
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
        futures = {
            executor.submit(
                load_url,
                f'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?do_download_alias=subject{subject}_{c}&format_value=raw_short&zip_value=gnuzip&download_for_real=%5BStart+download%21%5D',
            ): c
            for c in CLASSES
        }

        downloaded_data = {}
        for future in concurrent.futures.as_completed(futures):
            c = futures[future]
            downloaded_data[c] = future.result()
            progressbar.update(1)

    values = norm_([unpack(downloaded_data[c], shape=(362, 434, 362), dtype=np.uint16) for c in CLASSES])

    with h5py.File(outfilename, 'w') as f:
        f.create_dataset('classes', values.shape, dtype=values.dtype, data=values)
        f.attrs['classnames'] = list(CLASSES)
        f.attrs['subject'] = int(subject)


def download(output_directory: str | PathLike, workers: int = 4, progress: bool = False) -> None:
    """Download brainweb data with subjects in series and class files in parallel."""
    page = requests.get(URL, timeout=5)
    subjects = re.findall(r'option value=(\d*)>', page.text)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    total_steps = len(subjects) * len(CLASSES)
    with tqdm(total=total_steps, desc='Downloading brainweb data', disable=not progress) as progressbar:
        for subject in subjects:
            outfilename = output_directory / f's{subject}.h5'
            if outfilename.exists():
                md5 = hashlib.file_digest(outfilename.open('rb'), 'md5').hexdigest()
                if md5 == HASHES[subject]:
                    progressbar.update(len(CLASSES))
                    continue
            download_subject(subject, outfilename, workers, progressbar)


if __name__ == '__main__':
    import tempfile

    import platformdirs

    with tempfile.TemporaryDirectory() as tmpdirname:
        cache = platformdirs.user_cache_dir('mrpro')
        download(cache, workers=2)
