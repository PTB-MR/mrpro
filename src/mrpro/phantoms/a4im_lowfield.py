"""Download and select raw low-field ISMRMRD data from the A4IM project."""

import hashlib
import json
import re
import zipfile
from collections import defaultdict
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Literal

import platformdirs
import requests
import torch
from tqdm import tqdm

from mrpro.data.KData import KData
from mrpro.phantoms.ismrmrd import IsmrmrdDataset, natural_sort_key

# ~/.cache/mrpro/a4im_lowfield on Linux, %AppData%\Local\mrpro\a4im_lowfield on Windows
CACHE_DIR_A4IM_LOWFIELD = Path(platformdirs.user_cache_dir('mrpro')) / 'a4im_lowfield'

OSI2Subset = Literal['llr', 'partial_fourier', 'loraks', 'coil_comparison', 'repeatability']

_OSI2_FILES: dict[OSI2Subset, str] = {
    'llr': 'LLR.zip',
    'partial_fourier': 'partial_fourier.zip',
    'loraks': 'R1_R2.zip',
    'coil_comparison': 'coil_comparison.zip',
    'repeatability': 'repeatability.zip',
}

DatasetPath = str | PathLike | Sequence[str | PathLike]
A4IMAcquisition = Literal['pdw', 't1w', 't2w', 't1map', 't2map']
KneeSide = Literal['left', 'right']
OSI2Acquisition = Literal['pdw', 't2w', 'ir_t1w', '3t_t1w']


class OSI2Dataset(IsmrmrdDataset):
    """Raw 47 mT brain data acquired with LUMC's OSI² ONE scanner.

    The in-house Halbach system uses a Kea2 spectrometer, Barthel RF amplifier,
    AE Techron gradient amplifiers, and normally a single-channel dome-helix
    transmit/receive head coil. All scans are 3D Cartesian turbo spin echo with
    a 150 x 136 x 38 matrix, 225 x 205 x 190 mm field of view, 1.5 x 1.5 x 5 mm
    resolution, and 20 kHz bandwidth. PDw, T2w, IR-T1w, and 3T-like T1w use
    TE/TR 16/600, 20/2500, 20/1200, and 16/900 ms; ETL 8/15/8/10; and
    center-out/linear/center-out/center-out echo ordering. The inversion times
    are 90--91 ms for IR-T1w and 250 ms for 3T-like T1w.

    The 352 files comprise ``llr`` Poisson-disk scans (3 subjects; R=1, 1.7,
    and 2, plus PF comparison files), ``partial_fourier`` (4 subjects),
    ``loraks`` elliptical R=1/random R=2 scans (4 subjects), ``coil_comparison``
    dome-helix/open-coil scans (9 subjects), and ``repeatability`` scans (5
    subjects; Lab1--3, ICU1, and Van1). Every file is supplied with and without
    noise-correlation correction. The published XML omits required trajectory
    and patient-position fields; :class:`~mrpro.data.KData` repairs these in
    memory and leaves the files unchanged. See Zenodo record 19661402.
    """

    def __init__(
        self,
        path: DatasetPath = CACHE_DIR_A4IM_LOWFIELD / 'osi2_one',
        acquisition: OSI2Acquisition | None = None,
        noise_correction: bool | None = None,
        subsets: OSI2Subset | Sequence[OSI2Subset] | None = None,
        subjects: int | Sequence[int] | None = None,
    ) -> None:
        """Select OSI² ONE files.

        Parameters
        ----------
        path
            Extracted dataset directory, raw file, or sequence of raw files.
        acquisition
            ``pdw``, ``t2w``, ``ir_t1w``, or ``3t_t1w``; all if ``None``.
        noise_correction
            Select corrected or uncorrected data; both if ``None``.
        subsets
            One or more experiment subsets; all if ``None``.
        subjects
            One or more anonymized numeric directory IDs; all if ``None``.
        """
        super().__init__(path)
        if acquisition is not None:
            patterns = {
                'pdw': r'^(?:t1w_(?!ir_)|t1_3_4_0_)',
                't2w': r'^(?:t2w_|t2_3_4_0_)',
                'ir_t1w': r'^(?:ir_t1w_|t1w_ir_)',
                '3t_t1w': r'^3t_t1w_',
            }
            self.files = tuple(file for file in self.files if re.search(patterns[acquisition], file.stem, re.I))
        if noise_correction is not None:
            directory = f'noise_corr_{"on" if noise_correction else "off"}'
            self.files = tuple(file for file in self.files if directory in {part.casefold() for part in file.parts})
        if subsets is not None:
            subsets = (subsets,) if isinstance(subsets, str) else tuple(subsets)
            roots = {_OSI2_FILES[subset].removesuffix('.zip').casefold() for subset in subsets}
            self.files = tuple(
                file for file in self.files if roots.intersection(part.casefold() for part in file.parts)
            )
        if subjects is not None:
            subjects = (subjects,) if isinstance(subjects, int) else tuple(subjects)
            subject_ids = {str(subject) for subject in subjects}
            self.files = tuple(file for file in self.files if subject_ids.intersection(file.parts))


class IBTDataset(IsmrmrdDataset):
    """Raw 0.6 T brain data from a ramped-down Philips Ingenia Ambition X.

    The 45 mT/m, 200 T/m/s gradient system and a nominal 14-channel head coil
    were used for ten volunteers. The qualitative protocol comprises 2D
    Cartesian FSE PDw/T1w/T2w scans (three slices, 232 x 176 matrix, about
    1 x 1 x 5 mm, ETL 8) with TE/TR 5.125/3000, 5.125/600, and 70/3000 ms and
    3/14/3 averages. Quantitative scans comprise 11 inversion-delay b-TFE T1
    acquisitions (TR/TE 4.74/2.08 ms) and 8--10 GraSE/RARE T2 echoes
    (TE 21--118 ms, TR 1000 ms, EPI factor 7) with three averages. Subject 1
    lacks the last two T2 echoes. See Zenodo record 18847561.

    Files named ``*_averageN.h5`` are grouped by scan. Each item contains a
    leading average dimension, and ``header.acq_info.idx.average`` records the
    corresponding average numbers. Otherwise, the already-combined files are
    loaded. Philips metadata report either 14 or 15 active channels by scan.
    """

    def __init__(
        self,
        path: DatasetPath = CACHE_DIR_A4IM_LOWFIELD / 'ibt',
        acquisition: A4IMAcquisition | None = None,
        individual_averages: bool = False,
    ) -> None:
        """Select IBT files.

        Parameters
        ----------
        path
            Extracted dataset directory, raw file, or sequence of raw files.
        acquisition
            Weighted image or mapping acquisition; all if ``None``.
        individual_averages
            Stack the ``averageN`` files instead of loading combined files.
        """
        super().__init__(path)
        if acquisition is not None:
            patterns = {
                'pdw': r'-pd(?:_average\d+)?$',
                't1w': r'-t1(?:_average\d+)?$',
                't2w': r'-t2(?:_average\d+)?$',
                't1map': r'-t1map_set\d+$',
                't2map': r'-t2map_contrast\d+(?:_average\d+)?$',
            }
            self.files = tuple(
                file for file in self.files if re.search(patterns[acquisition], file.stem, re.IGNORECASE)
            )
        pattern = re.compile(r'(.+?)[_-]average(\d+)$', flags=re.IGNORECASE)
        self.files = tuple(
            file for file in self.files if (pattern.fullmatch(file.stem) is not None) is individual_averages
        )
        self.individual_averages = individual_averages
        groups: defaultdict[tuple[Path, str], list[tuple[int, Path]]] = defaultdict(list)
        for file in self.files:
            match = pattern.fullmatch(file.stem)
            if match is not None:
                groups[(file.parent, match.group(1))].append((int(match.group(2)), file))

        for (parent, name), group in groups.items():
            average_indices = [average for average, _ in group]
            if len(set(average_indices)) != len(average_indices):
                raise ValueError(f'Duplicate average indices found for {parent / name}.')
        ordered_groups = [
            sorted(group, key=lambda item: item[0])
            for _, group in sorted(groups.items(), key=lambda item: natural_sort_key(item[1][0][1]))
        ]
        self.groups = tuple(tuple(file for _, file in group) for group in ordered_groups)
        self.average_indices = tuple(tuple(average for average, _ in group) for group in ordered_groups)

    def __len__(self) -> int:
        """Get the number of selected scans."""
        return len(self.groups) if self.individual_averages else super().__len__()

    def __getitem__(self, idx: int) -> KData:
        """Load one scan, stacking individual averages when requested."""
        if not self.individual_averages:
            return super().__getitem__(idx)
        items = [self.load(file) for file in self.groups[idx]]
        kdata = items[0].stack(*items[1:])
        average_shape = (len(items), *([1] * (kdata.data.ndim - 1)))
        kdata.header.acq_info.idx.average = torch.tensor(self.average_indices[idx]).reshape(average_shape)
        return kdata


class I3MDataset(IsmrmrdDataset):
    """Raw knee and ACR-phantom data from i3M's 72 mT Physio 1 scanner.

    The 250 kg Halbach system uses 25/40/40 mT/m gradients, open MaRCoS/MaRGE
    control, and a single-channel transmit/receive Litz-wire coil. Seven
    volunteers were scanned with 3D RARE PDw/T1w/T2w (ETL 4/6/10,
    center-out/center-out/out-center ordering) and 3D STIR/RARE T1/T2 mapping.
    In-vivo scans have a 240 x 180 x 160 mm field of view, 160 x 120 x 32 matrix,
    1.5 x 1.5 x 5 mm resolution, and 40 kHz bandwidth. The qualitative protocols
    use 1/8/2 averages, already combined in each file; mapping samples TI
    0--500 ms (7 files) or TE 10--100 ms (5 files), with one average each.

    Subjects 1 and 5 contain qualitative scans, 2--4 contain both protocols, and
    6--7 contain mapping scans. The optional ACR phantom has five lab and three
    office repeats with a 100 x 120 x 30 matrix and 150 x 180 x 150 mm field of
    view. See Zenodo record 20700288.
    """

    def __init__(
        self,
        path: DatasetPath = CACHE_DIR_A4IM_LOWFIELD / 'i3m',
        acquisition: A4IMAcquisition | None = None,
        side: KneeSide | None = None,
    ) -> None:
        """Select i3M files.

        Parameters
        ----------
        path
            Extracted dataset directory, raw file, or sequence of raw files.
        acquisition
            Weighted image or mapping acquisition; all if ``None``.
        side
            In-vivo knee side. A side selection excludes ACR phantom files.
        """
        super().__init__(path)
        if acquisition is not None:
            patterns = {
                'pdw': r'(?:-pd \(|_pdw$)',
                't1w': r'(?:-t1 \(|_t1w$)',
                't2w': r'(?:-t2 \(|_t2w$)',
                't1map': r'(?:-ti\d+ms \(|_ti\d+ms$)',
                't2map': r'(?:-te\d+ms \(|_te\d+ms$)',
            }
            self.files = tuple(
                file for file in self.files if re.search(patterns[acquisition], file.stem, re.IGNORECASE)
            )
        if side is not None:
            self.files = tuple(file for file in self.files if f'-{side}knee-' in file.stem.casefold())


class FreeMaxDataset(IsmrmrdDataset):
    """Raw knee and phantom data from a Siemens MAGNETOM Free.Max at 0.55 T.

    The 80 cm-bore DryCool superconducting system provides 22--26 mT/m gradients
    and 45--55 T/m/s slew rate. Twenty volunteers were scanned bilaterally with a
    flexible six-element Contour S coil. The raw subject files nevertheless
    report 15 active receive channels (some phantom files report 6). The
    sagittal protocol contains 2D
    Cartesian TSE PDw/T1w/T2w scans (TE/TR 28/3070, 12/490, and 78/3040 ms;
    ETL 9/4/12; 112 x 112 matrix; 10--14 slices) and 2D MOLLI (3-3-5) T1 and
    T2-prepared bSSFP mapping (11 and 6 sets, 128 matrix, 1.6 x 1.6 x 8 mm, six
    slices). The descriptor reports 0.8 x 0.8 x 5 mm for the TSE data, whereas
    the raw reconstruction metadata imply about 1.52 x 1.52 x 5 mm. All
    acquisitions have one average.

    Each of the 20 subjects and 3 phantoms contains 16 files. The 6 qualitative
    scans intended for vendor Deep Resolve processing are excluded, leaving 10
    raw files per subject or phantom. The published
    ``Subject9/raw/pd_tse_sag_Right.mrd`` file is truncated and cannot be loaded.
    See Zenodo record 20516472.
    """

    def __init__(
        self,
        path: DatasetPath = CACHE_DIR_A4IM_LOWFIELD / 'free_max',
        acquisition: A4IMAcquisition | None = None,
        side: KneeSide | None = None,
    ) -> None:
        """Select Free.Max files.

        Parameters
        ----------
        path
            Extracted dataset directory, raw file, or sequence of raw files.
        acquisition
            Weighted image or mapping acquisition; all if ``None``.
        side
            Knee or phantom side; both if ``None``.
        """
        super().__init__(path)
        self.files = tuple(file for file in self.files if '_deepresolve' not in file.stem.casefold())
        if acquisition is not None:
            prefixes = {
                'pdw': 'pd_tse_',
                't1w': 't1_tse_',
                't2w': 't2_tse_',
                't1map': 't1map_',
                't2map': 't2map_',
            }
            self.files = tuple(file for file in self.files if file.name.casefold().startswith(prefixes[acquisition]))
        if side is not None:
            self.files = tuple(file for file in self.files if f'_{side}' in file.stem.casefold())


def extract_zip(archive: str | PathLike, output_directory: str | PathLike) -> list[str]:
    """Extract a ZIP archive while rejecting paths outside the destination."""
    archive, output_directory = Path(archive), Path(output_directory)
    root = output_directory.resolve()
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            destination = (output_directory / member.filename).resolve()
            if not destination.is_relative_to(root):
                raise ValueError(f'Archive member points outside output directory: {member.filename}')
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f'Archive contains a symbolic link: {member.filename}')
        zip_file.extractall(output_directory)
        return [member.filename for member in zip_file.infolist() if not member.is_dir()]


def download_zenodo_files(
    record: int | str,
    filenames: str | Sequence[str],
    output_directory: str | PathLike,
    *,
    progress: bool = False,
) -> Path:
    """Download, verify, and extract files from a Zenodo record.

    ZIP archives are removed after extraction. A small manifest prevents completed
    files from being downloaded again.

    Parameters
    ----------
    record
        Zenodo record identifier.
    filenames
        Filename or filenames from the record to download.
    output_directory
        Directory in which to extract or store the files.
    progress
        Show byte-level download progress.
    """
    filenames = tuple(dict.fromkeys((filenames,) if isinstance(filenames, str) else filenames))
    if not filenames:
        raise ValueError('At least one filename must be selected.')
    unsafe_filenames = [filename for filename in filenames if Path(filename).name != filename or '\\' in filename]
    if unsafe_filenames:
        raise ValueError(f'Filenames must not contain directory components: {unsafe_filenames}')

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    manifest_directory = output / '.mrpro-downloads'
    manifest_directory.mkdir(exist_ok=True)

    metadata_response = requests.get(
        f'https://zenodo.org/api/records/{record}', timeout=30, headers={'User-Agent': 'mrpro'}
    )
    try:
        metadata_response.raise_for_status()
        available = {file['key']: file for file in metadata_response.json()['files']}
    finally:
        metadata_response.close()
    missing = sorted(set(filenames) - available.keys())
    if missing:
        raise ValueError(f'Files not found in Zenodo record {record}: {missing}')

    for filename in filenames:
        metadata = available[filename]
        algorithm, expected_checksum = metadata['checksum'].split(':', maxsplit=1)
        manifest_path = manifest_directory / f'{filename}.json'
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                members = manifest.get('members', [])
                if manifest.get('checksum') == metadata['checksum'] and all(
                    (output / member).exists() for member in members
                ):
                    continue
            except json.JSONDecodeError:
                pass

        temporary_path = output / f'.{filename}.part'
        digest = hashlib.new(algorithm, usedforsecurity=False)
        response = requests.get(
            metadata['links']['self'], stream=True, timeout=(10, 120), headers={'User-Agent': 'mrpro'}
        )
        try:
            response.raise_for_status()
            with (
                temporary_path.open('wb') as stream,
                tqdm(
                    total=metadata['size'], desc=filename, unit='B', unit_scale=True, disable=not progress
                ) as progressbar,
            ):
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)
                        digest.update(chunk)
                        progressbar.update(len(chunk))
        finally:
            response.close()
        if digest.hexdigest() != expected_checksum:
            temporary_path.unlink(missing_ok=True)
            raise OSError(f'Checksum verification failed for {filename}.')

        destination = output / filename
        temporary_path.replace(destination)
        if destination.suffix.lower() == '.zip':
            members = extract_zip(destination, output)
            destination.unlink()
        else:
            members = [filename]
        manifest_path.write_text(json.dumps({'checksum': metadata['checksum'], 'members': members}))
    return output


def download_osi2(
    output_directory: str | PathLike = CACHE_DIR_A4IM_LOWFIELD / 'osi2_one',
    subsets: OSI2Subset | Sequence[OSI2Subset] = 'llr',
    *,
    documentation: bool = False,
    progress: bool = False,
) -> Path:
    """Download selected OSI² ONE 47 mT datasets from Zenodo record 19661402.

    Available subsets are ``llr``, ``partial_fourier``, ``loraks``,
    ``coil_comparison``, and ``repeatability``.
    """
    subsets = (subsets,) if isinstance(subsets, str) else tuple(subsets)
    unknown = sorted(set(subsets) - _OSI2_FILES.keys())
    if unknown:
        raise ValueError(f'Unknown OSI² subsets: {unknown}. Available subsets: {list(_OSI2_FILES)}')
    filenames = [_OSI2_FILES[subset] for subset in subsets]
    if documentation:
        filenames.append('Low_field_MRI_dataset_with_OSI_ONE.pdf')
    return download_zenodo_files(19661402, filenames, output_directory, progress=progress)


def _subject_files(subjects: int | Sequence[int], maximum: int, template: str) -> list[str]:
    """Validate subject numbers and format archive names."""
    subjects = [subjects] if isinstance(subjects, int) else sorted(set(subjects))
    invalid = [subject for subject in subjects if not 1 <= subject <= maximum]
    if invalid:
        raise ValueError(f'Subject numbers must be between 1 and {maximum}, got {invalid}')
    return [template.format(subject=subject) for subject in subjects]


def download_ibt(
    output_directory: str | PathLike = CACHE_DIR_A4IM_LOWFIELD / 'ibt',
    subjects: int | Sequence[int] = 1,
    *,
    progress: bool = False,
) -> Path:
    """Download selected Philips 0.6 T IBT brain subjects from record 18847561."""
    filenames = _subject_files(subjects, 10, 'Subject {subject:02d}.zip')
    return download_zenodo_files(18847561, filenames, output_directory, progress=progress)


def download_i3m(
    output_directory: str | PathLike = CACHE_DIR_A4IM_LOWFIELD / 'i3m',
    subjects: int | Sequence[int] = 1,
    *,
    phantom: bool = False,
    documentation: bool = False,
    progress: bool = False,
) -> Path:
    """Download selected i3M Physio 1 72 mT knee subjects from record 20700288."""
    filenames = _subject_files(subjects, 7, 'Subject {subject}.zip')
    if phantom:
        filenames.append('ACRphantomData.zip')
    if documentation:
        filenames.extend(
            (
                'Useful information for data users.pdf',
                'In vivo imaging with MRILab\N{RIGHT SINGLE QUOTATION MARK}s Physio 1 Scanner.pdf',
            )
        )
    return download_zenodo_files(20700288, filenames, output_directory, progress=progress)


def download_free_max(
    output_directory: str | PathLike = CACHE_DIR_A4IM_LOWFIELD / 'free_max',
    subjects: int | Sequence[int] = 1,
    *,
    phantoms: bool = False,
    progress: bool = False,
) -> Path:
    """Download selected Free.Max 0.55 T subjects from Zenodo record 20516472."""
    filenames = _subject_files(subjects, 20, 'Subject{subject}.zip')
    if phantoms:
        filenames.append('PHANTOMS.zip')
    return download_zenodo_files(20516472, filenames, output_directory, progress=progress)
