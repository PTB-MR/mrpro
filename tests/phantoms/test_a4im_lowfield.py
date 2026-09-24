"""Tests for the A4IM low-field ISMRMRD datasets."""

import hashlib
import io
import shutil
import zipfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from mrpro.phantoms.a4im_lowfield import (
    FreeMaxDataset,
    I3MDataset,
    IBTDataset,
    OSI2Dataset,
    download_free_max,
    download_i3m,
    download_ibt,
    download_osi2,
    download_zenodo_files,
    extract_zip,
)


def test_ibt_average_dataset(ismrmrd_cart, tmp_path) -> None:
    """Separate Philips average files become a leading average dimension."""
    for average in (2, 0, 1):
        shutil.copy2(ismrmrd_cart.filename, tmp_path / f'Subject01-Brain-T2_average{average}.h5')
    shutil.copy2(ismrmrd_cart.filename, tmp_path / 'Subject01-Brain-T2.h5')

    dataset = IBTDataset(tmp_path, acquisition='t2w', individual_averages=True)
    assert len(dataset) == 1
    kdata = dataset[0]
    assert kdata.shape[0] == 3
    torch.testing.assert_close(kdata.header.acq_info.idx.average[:, 0, 0, 0, 0, 0], torch.arange(3))


def test_ibt_average_dataset_rejects_duplicate_average(ismrmrd_cart, tmp_path) -> None:
    """A repeated path cannot silently create two copies of the same average."""
    filename = tmp_path / 'Subject01-Brain-T2_average0.h5'
    shutil.copy2(ismrmrd_cart.filename, filename)
    with pytest.raises(ValueError, match='Duplicate average indices'):
        IBTDataset([filename, filename], individual_averages=True)


def test_ibt_dataset_selects_acquisition_and_combined_data(ismrmrd_cart, tmp_path) -> None:
    """IBT selectors distinguish combined acquisitions from individual averages."""
    for name in ('Subject01-Brain-T1.h5', 'Subject01-Brain-T2.h5', 'Subject01-Brain-T2_average0.h5'):
        shutil.copy2(ismrmrd_cart.filename, tmp_path / name)
    dataset = IBTDataset(tmp_path, acquisition='t2w')
    assert [file.name for file in dataset.files] == ['Subject01-Brain-T2.h5']


def test_osi2_dataset_selectors(tmp_path) -> None:
    """OSI² selectors combine subset, contrast, noise correction, and subject."""
    filenames = (
        'LLR/noise_corr_on/9147/T2w_TSE_underS_R1.h5',
        'LLR/noise_corr_off/9147/T2w_TSE_underS_R1.h5',
        'LLR/noise_corr_on/9003/T2w_TSE_underS_R1p7.h5',
        'R1_R2/noise_corr_on/9147/T2w_TSE_R2.h5',
    )
    for filename in filenames:
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    dataset = OSI2Dataset(tmp_path, acquisition='t2w', noise_correction=True, subsets='llr', subjects=9147)
    assert [file.relative_to(tmp_path).as_posix() for file in dataset.files] == [
        'LLR/noise_corr_on/9147/T2w_TSE_underS_R1.h5'
    ]


@pytest.mark.parametrize(
    ('acquisition', 'filename'),
    [
        ('pdw', 'T1_3_4_0_Lab2.h5'),
        ('ir_t1w', 'T1w_IR_TSE_R1_Lab2.h5'),
        ('3t_t1w', '3T_T1w_TSE_PF.h5'),
    ],
)
def test_osi2_acquisition_names(tmp_path, acquisition, filename) -> None:
    """OSI² selectors cover the alternate names used in the archives."""
    for name in ('T1_3_4_0_Lab2.h5', 'T1w_IR_TSE_R1_Lab2.h5', '3T_T1w_TSE_PF.h5'):
        (tmp_path / name).touch()
    assert [file.name for file in OSI2Dataset(tmp_path, acquisition=acquisition).files] == [filename]


def test_i3m_dataset_selectors(tmp_path) -> None:
    """i3M mapping selectors recognize in-vivo and phantom filenames."""
    for name in (
        'Subject2-LeftKnee-TI40ms (415).h5',
        'Subject2-RightKnee-TI40ms (287).h5',
        'Subject2-RightKnee-TE50ms (473).h5',
        'Subject2-RightKnee-T1 (331).h5',
        'Lab_n01_TI40ms.h5',
    ):
        (tmp_path / name).touch()
    dataset = I3MDataset(tmp_path, acquisition='t1map', side='right')
    assert [file.name for file in dataset.files] == ['Subject2-RightKnee-TI40ms (287).h5']
    assert [file.name for file in I3MDataset(tmp_path, acquisition='t1map').files] == [
        'Lab_n01_TI40ms.h5',
        'Subject2-LeftKnee-TI40ms (415).h5',
        'Subject2-RightKnee-TI40ms (287).h5',
    ]


@pytest.mark.parametrize(
    ('kwargs', 'expected'),
    [
        ({'acquisition': 't2w', 'side': 'left'}, ['t2_tse_sag_Left.mrd']),
        ({'acquisition': 't1map', 'side': 'right'}, ['t1map_MOLLI_Right.mrd']),
    ],
)
def test_free_max_dataset_selectors(tmp_path, kwargs, expected) -> None:
    """Free.Max selectors recognize weighted and mapping acquisitions by side."""
    for name in (
        't2_tse_sag_Left.mrd',
        't2_tse_sag_Left_DeepResolve.mrd',
        't2_tse_sag_Right.mrd',
        't1map_MOLLI_Right.mrd',
    ):
        (tmp_path / name).touch()
    dataset = FreeMaxDataset(tmp_path, **kwargs)
    assert [file.name for file in dataset.files] == expected


def test_download_zenodo_files(monkeypatch, tmp_path) -> None:
    """Downloads are verified, safely extracted, and cached by a manifest."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('dataset/scan.h5', b'raw data')
    content = buffer.getvalue()
    metadata = {
        'files': [
            {
                'key': 'data.zip',
                'size': len(content),
                'checksum': f'md5:{hashlib.md5(content, usedforsecurity=False).hexdigest()}',
                'links': {'self': 'https://example.test/data.zip'},
            }
        ]
    }
    metadata_response = Mock()
    metadata_response.json.return_value = metadata
    download_response = Mock()
    download_response.iter_content.return_value = [content]
    get = Mock(side_effect=(metadata_response, download_response, metadata_response))
    monkeypatch.setattr('mrpro.phantoms.a4im_lowfield.requests.get', get)
    download_zenodo_files(1, ['data.zip', 'data.zip'], tmp_path)
    download_zenodo_files(1, ['data.zip'], tmp_path)

    assert (tmp_path / 'dataset/scan.h5').read_bytes() == b'raw data'
    assert not (tmp_path / 'data.zip').exists()
    assert get.call_count == 3


def test_extract_zip_rejects_parent_path(tmp_path) -> None:
    """ZIP members cannot escape the output directory."""
    archive = tmp_path / 'unsafe.zip'
    with zipfile.ZipFile(archive, 'w') as zip_file:
        zip_file.writestr('../outside', b'unsafe')
    with pytest.raises(ValueError, match='outside output directory'):
        extract_zip(archive, tmp_path / 'output')


@pytest.mark.parametrize(
    ('download', 'kwargs', 'record', 'filenames'),
    [
        (download_osi2, {'subsets': ('llr', 'loraks')}, 19661402, ['LLR.zip', 'R1_R2.zip']),
        (download_ibt, {'subjects': (10, 1, 1)}, 18847561, ['Subject 01.zip', 'Subject 10.zip']),
        (
            download_i3m,
            {'subjects': 2, 'phantom': True, 'documentation': True},
            20700288,
            [
                'Subject 2.zip',
                'ACRphantomData.zip',
                'Useful information for data users.pdf',
                'In vivo imaging with MRILab\N{RIGHT SINGLE QUOTATION MARK}s Physio 1 Scanner.pdf',
            ],
        ),
        (download_free_max, {'subjects': 3, 'phantoms': True}, 20516472, ['Subject3.zip', 'PHANTOMS.zip']),
    ],
)
def test_dataset_download_selection(monkeypatch, tmp_path, download, kwargs, record, filenames) -> None:
    """Dataset selectors map to the correct Zenodo archives."""
    selected = {}

    def mock_download(selected_record, selected_filenames, output_directory, **_):
        selected['record'] = selected_record
        selected['filenames'] = selected_filenames
        return Path(output_directory)

    monkeypatch.setattr('mrpro.phantoms.a4im_lowfield.download_zenodo_files', mock_download)
    assert download(output_directory=tmp_path, **kwargs) == tmp_path
    assert selected == {'record': record, 'filenames': filenames}
