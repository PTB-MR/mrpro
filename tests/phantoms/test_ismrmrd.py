"""Tests for the generic ISMRMRD dataset wrapper."""

from mrpro.data import KData
from mrpro.phantoms.ismrmrd import IsmrmrdDataset


def test_ismrmrd_dataset_natural_sort(tmp_path) -> None:
    """ISMRMRD files are found recursively and naturally sorted."""
    (tmp_path / 'nested').mkdir()
    for name in ('Subject10.mrd', 'Subject3.HDF5', 'Subject2.h5', 'ignore.txt', 'Subject1.h5'):
        (tmp_path / 'nested' / name).touch()
    dataset = IsmrmrdDataset(tmp_path)
    assert [file.name for file in dataset.files] == ['Subject1.h5', 'Subject2.h5', 'Subject3.HDF5', 'Subject10.mrd']


def test_ismrmrd_dataset_getitem(ismrmrd_cart) -> None:
    """The generic wrapper returns KData."""
    dataset = IsmrmrdDataset([ismrmrd_cart.filename])
    assert isinstance(dataset[0], KData)
