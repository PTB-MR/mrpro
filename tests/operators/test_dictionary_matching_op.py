"""Tests for the Dictionary Matching Operator."""

import pytest
import torch
from mrpro.operators import DictionaryMatchOp
import tempfile
import zipfile
import random
from pathlib import Path
from tests import RandomGenerator
import matplotlib.pyplot as plt

import zenodo_get

from mrpro.data import IData
from mrpro.operators import MagnitudeOp, DictionaryMatchOp
from mrpro.operators.functionals import MSE
from mrpro.operators.models import InversionRecovery

def create_data():
    # Dataset
    data_folder = Path(tempfile.mkdtemp())
    dataset = '10868350'
    zenodo_get.zenodo_get([dataset, '-r', 5, '-o', data_folder])  # r: retries
    with zipfile.ZipFile(data_folder / Path('T1 IR.zip'), 'r') as zip_ref:
        zip_ref.extractall(data_folder)
    # Create image data (IData) object with different inversion times
    ti_dicom_files = data_folder.glob('**/*.dcm')
    idata_multi_ti = IData.from_dicom_files(ti_dicom_files)

    if idata_multi_ti.header.ti is None:
        raise ValueError('Inversion times need to be defined in the DICOM files.')

    # Signalmodel and loss function
    model = MagnitudeOp() @ InversionRecovery(ti=idata_multi_ti.header.ti)
    mse = MSE(idata_multi_ti.data.abs())
    functional = mse @ model

    # Create signal dictionary
    min_value = random.uniform(0.0, 10.0)
    max_value = random.uniform(min_value, 100.0)
    number_of_values = random.randint(100, 1000)
    signal_dictionary = torch.linspace(min_value, max_value, number_of_values)
    m0 = torch.ones(1)
    return model, signal_dictionary, m0


def test_empty_dictionary():

    model, signal_dictionary, m0 = create_data()
    # Initialize DictionaryMatchOp
    operator = DictionaryMatchOp(model)

    # Test with an input signal without adding any entries to the dictionary
    input_signal = torch.tensor([10,2,5])

    try:
        operator.forward(input_signal)
        print("Test failed: Expected KeyError due to empty dictionary")
    except KeyError as e:
        print(f"Error caught as expected: {e}")



def test_mismatch_model_input_signal():

    model, signal_dictionary, m0 = create_data()

    first_dimension_model = model(m0,signal_dictionary)[0].shape[0]

    input_signal = torch.tensor([10,2,5])
    first_dimension_input_signal = input_signal.shape[0]
    assert first_dimension_model == first_dimension_input_signal, f"Mismatch in the first dimension: expected {first_dimension_model}, but got {first_dimension_input_signal}."


