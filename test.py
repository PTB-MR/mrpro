import torch
from mrpro.operators.models.EPG import CardiacFingerprinting

from time import perf_counter


def test_cmrf_model():
    """Test the CMRF model."""
    acquisition_times = torch.linspace(0, 10, 705)
    te = 0.05
    model = CardiacFingerprinting(acquisition_times=acquisition_times, te=te)
    t1, t2, m0 = torch.rand(3, 256, 256)
    start = perf_counter()
    signal = model(t1, t2, m0)
    end = perf_counter()
    print(f'Time taken: {end - start}')

    for i in range(3):
        signal = model(t1, t2, m0)
    start = perf_counter()

    for i in range(5):
        signal = model(t1, t2, m0)
    end = perf_counter()
    print(f'Time taken: {(end - start)/5}')


test_cmrf_model()
