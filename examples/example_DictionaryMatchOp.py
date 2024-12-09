# %% [markdown]
# # QMRI Challenge ISMRM 2024 - T1 mapping

# %%
# Imports
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import torch
import zenodo_get
from einops import rearrange
from mrpro.data import IData
from mrpro.operators import MagnitudeOp, Operator
from mrpro.operators.functionals import MSEDataDiscrepancy
from mrpro.operators.models import InversionRecovery
from typing_extensions import Self, TypeVarTuple, Unpack

# %% [markdown]
# ### Overview
# The dataset consists of images obtained at 10 different inversion times using a turbo spin echo sequence. Each
# inversion time is saved in a separate DICOM file. In order to obtain a T1 map, we are going to:
# - download the data from Zenodo
# - read in the DICOM files (one for each inversion time) and combine them in an IData object
# - define a signal model and data loss (mean-squared error) function
# - find good starting values for each pixel
# - carry out a fit using ADAM from PyTorch

# %% [markdown]
# ### Get data from Zenodo

# %%
data_folder = Path(tempfile.mkdtemp())
dataset = '10868350'
zenodo_get.zenodo_get([dataset, '-r', 5, '-o', data_folder])  # r: retries
with zipfile.ZipFile(data_folder / Path('T1 IR.zip'), 'r') as zip_ref:
    zip_ref.extractall(data_folder)

# %% [markdown]
# ### Create image data (IData) object with different inversion times
# %%
ti_dicom_files = data_folder.glob('**/*.dcm')
idata_multi_ti = IData.from_dicom_files(ti_dicom_files)

if idata_multi_ti.header.ti is None:
    raise ValueError('Inversion times need to be defined in the DICOM files.')

# %%
# Let's have a look at some of the images
fig, axes = plt.subplots(1, 3, squeeze=False)
for idx, ax in enumerate(axes.flatten()):
    ax.imshow(torch.abs(idata_multi_ti.data[idx, 0, 0, :, :]))
    ax.set_title(f'TI = {idata_multi_ti.header.ti[idx]:.0f}ms')

# %% [markdown]
# ### Signal model and loss function
# We use the model $q$
#
# $q(TI) = M_0 (1 - e^{-TI/T1})$
#
# with the equilibrium magnetization $M_0$, the inversion time $TI$, and $T1$. We have to keep in mind that the DICOM
# images only contain the magnitude of the signal. Therefore, we need $|q(TI)|$:

# %%
model = MagnitudeOp() @ InversionRecovery(ti=idata_multi_ti.header.ti)

# %% [markdown]
# As a loss function for the optimizer, we calculate the mean-squared error between the image data $x$ and our signal
# model $q$.
# %%
mse = MSEDataDiscrepancy(idata_multi_ti.data.abs())

# %% [markdown]
# Now we can simply combine the two into a functional to solve
#
# $ \min_{M_0, T1} || |q(M_0, T1, TI)| - x||_2^2$
# %%
functional = mse @ model

# %% [markdown]
# ### Starting values for the fit
# We are trying to minimize a non-linear function $q$. There is no guarantee that we reach the global minimum, but we
# can end up in a local minimum.
#
# To increase our chances of reaching the global minimum, we can ensure that our starting
# values are already close to the global minimum. We need a good starting point for each pixel.
#
# One option to get a good starting point is to calculate the signal curves for a range of T1 values and then check
# for each pixel which of these signal curves fits best. This is similar to what is done for MR Fingerprinting. So we
# are going to:
# - define a list of realistic T1 values (we call this a dictionary of T1 values)
# - calculate the signal curves corresponding to each of these T1 values
# - compare the signal curves to the signals of each voxel (we use the maximum of the dot-product as a metric of how
# well the signals fit to each other)

# %%
# Define 100 T1 values between 100 and 3000 ms
t1_dictionary = torch.linspace(0.1, 3, 100).double()


# Calculate the signal corresponding to each of these T1 values. We set M0 to 1, but this is arbitrary because M0 is
# just a scaling factor and we are going to normalize the signal curves.
(signal_dictionary,) = model(torch.ones(1), t1_dictionary)
signal_dictionary = signal_dictionary + 0j
# signal_dictionary = signal_dictionary.to(dtype=torch.complex128)
vector_norm = torch.linalg.vector_norm(signal_dictionary, dim=0)
signal_dictionary /= vector_norm

# Calculate the dot-product and select for each voxel the T1 values that correspond to the maximum of the dot-product
n_y, n_x = idata_multi_ti.data.shape[-2:]
data = idata_multi_ti.data.to(torch.complex128)
dot_product = torch.mm(rearrange(data, 'other 1 z y x->(z y x) other'), signal_dictionary)
# print(signal_dictionary)
idx_best_match = torch.argmax(torch.abs(dot_product), dim=1)
# print(torch.abs(dot_product))
t1_start = rearrange(t1_dictionary[idx_best_match], '(y x)->1 1 y x', y=n_y, x=n_x)


Tin = TypeVarTuple('Tin')


class DictionaryMatchOp(Operator[torch.Tensor, tuple[*Tin]]):
    def __init__(self, generating_function: Callable[[Unpack[Tin]], tuple[torch.Tensor,]]):
        super().__init__()
        self._f = generating_function
        self.x: list[torch.Tensor] = []
        self.y = torch.tensor([])

    def append(self, *x: Unpack[Tin]) -> Self:
        (newy,) = self._f(*x)
        newy = newy / torch.linalg.norm(newy, dim=0, keepdim=True)
        newy = newy.flatten(start_dim=1)
        newx = [x.flatten() for x in torch.broadcast_tensors(*x)]
        if not self.x:
            self.x = newx
            self.y = newy
            return self
        self.x = [torch.cat(old, new) for old, new in zip(self.x, newx, strict=True)]
        self.y = torch.cat((self.y, newy))
        return

    def forward(self, input_signal: torch.Tensor) -> tuple[Unpack[Tin]]:
        similar = einops.einsum(input_signal, self.y, 't ..., t idx -> idx ...')
        idx = torch.argmax(similar, dim=0)
        match = [x[idx] for x in self.x]
        return match


dict_match_op = DictionaryMatchOp(model)
dictionary = dict_match_op.append(torch.ones(1), t1_dictionary)
t1_start_new = dict_match_op.forward(idata_multi_ti.rss().double())[1]
(t1_start == t1_start_new).all()
# %%


# %%
import matplotlib.pyplot as plt

plt.matshow(t1_start.real.squeeze())
# %%
import matplotlib.pyplot as plt

plt.matshow(t1_start_new.real.squeeze())
# %%
