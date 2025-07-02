# %%
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import mrpro
import torch


class BatchType(TypedDict):
    data: mrpro.data.KData
    target: mrpro.data.IData
    csm: mrpro.data.CsmData


class AcceleratedFastMRI(torch.utils.data.Dataset):
    def __init__(self, path: Path, acceleration: int = 4):
        self.acceleration = acceleration
        self.dataset = mrpro.phantoms.FastMRIKDataDataset(path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> BatchType:
        data = self.dataset[index]
        reconstruction = mrpro.algorithms.reconstruction.DirectReconstruction(
            data,
            csm=lambda data: mrpro.data.CsmData.from_idata_inati(data, downsampled_size=64),
        )
        csm = reconstruction.csm
        target = reconstruction(data)
        data_undersampled = data[..., :: self.acceleration, :]
        assert csm is not None  # for mypy
        if csm.data.isnan().any():
            print('csm nan')
            csm = mrpro.data.CsmData.from_kdata_inati(data, downsampled_size=64)

        return {'data': data_undersampled, 'target': target, 'csm': csm}


class MODL(torch.nn.Module):
    def __init__(self, iterations: int = 10, n_features: Sequence[int] = (64, 64, 64)):
        super().__init__()
        cnn = mrpro.nn.nets.BasicCNN(
            dim=2,
            channels_in=2,
            channels_out=2,
            batch_norm=True,
            n_features=(64, 64, 64),
        )
        self.network = mrpro.nn.Residual(mrpro.nn.ComplexAsChannel(mrpro.nn.PermutedBlock((-1, -2), cnn)))
        self.iterations = iterations
        self.regularization_weight = torch.nn.Parameter(torch.tensor(1.0))

    def prepare_dataconsistency(
        self,
        gram: mrpro.operators.LinearOperator,
        zero_filled_image: torch.Tensor,
    ):
        return mrpro.operators.ConjugateGradientOp(
            operator_factory=lambda _: gram + self.regularization_weight,
            rhs_factory=lambda regularization_image: zero_filled_image
            + self.regularization_weight * regularization_image,
        )

    def __call__(self, kdata: mrpro.data.KData, csm: mrpro.data.CsmData) -> mrpro.data.IData:
        return super().__call__(kdata, csm)

    def forward(self, kdata: mrpro.data.KData, csm: mrpro.data.CsmData) -> mrpro.data.IData:
        fourier_op = mrpro.operators.FourierOp.from_kdata(kdata)
        acquisition_op = fourier_op @ csm.as_operator()

        (image,) = acquisition_op.H(kdata.data)
        data_consistency_op = self.prepare_dataconsistency(acquisition_op.gram, image)

        for _ in range(self.iterations):
            regularization = self.network(image)
            (image,) = data_consistency_op(regularization)
            if image.isnan().any():
                raise ValueError('NaN in image')

        return mrpro.data.IData(image, header=mrpro.data.IHeader.from_kheader(kdata.header))


# %%
from tqdm import tqdm

path = Path('/echo/allgemein/resources/publicTrainingData/fastmri/brain_multicoil_train/')
dataset = AcceleratedFastMRI(path)
dataloader = torch.utils.data.DataLoader(dataset, num_workers=0, shuffle=True, collate_fn=lambda batch: batch[0])
modl = MODL().cuda()
optimizer = torch.optim.Adam(modl.parameters(), lr=1e-4)
pbar = tqdm(dataloader)
for batch in pbar:
    optimizer.zero_grad()
    kdata, csm, target = batch['data'].cuda(), batch['csm'].cuda(), batch['target'].cuda()
    pred = modl(kdata, csm)
    (loss,) = mrpro.operators.functionals.MSE(target.data)(pred.data)
    loss.backward()
    optimizer.step()
    pbar.set_postfix(loss=loss.item())
# %%
