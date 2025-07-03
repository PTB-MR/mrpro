# %%
# %matplotlib inline
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import matplotlib.axes
import matplotlib.pyplot as plt
import mrpro
import torch
from tqdm import tqdm


class BatchType(TypedDict):
    data: mrpro.data.KData
    target: mrpro.data.IData
    csm: mrpro.data.CsmData


class AcceleratedFastMRI(torch.utils.data.Dataset):
    def __init__(self, path: Path, acceleration: float = 16, noise_level: float = 0.2):
        self.acceleration = acceleration
        self.dataset = mrpro.phantoms.FastMRIKDataDataset(path)
        self.noise_level = noise_level

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> BatchType:
        data = self.dataset[index]
        data = data.remove_readout_os()
        data.data /= data.data.std()
        reconstruction = mrpro.algorithms.reconstruction.DirectReconstruction(
            data,
            csm=lambda data: mrpro.data.CsmData.from_idata_inati(data, downsampled_size=64),
        )
        csm = reconstruction.csm
        target = reconstruction(data)

        n = max(data.data.shape[-2:])
        distance = (torch.linspace(-1, 1, n)[:, None] ** 2 + torch.linspace(-1, 1, n) ** 2).sqrt()
        random = 0.1 / (distance + 0.1) + torch.rand_like(distance)
        threshold = torch.kthvalue(random.ravel(), int(n**2 * (1 - 1 / self.acceleration))).values
        undersampling_mask = mrpro.utils.pad_or_crop(random > threshold, data.data.shape[-2:])
        data_undersampled = data[..., undersampling_mask].rearrange('k ... 1 -> ... k')

        noise = mrpro.utils.RandomGenerator(seed=index).randn_like(data_undersampled.data)
        data_undersampled.data += self.noise_level * noise

        assert csm is not None  # for mypy
        return {'data': data_undersampled, 'target': target, 'csm': csm}


class MODL(torch.nn.Module):
    def __init__(self, iterations: int = 10, n_features: Sequence[int] = (64, 64, 64, 64)):
        super().__init__()
        cnn = mrpro.nn.nets.BasicCNN(
            dim=2,
            channels_in=2,
            channels_out=2,
            n_features=n_features,
        )
        self.network = mrpro.nn.Residual(mrpro.nn.ComplexAsChannel(mrpro.nn.PermutedBlock((-1, -2), cnn)))
        self.network = torch.compile(self.network, dynamic=True, fullgraph=True)
        self.iterations = iterations
        self.regularization_weight = torch.nn.Parameter(torch.tensor(1.0))

    def _prepare_dataconsistency(
        self,
        gram: mrpro.operators.LinearOperator,
        zero_filled_image: torch.Tensor,
    ) -> mrpro.operators.ConjugateGradientOp:
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
        data_consistency_op = self._prepare_dataconsistency(acquisition_op.gram, image)

        for _ in range(self.iterations):
            regularization = self.network(image)
            (image,) = data_consistency_op(regularization)

        return mrpro.data.IData(image, header=mrpro.data.IHeader.from_kheader(kdata.header))


def plot(batch: BatchType, prediction: mrpro.data.IData):
    target = batch['target'].rss().cpu().squeeze()
    direct = mrpro.algorithms.reconstruction.DirectReconstruction(batch['data'], csm=batch['csm'])(batch['data'])
    direct = direct.rss().cpu().squeeze()
    direct *= target.std() / direct.std()
    sense = mrpro.algorithms.reconstruction.IterativeSENSEReconstruction(batch['data'], csm=batch['csm'])(batch['data'])
    sense = sense.rss().cpu().squeeze()
    prediction_ = prediction.rss().cpu().squeeze().detach()

    ssim = mrpro.operators.functionals.SSIM(mrpro.utils.pad_or_crop(target[None], (320, 320)))

    def show(ax: matplotlib.axes.Axes, data: torch.Tensor, label: str):
        data = mrpro.utils.pad_or_crop(data, (320, 320))
        ax.imshow(data, vmin=0, vmax=target.max().item(), cmap='gray')
        if label != 'Ground Truth':
            (ssim_value,) = ssim(data[None])
            ax.text(
                0.98,
                0.1,
                f'{ssim_value.item():.2f}',
                color='white',
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes,
            )
        ax.set_title(label)
        ax.set_axis_off()

    fig, ax = plt.subplots(1, 4)
    show(ax[0], direct, 'Direct')
    show(ax[1], sense, 'CG-SENSE')
    show(ax[2], prediction_, 'MODL')
    show(ax[3], target, 'Ground Truth')
    fig.tight_layout()
    plt.show()


# %%

path = Path('/echo/allgemein/resources/publicTrainingData/fastmri/brain_multicoil_train/')
dataset = AcceleratedFastMRI(path)
dataloader = torch.utils.data.DataLoader(dataset, num_workers=8, shuffle=True, collate_fn=lambda batch: batch[0])
modl = MODL().cuda()
optimizer = torch.optim.Adam(modl.parameters(), lr=1e-4)
pbar = tqdm(dataloader)
for i, batch in enumerate(pbar):
    kdata, csm, target = batch['data'].cuda(), batch['csm'].cuda(), batch['target'].cuda()
    prediction = modl(kdata, csm)
    objective = mrpro.operators.functionals.MSE(target.data) - mrpro.operators.functionals.SSIM(target.data)
    (loss,) = objective(prediction.data)
    loss.backward()

    if i % 4 == 0:
        optimizer.step()
        optimizer.zero_grad()

    pbar.set_postfix(loss=loss.item())

    if i % 100 == 0:
        plot(batch, prediction)

# %%
