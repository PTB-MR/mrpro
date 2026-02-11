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
    """A single Batch."""

    data: mrpro.data.KData
    target: mrpro.data.IData
    csm: mrpro.data.CsmData


class AcceleratedFastMRI(torch.utils.data.Dataset):
    """An undersampled FastMRI Dataset."""

    def __init__(self, path: Path, acceleration: float = 12, noise_level: float = 0.1):
        """Create an undersampled FastMRI Dataset.

        Parameters
        ----------
        path
            Path to the FastMRI dataset.
        acceleration
            Undersampling factor; higher values mean more acceleration. Default is 12.
        noise_level
            Level of additive Gaussian noise applied to the FastMRI dataset. Default is 0.1.
        """
        self.acceleration = acceleration
        files = list(path.glob('*AXT1*'))
        self.dataset = mrpro.phantoms.FastMRIKDataDataset(files)
        self.noise_level = noise_level

    def __len__(self):
        """Get length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> BatchType:
        """Get a single batch of data.

        Parameters
        ----------
        index
            Index of the batch.

        Returns
        -------
        A single batch of data with keys 'data', 'target', and 'csm'.and
        """
        data = self.dataset[index]
        data = data.remove_readout_os()
        data.data /= data.data.std()
        reconstruction = mrpro.algorithms.reconstruction.DirectReconstruction(
            data, csm=lambda data: mrpro.data.CsmData.from_idata_inati(data, downsampled_size=64)
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
    """MODL network."""

    def __init__(self, iterations: int = 8, n_features: Sequence[int] = (64, 64, 64, 64)):
        """Initialize MODL network.

        Parameters
        ----------
        iterations
            Number of iterations.
        n_features
            Number of features in the network.
        """
        super().__init__()
        cnn = mrpro.nn.nets.BasicCNN(
            dim=2,
            channels_in=2,
            channels_out=2,
            n_features=n_features,
            batch_norm=True,
        )
        self.network = mrpro.nn.Residual(mrpro.nn.ComplexAsChannel(mrpro.nn.PermutedBlock((-1, -2), cnn)))
        self.network = torch.compile(self.network, dynamic=True, fullgraph=True)
        self.iterations = iterations
        self.regularization_weights = torch.nn.Parameter(0.2 * torch.ones(iterations))

    def __call__(self, kdata: mrpro.data.KData, csm: mrpro.data.CsmData) -> mrpro.data.IData:
        """Apply MODL network.

        Parameters
        ----------
        kdata
            The k-space data.
        csm
            The coil sensitivity maps.

        Returns
        -------
        The reconstructed image.
        """
        return super().__call__(kdata, csm)

    def forward(self, kdata: mrpro.data.KData, csm: mrpro.data.CsmData) -> mrpro.data.IData:
        """Apply the MODL network."""
        fourier_op = mrpro.operators.FourierOp.from_kdata(kdata)
        acquisition_op = fourier_op @ csm.as_operator()
        (zero_filled_image,) = acquisition_op.H(kdata.data)
        gram = acquisition_op.gram
        data_consistency_op = mrpro.operators.ConjugateGradientOp(
            operator_factory=lambda _image, weight: gram + weight,
            rhs_factory=lambda image, weight: zero_filled_image + weight * image,
        )

        (image,) = mrpro.algorithms.optimizers.cg(gram, zero_filled_image, max_iterations=5)
        for iteration in range(self.iterations):
            regularization = self.network(image)
            (image,) = data_consistency_op(regularization, self.regularization_weights[iteration])

        return mrpro.data.IData(image, header=mrpro.data.IHeader.from_kheader(kdata.header))


def plot(batch: BatchType, prediction: mrpro.data.IData, step: int) -> None:
    """Plot the direct, sense, and modl reconstructions."""
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
                f'SSIM: {ssim_value.item():.2f}',
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
    fig.savefig(f'modl_{step}.pdf', bbox_inches='tight', pad_inches=0)


# %%.
path = Path('/echo/allgemein/resources/publicTrainingData/fastmri/brain_multicoil_train/')
dataset = AcceleratedFastMRI(path)
dataloader = torch.utils.data.DataLoader(dataset, num_workers=16, shuffle=True, collate_fn=lambda batch: batch[0])
modl = MODL().cuda()
optimizer = torch.optim.Adam(modl.parameters(), lr=1e-3)
pbar = tqdm(dataloader)
for i, batch in enumerate(pbar):
    optimizer.zero_grad()
    kdata, csm, target = (batch['data'].cuda(), batch['csm'].cuda(), batch['target'].cuda())
    prediction = modl(kdata, csm)
    objective = 0.5 * mrpro.operators.functionals.MSE(target.data) - mrpro.operators.functionals.SSIM(target.data)
    (loss,) = objective(prediction.data)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(modl.parameters(), 5.0)
    optimizer.step()

    pbar.set_postfix(loss=loss.item())
    if i % 200 == 0:
        plot(batch, prediction, i)
        print(modl.regularization_weights)
        state = {'modl': modl.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state, f'modl_{i}.pt')

# %%
