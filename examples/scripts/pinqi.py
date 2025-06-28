# %%
import einops
import einops.layers
import mrpro
import torch


# %%
class Dataset(torch.utils.data.Dataset):
    def __init__(self, size=64, acceleration=8, n_coils=8, random=True):
        self.phantom = mrpro.phantoms.brainweb.BrainwebSlices(
            what=('m0', 't1', 'mask'),
            seed='index' if not random else 'random',
            slice_preparation=mrpro.phantoms.brainweb.augment(size=size),
        )
        self.signalmodel = mrpro.operators.models.SaturationRecovery((0.5, 1.0, 1.5, 2, 8))
        self.encoding_matrix = mrpro.data.SpatialDimension(1, size, size)
        self.fov = mrpro.data.SpatialDimension(0.01, 0.25, 0.25)
        self.acceleration = acceleration
        self.n_coils = n_coils
        self._random = random

    @property
    def n_images(self):
        return 5

    @property
    def n_parameters(self):
        return 2

    def __len__(self):
        return len(self.phantom)

    def __getitem__(self, index):
        phantom = self.phantom[index]
        (images,) = self.signalmodel(phantom['m0'], phantom['t1'])
        seed = torch.randint(0, 1000000, (1,)).item() if self._random else index
        traj = mrpro.data.traj_calculators.KTrajectoryCartesian.gaussian_variable_density(
            encoding_matrix=self.encoding_matrix,
            seed=seed,
            fwhm_ratio=2,
        )
        header = mrpro.data.KHeader(
            encoding_matrix=self.encoding_matrix,
            recon_matrix=self.encoding_matrix,
            recon_fov=self.fov,
            encoding_fov=self.fov,
        )
        header.ti = self.signalmodel.saturation_time.tolist()
        fourier_op = mrpro.operators.FourierOp(self.encoding_matrix, self.encoding_matrix, traj)
        csm = mrpro.data.CsmData(mrpro.phantoms.coils.birdcage_2d(self.n_coils, self.encoding_matrix), header)
        images = einops.rearrange(images, 't y x -> t 1 1 y x')
        (data,) = (fourier_op @ csm.as_operator())(images)
        kdata = mrpro.data.KData(header, data, traj)
        return {'kdata': kdata, 'csm': csm, **phantom}

    @staticmethod
    def collate_fn(batch):
        return torch.utils.data._utils.collate.collate(
            batch,
            collate_fn_map={
                mrpro.data.Dataclass: lambda batch, *, collate_fn_map: batch[0].stack(*batch[1:]),
                **torch.utils.data._utils.collate.default_collate_fn_map,
            },
        )


# %%
ds = Dataset()
dl = torch.utils.data.DataLoader(
    ds, batch_size=4, collate_fn=ds.collate_fn, num_workers=4, worker_init_fn=lambda *_: torch.set_num_threads(1)
)

# %%


class PINQI(torch.nn.Module):
    def __init__(self, signalmodel, n_parameters, n_images, n_iterations=2):
        super().__init__()
        self.signalmodel = mrpro.operators.RearrangeOp('t batch ... -> batch t ...') @ signalmodel
        self._n_parameters = n_parameters
        self._n_images = n_images
        self.parameter_net = torch.nn.Conv2d(n_images * 2, n_parameters, kernel_size=1)
        self.image_net = torch.nn.Conv3d(2, 2, kernel_size=1)
        self.lambdas_raw = torch.nn.Parameter(torch.ones(n_iterations, 3))
        self.softplus = torch.nn.Softplus()

        def objective_factory(parameter_reg, lambda_parameters, image):
            dc = mrpro.operators.functionals.L2NormSquared(image) @ self.signalmodel
            reg = mrpro.operators.functionals.L2NormSquared(parameter_reg)
            return dc + lambda_parameters * reg

        self.nonlinear_solver = mrpro.operators.OptimizerOp(objective_factory, lambda parameter_reg, *_: parameter_reg)
        self.linear_solver = mrpro.operators.ConjugateGradientOp(
            operator_factory=lambda gram, lambda_image, lambda_q, *_: gram + lambda_image + lambda_q,
            rhs_factory=lambda _gram, lambda_image, lambda_q, image_reg, signal, zero_filled_image: (
                zero_filled_image + lambda_image * image_reg + lambda_q * signal,
            ),
        )

    def get_parameter_reg(self, image):
        image = einops.rearrange(torch.view_as_real(image), 'batch t 1 1 y x complex-> batch (t complex) y x')
        parameters = self.parameter_net(image)
        parameters = einops.rearrange(parameters, 'batch parameters y x-> parameters batch 1 1 y x')
        return tuple(parameters)

    def get_image_reg(self, image):
        image = einops.rearrange(torch.view_as_real(image), 'batch t 1 1 y x complex-> batch complex t y x')
        image = image + self.image_net(image)
        image = einops.rearrange(image, 'batch complex t y x-> batch t 1 1 y x complex')
        return torch.view_as_complex(image.contiguous())

    def forward(self, kdata: mrpro.data.KData, csm: mrpro.data.CsmData):
        csm_op = csm.as_operator()
        fourier_op = mrpro.operators.FourierOp.from_kdata(kdata)
        aquisition_op = fourier_op @ csm_op
        gram = aquisition_op.gram
        (zero_filled_image,) = aquisition_op.H(kdata.data)
        images = list(mrpro.algorithms.optimizers.cg(gram, zero_filled_image, max_iterations=3))
        parameters = [self.get_parameter_reg(images[-1])]
        for lambda_image, lambda_q, lambda_parameter in self.softplus(self.lambdas_raw):
            # subproblem 1
            image_reg = self.get_image_reg(images[-1])
            (signal,) = self.signalmodel(*parameters[-1])
            images.append(self.linear_solver(gram, lambda_image, lambda_q, image_reg, signal, zero_filled_image))
            # subproblem 2
            parameters_reg = self.get_parameter_reg(images[-1])
            parameters.append(self.nonlinear_solver(parameters_reg, lambda_parameter, images[-1]))

        return images, parameters


# %%
from tqdm import tqdm

pinqi = PINQI(ds.signalmodel, ds.n_parameters, ds.n_images)

for batch in tqdm(dl):
    pred = pinqi(batch['kdata'], batch['csm'])
# %%
