# MRpro

![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Coverage Bagde](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/ckolbPTB/48e334a10caf60e6708d7c712e56d241/raw/coverage.json)

MR image reconstruction and processing package specifically developed for PyTorch.

- **Source code:** <https://github.com/PTB-MR/mrpro>
- **Documentation:** <https://ptb-mr.github.io/mrpro/>
- **Bug reports:** <https://github.com/PTB-MR/mrpro/issues>
- **Try it out:** [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/PTB-MR/mrpro.git/main?labpath=examples)

## Awards

- 2024 ISMRM QMRI Study Group Challenge, 2nd prize for Relaxometry (T2* and T1)

## Main features

- **ISMRMRD support** MRpro supports [ismrmrd-format](https://ismrmrd.readthedocs.io/en/latest/) for MR raw data.
- **PyTorch** All data containers utilize PyTorch tensors to ensure easy integration in PyTorch-based network schemes.
- **Cartesian and non-Cartesian trajectories** MRpro can reconstruct data obtained with Cartesian and non-Cartesian (e.g. radial, spiral...) sapling schemes. MRpro automatically detects if FFT or nuFFT is required to reconstruct the k-space data.
- **Pulseq support** If the data acquisition was carried out using a [pulseq-based](http://pulseq.github.io/) sequence, the seq-file can be provided to MRpro and the used trajectory is automatically calculated.
- **Signal models** A range of different MR signal models are implemented (e.g. T1 recovery, WASABI).
- **Regularized image reconstruction** Regularized image reconstruction algorithms including Wavelet-based compressed sensing or total variation regularized image reconstruction are available.

## Examples

In the following, we show some code snippets to highlight the use of MRpro. Each code snippet only shows the main steps. A complete working notebook can be found in the provided link.

### Simple reconstruction

Read the data and trajectory and reconstruct an image by applying a density compensation function and then the adjoint of the Fourier operator and the adjoint of the coil sensitivity operator.

```python
# Read the trajectory from the ISMRMRD file
trajectory = mrpro.data.traj_calculators.KTrajectoryIsmrmrd()
# Load in the Data from the ISMRMRD file
kdata = mrpro.data.KData.from_file(data_file.name, trajectory)
# Perform the reconstruction
reconstruction = mrpro.algorithms.reconstruction.DirectReconstruction.from_kdata(kdata)
img = reconstruction(kdata)
```

Full example: <https://github.com/PTB-MR/mrpro/blob/main/examples/direct_reconstruction.py>

### Estimate quantitative parameters

Quantitative parameter maps can be obtained by creating a functional to be minimized and calling a non-linear solver such as ADAM.

```python
# Define signal model
model = MagnitudeOp() @ InversionRecovery(ti=idata_multi_ti.header.ti)
# Define loss function and combine with signal model
mse = MSEDataDiscrepancy(idata_multi_ti.data.abs())
functional = mse @ model
[...]
# Run optimization
params_result = adam(functional, [m0_start, t1_start], max_iter=max_iter, lr=lr)
```

Full example: <https://github.com/PTB-MR/mrpro/blob/main/examples/qmri_sg_challenge_2024_t1.py>

### Pulseq support

The trajectory can be calculated directly from a provided pulseq-file.

```python
# Read raw data and calculate trajectory using KTrajectoryPulseq
kdata = KData.from_file(data_file.name, KTrajectoryPulseq(seq_path=seq_file.name))
```

Full example: <https://github.com/PTB-MR/mrpro/blob/main/examples/pulseq_2d_radial_golden_angle.py>

## Contributing

### Installation for developers

1. Clone the MRpro repository
2. Create/select a python environment
3. Install "MRpro" in editable mode including test dependencies: ``` pip install -e ".[test]" ```
4. Setup pre-commit hook: ``` pre-commit install ```

### Recommended IDE and Extensions

We recommend to use [Microsoft Visual Studio Code](https://code.visualstudio.com/download). A list of recommended extensions for VSCode is given in the [.vscode/extensions.json](.vscode\extensions.json)

### Style

Please look at our [contributor guide](https://ptb-mr.github.io/mrpro/contributor_guide.html) for more information on the repository structure, naming conventions, and other useful information.
