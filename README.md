<h1 align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/PTB-MR/mrpro/refs/heads/main/docs/source/_static/logo_white.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/PTB-MR/mrpro/refs/heads/main/docs/source/_static/logo.svg">
  <img src="https://raw.githubusercontent.com/PTB-MR/mrpro/refs/heads/main/docs/source/_static/logo.svg" alt="MRpro logo" width="50%">
</picture>

</h1><br>

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org/project/mrpro/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Coverage Bagde](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/ckolbPTB/48e334a10caf60e6708d7c712e56d241/raw/coverage.json)](https://github.com/PTB-MR/mrpro/actions?query=workflow%3A%22%22Report+PyTest%22%22)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14509598.svg)](https://doi.org/10.5281/zenodo.14509598)

MR image reconstruction and processing package specifically developed for PyTorch.

- **Source code:** <https://github.com/PTB-MR/mrpro>
- **Documentation:** <https://ptb-mr.github.io/mrpro/>
- **Bug reports:** <https://github.com/PTB-MR/mrpro/issues>
- **Try it out:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PTB-MR/mrpro)

## Awards

- 2024 ISMRM QMRI Study Group Challenge, 2nd prize for Relaxometry ([T2*](https://github.com/PTB-MR/mrpro/blob/8d2133c4a7ce63ac490798c4eb5a70cc1c543646/examples/qmri_sg_challenge_2024_t2_star.ipynb) and [T1](https://github.com/PTB-MR/mrpro/blob/8d2133c4a7ce63ac490798c4eb5a70cc1c543646/examples/qmri_sg_challenge_2024_t1.ipynb))

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
reconstruction = mrpro.algorithms.reconstruction.DirectReconstruction(kdata)
img = reconstruction(kdata)
```

Full example: <https://github.com/PTB-MR/mrpro/blob/main/examples/scripts/direct_reconstruction.py>

### Estimate quantitative parameters

Quantitative parameter maps can be obtained by creating a functional to be minimized and calling a non-linear solver such as ADAM.

```python
# Define signal model
model = MagnitudeOp() @ InversionRecovery(ti=idata_multi_ti.header.ti)
# Define loss function and combine with signal model
mse = MSE(idata_multi_ti.data.abs())
functional = mse @ model
[...]
# Run optimization
params_result = adam(functional, [m0_start, t1_start], n_iterations=n_iterations, learning_rate=learning_rate)
```

Full example: <https://github.com/PTB-MR/mrpro/blob/main/examples/scripts/qmri_sg_challenge_2024_t1.py>

### Pulseq support

The trajectory can be calculated directly from a provided pulseq-file.

```python
# Read raw data and calculate trajectory using KTrajectoryPulseq
kdata = KData.from_file(data_file.name, KTrajectoryPulseq(seq_path=seq_file.name))
```

Full example: <https://github.com/PTB-MR/mrpro/blob/main/examples/scripts/comparison_trajectory_calculators.py>

## Contributing

We are looking forward to your contributions via "fork and pull requests". If you would like to fix a bug or add a new feature:

1. Create your own copy of MRpro (i.e. create a fork via GitHub)
2. Clone your forked copy of the MRpro repository
3. Create/select a python environment (e.g. ``` conda create -n mrpro python=3.12 ```)
4. Install MRpro in editable mode with developer dependencies: ``` pip install -e ".[dev]" ```
5. Setup pre-commit hook: ``` pre-commit install ```
6. Create a new branch
7. Implement your changes to MRpro
8. Commit and push them to GitHub
9. Open a pull request via GitHub

You can find more information on "fork and pull requests" on the [GitHub documentation](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project)

Please also look at our [contributor guide](https://ptb-mr.github.io/mrpro/contributor_guide.html) for more information on the repository structure, naming conventions, and other useful information.

> [!NOTE]  
> There are a few things which cannot be modified as "fork and pull requests" such as modifications of the docker images. If you think something needs to be changed there, please open up an issue first. 
