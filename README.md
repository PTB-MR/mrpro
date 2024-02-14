# MRpro

MR image reconstruction and processing package specifically developed for PyTorch.

This package supports ismrmrd-format for MR raw data. All data containers utilize PyTorch-tensors to ensure easy integration in PyTorch-based network schemes.

![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Coverage Bagde](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/ckolbPTB/48e334a10caf60e6708d7c712e56d241/raw/coverage.json)

If you want to give MRpro a try you can use
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/PTB-MR/mrpro.git/example_framework?labpath=examples)

## Installation for developers

1. Clone the repo
2. Create/select a python environment
3. Open terminal in the "MRpro" main folder
4. Install "MRpro" in editable mode with linting and testing: ``` pip install -e ".[lint,test]" ```
5. Setup Pre-Commit Hook: ``` pre-commit install ```

## Recommended IDE and Extensions

We recommend to use [Microsoft Visual Studio Code](https://code.visualstudio.com/download).

A list of recommended extensions for VSCode is given in the [.vscode/extensions.json](.vscode\extensions.json)

Further extensions that might be useful:

- IntelliCode (Microsoft)
- Remote - SHH (Microsoft)
- GitHub Copilot (GitHub - **fee-based** )
- Git Graph (mhutchie)
- GitLens (GitKraken)
