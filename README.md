# MRpro
MR image reconstruction and processing package specifically developed for PyTorch.

This package supports ismrmrd-format for MR raw data. All data containers utilise PyTorch-tensors to ensure easy integration in PyTorch-based network schemes.

![Python](https://img.shields.io/badge/python-3.10+-blue)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


## Installation for developers

1. Clone the repo
2. Create/select a python environment
3. Open terminal in the "MRpro" main folder
4. Install "MRpro" in editable mode with linting and testing: ``` pip install -e ".[lint,test]" ```
5. Setup Pre-Commit Hook: ``` pre-commit install ```

## Recommended IDE and Extensions

We recommend to use [Microsoft Visual Studio Code](https://code.visualstudio.com/download) with the following extensions:

- Python (Microsoft)
- Pylance (Microsoft)
- isort (Microsoft)
- Python Indent (Kevin Rose)
- Python Type Hint (njqdev)
- Editorconfig for VS Code (EditorConfig)
- Mypy (Martan Gover)
- autoDocstring (Nils Werner)
- markdownlint (David Anson)
- Even Better TOML (tamasfe)

Further extensions that might be useful:

- IntelliCode (Microsoft)
- Remote - SHH (Microsoft)
- GitHub Copilot (GitHub - **fee-based** )

### *Note that this project uses a pyproject.toml instead of setup.py file*

