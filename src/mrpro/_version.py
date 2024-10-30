# This file is used to get the version of the package from the VERSION file
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version('package-name')
except PackageNotFoundError:
    __version__ = Path(__file__).parent.joinpath('VERSION').read_text().strip()
