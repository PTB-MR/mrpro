from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
from mrpro import algorithms, operators, data, phantoms, utils


try:
    __version__ = version("package-name")
except PackageNotFoundError:
    # package is not installed
    __version__ = Path(__file__).parent.parent.parent.joinpath("VERSION").read_text().strip()
