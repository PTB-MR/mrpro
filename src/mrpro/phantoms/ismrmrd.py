"""Dataset wrapper for collections of ISMRMRD files."""

import re
from collections.abc import Callable, Mapping, Sequence
from os import PathLike
from pathlib import Path

import ismrmrd
import torch

from mrpro.data.acq_filters import is_image_acquisition
from mrpro.data.KData import KData
from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.traj_calculators.KTrajectoryCalculator import KTrajectoryCalculator
from mrpro.data.traj_calculators.KTrajectoryCartesian import KTrajectoryCartesian
from mrpro.data.traj_calculators.KTrajectoryIsmrmrd import KTrajectoryIsmrmrd


def natural_sort_key(path: Path) -> tuple[str | int, ...]:
    """Sort paths case-insensitively and numbers numerically."""
    return tuple(int(part) if part.isdigit() else part.casefold() for part in re.split(r'(\d+)', path.as_posix()))


class IsmrmrdDataset(torch.utils.data.Dataset):
    """Dataset of standard ISMRMRD files.

    Directories are searched recursively for ``.h5``, ``.hdf5``, and ``.mrd``
    files. Files are naturally sorted, and each item is loaded as
    :class:`~mrpro.data.KData`.
    """

    def __init__(
        self,
        path: str | PathLike | Sequence[str | PathLike],
        trajectory: KTrajectoryCalculator | KTrajectory | KTrajectoryIsmrmrd | None = None,
        header_overwrites: Mapping[str, object] | None = None,
        dataset_idx: int = -1,
        acquisition_filter_criterion: Callable[[ismrmrd.Acquisition], bool] = is_image_acquisition,
    ) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        path
            Directory, raw file, or explicit sequence of raw files.
        trajectory
            Trajectory calculator or trajectory. Cartesian is used by default.
        header_overwrites
            Values passed to :meth:`~mrpro.data.KData.from_file`.
        dataset_idx
            ISMRMRD group index within each file.
        acquisition_filter_criterion
            Predicate selecting acquisitions within each file.
        """
        if isinstance(path, str | PathLike):
            path = Path(path)
            files = (
                [file for file in path.rglob('*') if file.suffix.lower() in {'.h5', '.hdf5', '.mrd'}]
                if path.is_dir()
                else [path]
            )
        else:
            files = [Path(file) for file in path]
        self.files = tuple(sorted(files, key=natural_sort_key))
        self.trajectory = trajectory if trajectory is not None else KTrajectoryCartesian()
        self.header_overwrites = header_overwrites
        self.dataset_idx = dataset_idx
        self.acquisition_filter_criterion = acquisition_filter_criterion

    def __len__(self) -> int:
        """Get the number of ISMRMRD files."""
        return len(self.files)

    def __getitem__(self, idx: int) -> KData:
        """Load one ISMRMRD file as KData."""
        return self.load(self.files[idx])

    def load(self, filename: str | PathLike) -> KData:
        """Load an ISMRMRD file using this dataset's loading options."""
        return KData.from_file(
            filename,
            self.trajectory,
            header_overwrites=self.header_overwrites,
            dataset_idx=self.dataset_idx,
            acquisition_filter_criterion=self.acquisition_filter_criterion,
        )
