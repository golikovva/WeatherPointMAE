"""Synthetic point-cloud datasets.

This module provides lightweight, ``numpy``-based dataset implementations that
emulate point-cloud style samples.  They intentionally avoid any dependency on
PyTorch so that they can run in minimal environments while still mimicking the
``__len__``/``__getitem__`` protocol of :class:`torch.utils.data.Dataset`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np


@dataclass
class SyntheticVariableDataset:
    """Generate random point-cloud samples for a single variable.

    Each call to :meth:`__getitem__` returns a *new* sample consisting of
    ``N`` points, where ``N`` is drawn randomly for that call.  The returned
    tuple ``(data, coords)`` has shapes ``(N, V)`` and ``(N, R)`` respectively
    for ``V`` features and ``R`` spatial coordinate dimensions.

    Parameters
    ----------
    variable:
        Name of the represented variable.
    num_features:
        Number of features ``V`` for each point.
    num_coords:
        Dimensionality ``R`` of the coordinate space.
    length:
        Number of samples exposed by the dataset (``__len__``).
    seed:
        Optional seed for the feature RNG.
    coord_seed:
        Optional seed controlling the number of points and coordinates.  Datasets
        fused together should share the same ``coord_seed`` so that they produce
        identical point locations per sample.
    min_points, max_points:
        Range from which the number of points ``N`` is uniformly drawn.
    """

    variable: str
    num_features: int = 1
    num_coords: int = 3
    length: int = 100
    seed: int | None = None
    coord_seed: int | None = None
    min_points: int = 1
    max_points: int = 1024

    def __post_init__(self) -> None:
        self.num_features = int(self.num_features)
        self.num_coords = int(self.num_coords)
        self.length = int(self.length)
        self.min_points = int(self.min_points)
        self.max_points = int(self.max_points)
        if self.min_points <= 0 or self.max_points < self.min_points:
            raise ValueError("Invalid point range")

        # Separate RNGs so that different variables can share coordinates while
        # keeping feature generation independent.
        self._data_rng = np.random.default_rng(self.seed)
        self._coord_rng = np.random.default_rng(self.coord_seed)

    # -- basic dataset protocol -------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.length

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return a newly generated sample.

        The ``idx`` argument is accepted for compatibility but does not affect
        the generated sample.  Each invocation produces a fresh point cloud.
        """

        n_points = self._coord_rng.integers(self.min_points, self.max_points + 1)
        coords = self._coord_rng.uniform(-1.0, 1.0, size=(n_points, self.num_coords)).astype(
            np.float32
        )
        data = self._data_rng.normal(size=(n_points, self.num_features)).astype(np.float32)
        return data, coords


class FusionDataset:
    """Fuse multiple :class:`SyntheticVariableDataset` objects.

    The datasets must expose the same length and coordinate dimensionality.
    When indexed, all underlying datasets are sampled and their features merged
    into a dictionary keyed by variable name.  Coordinates are taken from the
    first dataset and validated to match across all variables.
    """

    def __init__(self, datasets: Sequence[SyntheticVariableDataset]):
        if not datasets:
            raise ValueError("At least one dataset is required")

        lengths = {len(ds) for ds in datasets}
        if len(lengths) != 1:
            raise ValueError("All datasets must have the same length")

        coord_dims = {ds.num_coords for ds in datasets}
        if len(coord_dims) != 1:
            raise ValueError("All datasets must use the same coordinate dimensions")

        self._datasets: Dict[str, SyntheticVariableDataset] = {
            ds.variable: ds for ds in datasets
        }
        self._length = lengths.pop()

    # -- basic dataset protocol -------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._length

    def __getitem__(self, idx: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Generate a fused sample at ``idx``."""

        features: Dict[str, np.ndarray] = {}
        coords_ref: np.ndarray | None = None

        for name, ds in self._datasets.items():
            data, coords = ds[idx]
            if coords_ref is None:
                coords_ref = coords
            else:
                if coords.shape != coords_ref.shape or not np.allclose(coords, coords_ref):
                    raise ValueError("Datasets yielded mismatched coordinates")
            features[name] = data

        assert coords_ref is not None  # for mypy; guaranteed by dataset check
        return features, coords_ref

