"""Synthetic point-cloud datasets.

The classes in this module are intentionally lightweight and depend only on
``numpy`` so that they can run in environments without PyTorch installed.
They provide minimal ``__len__``/``__getitem__`` interfaces that emulate the
behaviour of :class:`torch.utils.data.Dataset`.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


class SyntheticVariableDataset:
    """Generate a synthetic point cloud for a single variable.

    Parameters
    ----------
    variable: str
        Name of the variable represented by this dataset.
    num_points: int
        Number of spatial points to generate (``N`` in the description).
    num_features: int
        Number of features per point (``V``).
    num_coords: int
        Number of coordinate dimensions (``R``).
    seed: int | None
        Optional random seed for reproducibility.
    """

    def __init__(
        self,
        variable: str,
        num_points: int = 1024,
        num_features: int = 1,
        num_coords: int = 3,
        seed: int | None = None,
    ) -> None:
        self.variable = variable
        self.num_points = int(num_points)
        self.num_features = int(num_features)
        self.num_coords = int(num_coords)
        rng = np.random.default_rng(seed)
        # synthetic coordinates and features
        self._coords = rng.uniform(-1.0, 1.0, size=(self.num_points, self.num_coords)).astype(
            np.float32
        )
        self._data = rng.normal(size=(self.num_points, self.num_features)).astype(np.float32)

    # -- basic dataset protocol -------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.num_points

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return features and coordinates for a single point.

        Parameters
        ----------
        idx: int
            Index of the point to fetch.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple of ``(features, coords)`` where ``features`` has shape
            ``(V,)`` and ``coords`` has shape ``(R,)``.
        """

        return self._data[idx], self._coords[idx]

    # -- convenience -----------------------------------------------------------
    def point_cloud(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the full point cloud as ``(data, coords)``.

        ``data`` has shape ``(N, V)`` and ``coords`` has shape ``(N, R)``.
        """

        return self._data.copy(), self._coords.copy()


class FusionDataset:
    """Combine multiple :class:`SyntheticVariableDataset` objects.

    The datasets must have the same number of points and coordinate
    dimensionality.  The resulting fused dataset exposes a unified point cloud
    with coordinates taken from the first dataset and a dictionary of features
    keyed by variable name.
    """

    def __init__(self, datasets: Sequence[SyntheticVariableDataset]):
        if not datasets:
            raise ValueError("At least one dataset is required")

        lengths = {len(ds) for ds in datasets}
        if len(lengths) != 1:
            raise ValueError("All datasets must have the same number of points")

        coord_dims = {ds._coords.shape[1] for ds in datasets}
        if len(coord_dims) != 1:
            raise ValueError("All datasets must use the same coordinate dimensions")

        self._datasets: Dict[str, SyntheticVariableDataset] = {
            ds.variable: ds for ds in datasets
        }
        self._variables = list(self._datasets.keys())
        self._length = lengths.pop()
        self._coord_dim = coord_dims.pop()

    # -- basic dataset protocol -------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._length

    def __getitem__(self, idx: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Return features for all variables at ``idx`` and the coordinates."""

        features = {name: ds._data[idx] for name, ds in self._datasets.items()}
        coords = next(iter(self._datasets.values()))._coords[idx]
        return features, coords

    # -- convenience -----------------------------------------------------------
    def point_cloud(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Return the fused point cloud.

        The return value is ``(features, coords)`` where ``features`` is a
        dictionary mapping variable names to arrays of shape ``(N, V)`` and
        ``coords`` is an array of shape ``(N, R)`` shared across all variables.
        """

        features = {name: ds._data.copy() for name, ds in self._datasets.items()}
        coords = next(iter(self._datasets.values()))._coords.copy()
        return features, coords
