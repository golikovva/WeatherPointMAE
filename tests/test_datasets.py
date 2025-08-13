import os
import sys

import numpy as np

# Ensure the package root is importable regardless of the working directory
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datasets import FusionDataset, SyntheticVariableDataset


def test_synthetic_variable_dataset_generates_varied_samples():
    ds = SyntheticVariableDataset(
        "temperature",
        num_features=2,
        num_coords=3,
        length=5,
        seed=0,
        coord_seed=0,
        min_points=1,
        max_points=10,
    )

    data0, coords0 = ds[0]
    data1, coords1 = ds[1]

    assert data0.shape[1] == 2
    assert coords0.shape[1] == 3
    assert data0.shape[0] == coords0.shape[0]
    assert data1.shape[0] == coords1.shape[0]
    # With the deterministic ``coord_seed`` the first two samples use different N
    assert data0.shape[0] != data1.shape[0]


def test_fusion_dataset_combines_variables_with_shared_coords():
    temp = SyntheticVariableDataset(
        "temperature",
        num_features=1,
        num_coords=3,
        length=5,
        seed=0,
        coord_seed=42,
        min_points=1,
        max_points=10,
    )
    pressure = SyntheticVariableDataset(
        "pressure",
        num_features=2,
        num_coords=3,
        length=5,
        seed=1,
        coord_seed=42,  # share coordinates with ``temp``
        min_points=1,
        max_points=10,
    )

    fusion = FusionDataset([temp, pressure])

    features0, coords0 = fusion[0]
    assert set(features0.keys()) == {"temperature", "pressure"}
    n0 = coords0.shape[0]
    assert features0["temperature"].shape == (n0, 1)
    assert features0["pressure"].shape == (n0, 2)

    features1, coords1 = fusion[1]
    n1 = coords1.shape[0]
    assert features1["temperature"].shape == (n1, 1)
    assert features1["pressure"].shape == (n1, 2)

    # Different samples should typically have different numbers of points with
    # the chosen ``coord_seed``
    assert n0 != n1

