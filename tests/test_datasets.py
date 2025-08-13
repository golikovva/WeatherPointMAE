import os
import sys

import numpy as np

# Ensure the package root is importable regardless of the working directory
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datasets import SyntheticVariableDataset, FusionDataset


def test_synthetic_variable_dataset_shapes():
    ds = SyntheticVariableDataset("temperature", num_points=10, num_features=2, num_coords=3, seed=0)
    data, coords = ds.point_cloud()
    assert data.shape == (10, 2)
    assert coords.shape == (10, 3)

    # individual item check
    features, coord = ds[0]
    assert features.shape == (2,)
    assert coord.shape == (3,)


def test_fusion_dataset_combines_variables():
    temp = SyntheticVariableDataset("temperature", num_points=8, num_features=1, num_coords=3, seed=0)
    pressure = SyntheticVariableDataset("pressure", num_points=8, num_features=2, num_coords=3, seed=1)
    fusion = FusionDataset([temp, pressure])

    features, coords = fusion.point_cloud()
    assert set(features.keys()) == {"temperature", "pressure"}
    assert features["temperature"].shape == (8, 1)
    assert features["pressure"].shape == (8, 2)
    assert coords.shape == (8, 3)

    item_features, item_coords = fusion[0]
    assert set(item_features.keys()) == {"temperature", "pressure"}
    assert item_features["temperature"].shape == (1,)
    assert item_features["pressure"].shape == (2,)
    assert item_coords.shape == (3,)
