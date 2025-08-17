from typing import Callable, List, Dict, Tuple, Any

import torch
import numpy as np

from libs.datasets.gridded_datasets import NCs2sDataset, WRFs2sDataset, ERAs2sDataset

class PointWeatherDataset(NCs2sDataset):
    def __init__(self, data_folder, data_variables=None):
        super().__init__(
            data_folder,
            data_variables=data_variables, 
            seq_len=1, 
            time_resolution_h=1, 
            add_coords=False, 
            add_time_encoding=False
        )


def pc_dataset(cls):
    """
    Modifies the given Dataset class to return data as point cloud
    instead of grid.

    e.g. MNISTWithIndices = dataset_with_indices(MNIST)
         dataset = MNISTWithIndices('~/datasets/mnist')
    """
    def __init__(self, *args, **kwargs):
        if len(args) > 2:
            args.pop(2)  # remove 'seq_len' if passed
        kwargs['seq_len'] = 1
        kwargs['add_coords'] = False
        kwargs['add_time_encoding'] = False
        super(cls, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        data = cls.__getitem__(self, index).squeeze(0)  # remove seq_len dimension
        data = data.reshape(*data.shape[:-2], -1).T  # flatten all but the first dimension
        coords = np.stack((self.grid.latitude.flatten(), self.grid.longitude.flatten()), axis=-1)
        data = {'data': data, 'coords': coords}
        return data

    return type(cls.__name__, (cls,), {
        '__init__': __init__,
        '__getitem__': __getitem__,
    })