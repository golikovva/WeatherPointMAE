import math
import numpy as np
from pathlib import Path

def split_dates(start_date, end_date, train_size, validation_size, test_size=None, time_step='h'):
    days = np.arange(start_date, end_date, np.timedelta64(1, time_step), dtype=f'datetime64[{time_step}]')  # .astype(datetime)
    train_end = math.ceil(len(days) * train_size)
    val_end = math.ceil(len(days) * (train_size + validation_size))
    train = days[:train_end]
    val = days[train_end:val_end]
    test = days[val_end:]
    return train, val, test