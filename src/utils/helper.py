import time

import torch

from constants import MB


def tt(_t=0.0):
    """
    Returns the time difference from the given time
    """
    return time.time() - _t


def ma():
    """
    Returns the memory allocated in MB
    """
    _m = torch.cuda.memory_allocated()
    return _m / MB
