import numpy as np

from .pad_empty_frames import pad_empty_frames
from .fill_na_locs import fill_unknown_locs

def preprocessing(data: np.ndarray, center_joint_idx: int = 8) -> np.ndarray:
    """ Preprocesses the skeleton data 

    Args:
        data (np.ndarray): shape N, T, V, C
        center_joint_idx (int): The index of center joint 
    Returns:
        np.ndarray: _description_
    """
    data = fill_unknown_locs(data)
    data = pad_empty_frames(data)
    center = data[:, [0], [center_joint_idx], :][:, None, ...]
    data = data - center
    
    # Normalize to the range of [-1, 1]
    maxes, mines = data.max((0, 1, 2)), data.min((0, 1, 2))
    max_v = np.where(maxes > np.abs(mines), maxes, np.abs(mines))
    data = data / max_v

    return data
