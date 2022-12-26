import numpy as np

from .pad_empty_frames import pad_empty_frames
from .fill_na_locs import fill_unknown_locs

def preprocessing(data: np.ndarray, labels: np.ndarray, names: np.ndarray, 
        center_joint_idx: int = 8) -> np.ndarray:
    """ Preprocesses the skeleton data 

    Args:
        data (np.ndarray): shape N, T, V, C
        center_joint_idx (int): The index of center joint. _default: 8
        remove_hard_cases (bool): If True then remove cases which are hard to processed (Due to having NA locations). _default: False
    
    Returns:
        np.ndarray: _description_
    """
    data, hard_cases_id = fill_unknown_locs(data)

    mask = np.ones(data.shape[0], np.bool)
    mask[hard_cases_id] = False
    data = data[mask]
    labels = labels[mask]
    names = names[mask]
    
    data = pad_empty_frames(data)
    center = data[:, [0], [center_joint_idx], :][:, None, ...]
    data = data - center
    
    # Normalize to the range of [-1, 1]
    maxes, mines = data.max((0, 1, 2)), data.min((0, 1, 2))
    max_v = np.where(maxes > np.abs(mines), maxes, np.abs(mines))
    data = data / max_v
    
    return data, labels, names
