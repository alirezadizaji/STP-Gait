import numpy as np

from .pad_empty_frames import pad_empty_frames


def preprocessing(data: np.ndarray, center_joint_idx: int = 8) -> np.ndarray:
    """ Preprocesses the skeleton data 

    Args:
        data (np.ndarray): shape N, T, V, C
        center_joint_idx (int): The index of center joint 
    Returns:
        np.ndarray: _description_
    """
    data = pad_empty_frames(data)
    center = data[:, [0], [center_joint_idx], :][:, None, ...]
    data = data - center

    return data
