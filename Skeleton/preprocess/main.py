import numpy as np

from .pad_empty_frames import pad_empty_frames
from .fill_na_locs import fill_unknown_locs

def preprocessing(data: np.ndarray, labels: np.ndarray, names: np.ndarray, 
        center_joint_idx: int = 8, normalize: bool = False) -> np.ndarray:
    """ Preprocesses the skeleton data 

    Args:
        data (np.ndarray): shape N, T, V, C
        center_joint_idx (int): The index of center joint. _default_: 8
        remove_hard_cases (bool): If True then remove cases which are hard to processed (Due to having NA locations). _default_: False
        normalize (bool, optional): If True, then apply gaussian and minmax normalization, O.W. keep it intact. _default_: False
    Returns:
        np.ndarray: _description_
    """
    N = data.shape[0]
    print(f"@ Preprocessing phase: Number of samples {N} @", flush=True)
    data, hard_cases_id = fill_unknown_locs(data)

    mask = np.ones(data.shape[0], np.bool)
    mask[hard_cases_id] = False
    data = data[mask]
    labels = labels[mask]
    names = names[mask]
    print(f"@@ (NaN Location Filling): Number of samples {data.shape[0]} remained after filtering hard samples @@", flush=True)

    # Pad empty frames by replaying non-empty frames
    data = pad_empty_frames(data)
    center = data[:, [0], [center_joint_idx], :][:, None, ...]
    data = data - center
    print(f"@@ (Pad Empty Frames): Done @@", flush=True)
    
    # Revert upside-down direction
    data[..., 1] = -data[..., 1]
    
    if normalize:
        # Normalization
        coef = 4
        means, stds = data.mean((0, 1, 2)), data.std((0, 1, 2))
        data = (data - means) / (stds + 1e-4)

        ## Clip based on a limit per axis
        data = np.clip(data, -coef, coef)

        ## Shift all values into [0,1] range
        data = (data  + coef) / (2 * coef)

    return data, labels, names
