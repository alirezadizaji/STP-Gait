from typing import Tuple

import numpy as np

from .pad_empty_frames import pad_empty_frames
from .fill_na_locs import fill_unknown_locs
from ..context import Skeleton

def preprocessing(data: np.ndarray, labels: np.ndarray, names: np.ndarray, 
        normalize: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Preprocesses the skeleton data 

    Args:
        data (np.ndarray): shape N, T, V, C
        remove_hard_cases (bool): If True then remove cases which are hard to processed (Due to having NA locations). _default_: False
        normalize (bool, optional): If True, then apply gaussian and minmax normalization, O.W. keep it intact. _default_: False
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: _description_
    """
    data, hard_cases_id = fill_unknown_locs(data)
    print(f"@@ (NaN Location): Done @@", flush=True)

    # Pad empty frames by replaying non-empty frames
    data = pad_empty_frames(data)
    center = data[:, [0], [Skeleton.CENTER], :][:, None, ...]
    data = data - center
    print(f"@@ (Pad Empty Frames): Done @@", flush=True)
    
    # Revert upside-down direction
    data[..., 1] = -data[..., 1]
    
    if normalize:
        mask = np.isnan(data)
        data_valid: np.ndarray = data[~mask]

        # Normalization
        coef = 4
        means, stds = data_valid.mean((0, 1, 2)), data_valid.std((0, 1, 2))
        data_valid = (data_valid - means) / (stds + 1e-4)

        ## Clip based on a limit per axis
        data_valid = np.clip(data_valid, -coef, coef)

        ## Shift all values into [0,1] range
        data_valid = (data_valid  + coef) / (2 * coef)
        
        data[~mask] = data_valid
        print(f"@@ (Normalization): Done @@", flush=True)

    return data, labels, names, hard_cases_id
