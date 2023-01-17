from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .pad_empty_frames import pad_empty_frames
from .fill_na_locs import fill_unknown_locs
from ..context import Skeleton


@dataclass
class PreprocessingConfig:
    normalize: bool = False
    critical_limit: int = 30
    non_critical_limit: Optional[int] = None

def preprocessing(data: np.ndarray, config: PreprocessingConfig = PreprocessingConfig())\
        -> Tuple[np.ndarray, List[int]]:
    """ Preprocesses the skeleton data 

    Args:
        data (np.ndarray): shape N, T, V, C
        config (PreprocessingConfig): configuration to run preprocessing with

    Returns:
        Tuple[np.ndarray, List[int]]: processed data + IDs of hard samples(=having still NaN locations in their sequence)
    """
    data, hard_cases_id = fill_unknown_locs(data, config.critical_limit, config.non_critical_limit)
    print(f"@@ (NaN Location): Done @@", flush=True)

    # Pad empty frames by replaying non-empty frames
    data = pad_empty_frames(data)
    print(f"@@ (Pad Empty Frames): Done @@", flush=True)

    center = data[:, [0], [Skeleton.CENTER], :][:, None, ...]
    data = data - center
    
    # Revert upside-down direction
    data[..., 1] = -data[..., 1]
    
    if config.normalize:
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

    return data, hard_cases_id
