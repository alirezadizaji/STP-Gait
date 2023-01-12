from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..context import Skeleton
from ..utils import timer

@timer
def fill_unknown_locs(data: np.ndarray, critical_limit:int = 30, 
        non_critical_limit: Optional[int] = None) -> Tuple[np.ndarray, List[int]]:
    r""" It fills NaN locations of a bunch of sequences using both pandas 'backward' and 
    forward filling of NA values (`bfill` and `ffill`). It fill NaN locations of each node
    by finding the nearest non NaN locations of that node at next frames. Because the last frames 
    could have NaN locations too, here `ffill` is used to take the location from previous frames.
    
    Args:
        data (np.ndarray): shape (N, T, V, C)
        critical_limit (int, optional): Number (in frames) of consecutive NaN values permitted to be filled for critical joints. Default to 30.
        non_critical_limit (int, optional): Number (in frames) of consecutive NaN values permitted to be filled for non critical joints. Default to None: every consecutive is permitted.

    Returns:
        [np.ndarray, List[int]]: processed data and the list of cases which has at least one NaN value still.
    """

    def _proc(d: np.ndarray, limit: int) -> np.ndarray:
        N, T, V, C = d.shape
        d = d.reshape(N, T, -1)   # N, T, V, C -> N, T, V*C
        
        ids: List[int] = list()
        for idx, skeleton in enumerate(d):
            if not np.any(np.isnan(skeleton)):
                continue
            
            df = pd.DataFrame(skeleton, index=np.arange(T))
            skeleton = df.fillna(method='bfill', limit=limit).fillna(method='ffill', limit=limit).values
            
            if np.any(np.isnan(skeleton)):
                ids.append(idx)
                print(f"@@@ (NaN Location Filling) WARNING: {idx}th out of {N} samples has at least one NaN value with limit {limit} @@@", flush=True)
            
            d[idx] = skeleton

        d = d.reshape(N, T, V, C)
        return d, ids

    data_critical = data[..., Skeleton.CRITICAL, :]
    data_non_critical = data[..., Skeleton.NON_CRITICAL, :]

    data_critical, ids1 = _proc(data_critical, critical_limit)
    data_non_critical, ids2 = _proc(data_non_critical, non_critical_limit)

    data[..., Skeleton.CRITICAL, :] = data_critical
    data[..., Skeleton.NON_CRITICAL, :] = data_non_critical

    hard_cases_id = ids1 + ids2
    return data, hard_cases_id