from typing import List, Tuple

import numpy as np
import pandas as pd

from ..utils import timer

@timer
def fill_unknown_locs(data: np.ndarray, limit:int = 30) -> Tuple[np.ndarray, List[int]]:
    r""" It fills NaN locations of a bunch of sequences using both pandas 'backward' and 
    forward filling of NA values (`bfill` and `ffill`). It fill NaN locations of each node
    by finding the nearest non NaN locations of that node at next frames. Because the last frames 
    could have NaN locations too, here `ffill` is used to take the location from previous frames.
    
    Args:
        data (np.ndarray): shape (N, T, V, C)
        limit (int, optional): Number (in frames) of consecutive NaN values permitted to be filled. Default to 30.

    Returns:
        [np.ndarray, List[int]]: processed data and the list of cases which has at least one NaN value still.
    """

    N, T, V, C = data.shape
    data = data.reshape(N, T, -1)   # N, T, V, C -> N, T, V*C
    
    hard_cases_id: List[int] = list()
    for idx, skeleton in enumerate(data):
        if not np.any(np.isnan(skeleton)):
            continue
        
        df = pd.DataFrame(skeleton, index=np.arange(T))
        skeleton = df.fillna(method='bfill', limit=limit).fillna(method='ffill', limit=limit).values
        
        if np.any(np.isnan(skeleton)):
            hard_cases_id.append(idx)
            print(f"@@@ (NaN Location Filling) WARNING: {idx}th out of {N} samples has at least one NaN value still after filling a consecutive number of NaN values with limit {limit} @@@", flush=True)
        
        data[idx] = skeleton

    data = data.reshape(N, T, V, C)
    return data, hard_cases_id