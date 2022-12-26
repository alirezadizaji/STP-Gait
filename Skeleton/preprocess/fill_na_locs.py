from typing import List, Tuple

import numpy as np
import pandas as pd

def fill_unknown_locs(data: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    r""" It fills NaN locations of a bunch of sequences using both pandas 'backward' and 
    forward filling of NA values (`bfill` and `ffill`). It fill NaN locations of each node
    by finding the nearest non NaN locations of that node at next frames. Because the last frames 
    could have NaN locations too, here `ffill` is used to take the location from previous frames.
    
    Args:
        data (np.ndarray): shape (N, T, V, C)

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
        skeleton = df.fillna(method='bfill').fillna(method='ffill').values
        data[idx] = skeleton

        if np.any(np.isnan(skeleton)):
            hard_cases_id.append(idx)

    data = data.reshape(N, T, V, C)
    return data, hard_cases_id