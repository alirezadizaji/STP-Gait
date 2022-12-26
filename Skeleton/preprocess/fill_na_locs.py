import numpy as np
import pandas as pd

def fill_unknown_locs(data: np.ndarray) -> np.ndarray:
    r""" It fills NaN locations of a bunch of sequences using both pandas 'backward' and 
    forward filling of NA values (`bfill` and `ffill`). It fill NaN locations of each node
    by finding the nearest non NaN locations of that node at next frames. Because the last frames 
    could have NaN locations too, here `ffill` is used to take the location from previous frames.
    
    Args:
        data (np.ndarray): shape (N, T, V, C)
    """

    N, T, V, C = data.shape[1]
    data = data.reshape(N, T, -1)   # N, T, V, C -> N, T, V*C
    for idx, skeleton in enumerate(data):

        if not np.any(np.isnan(skeleton)):
            continue
        
        df = pd.DataFrame(index=np.arange(T), values=skeleton)
        skeleton = df.fillna(method='bfill').fillna(method='ffill').values
        data[idx] = skeleton

    data = data.reshape(N, T, V, C)
    return data