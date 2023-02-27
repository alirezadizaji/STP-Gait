from typing import Optional, Tuple

import numpy as np
import torch

from .skeleton import SkeletonDataset

class SkeletonCondDataset(SkeletonDataset):
    """ This is the very same version of SkeletonDataset (Please checkout its Doc) with a difference
    on returning 'condition mask'. If 'condition mask' is True this means it is a valid condition, O.W. invalid. """
    def __init__(self, 
            X: np.ndarray,
            names: np.ndarray,
            Y: np.ndarray,
            labeled: np.ndarray,
            frame_invalid: Optional[np.ndarray],
            cond_mask: np.ndarray) -> None:

        super().__init__(X, names, Y, labeled, frame_invalid)
        self.cond_mask = torch.from_numpy(cond_mask)

        assert self.X.size(0) == self.cond_mask.size(0) and self.X.size(1) == self.cond_mask.size(1)

    def __len__(self):
        return self.X.size(0)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        return self.X[index], self.Y[index], self.frame_invalid[index], index, self.labeled[index], self.cond_mask[index]