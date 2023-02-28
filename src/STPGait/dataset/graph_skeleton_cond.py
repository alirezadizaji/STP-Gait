from typing import Optional, Tuple

import numpy as np
import torch

from .skeleton_cond import SkeletonCondDataset

class GraphSkeletonCondDataset(SkeletonCondDataset):
    def __init__(self, X: np.ndarray, names: np.ndarray, Y: np.ndarray, labeled: np.ndarray,
            frame_invalid: Optional[np.ndarray] = None, node_invalid: Optional[np.ndarray] = None, cond_mask: np.ndarray = None) -> None:

        super().__init__(X, names, Y, labeled, frame_invalid, cond_mask)
        
        if node_invalid is None:
            *non_node, _ = self.X.size()
            node_invalid = np.zeros(non_node, dtype=np.bool)

        self._node_invalid = torch.from_numpy(node_invalid)

    def __len__(self):
        return self.X.size(0)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X, y, *_, labeled, cond_mask = super().__getitem__(index)
        return X, y, self._node_invalid[index], labeled, cond_mask