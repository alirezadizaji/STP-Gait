from typing import Optional, Tuple

import numpy as np
import torch

from .skeleton import SkeletonDataset

class GraphSkeletonDataset(SkeletonDataset):
    def __init__(self, X: np.ndarray, names: np.ndarray, Y: np.ndarray, labeled: np.ndarray,
            frame_invalid: Optional[np.ndarray] = None, node_invalid: Optional[np.ndarray] = None) -> None:
        """_summary_

        (NEW) Args:
            X, Y, names, frame_invalid: All same as parent class
            node_invalid (Optional[np.ndarray], optional): If an element is True, then that node is invalid, O.W. valid (shape: N, T, V). Defaults to None.
        """
        super().__init__(X, names, Y, labeled, frame_invalid)
        
        if node_invalid is None:
            *non_node, _ = self.X.size()
            node_invalid = np.zeros(non_node, dtype=np.bool)

        self._node_invalid = torch.from_numpy(node_invalid)

    def __len__(self):
        return self.X.size(0)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X, y, *_, labeled = super().__getitem__(index)
        return X, y, self._node_invalid[index], labeled