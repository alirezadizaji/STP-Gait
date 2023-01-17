from typing import Callable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .skeleton import SkeletonDataset

class GraphSkeletonDataset(SkeletonDataset):
    def __init__(self, 
            X: np.ndarray,
            names: np.ndarray,
            Y: np.ndarray,
            get_edge_index: Callable[[int], torch.Tensor]) -> None:
        super().__init__(X, names, Y)

        self.get_edge_index = get_edge_index

    def __len__(self):
        return self.X.size(0)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[index], self.Y[index], self.get_edge_index(self.X.size(1))