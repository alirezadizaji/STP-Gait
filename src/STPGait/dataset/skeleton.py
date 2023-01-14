from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

class SkeletonDataset(Dataset):
    def __init__(self, 
            X: np.ndarray,
            names: np.ndarray,
            Y: np.ndarray,
            mask: np.ndarray) -> None:

        super().__init__()
        self.X: torch.Tensor = torch.from_numpy(X).float()      # N, T, V, C
        self.Y: torch.Tensor = torch.from_numpy(Y)              # N
        self.mask: torch.Tensor = torch.from_numpy(mask)        # N, T
        self.names = names                                      # N

        assert self.X.size(0) == self.names.size == self.Y.size(0) == self.mask.size(0), f"Mismatch number of samples between data ({X.size(0)}), names ({names.size}), labels ({Y.size(0)}), and masks ({self.mask.size(0)})."
        
    def __len__(self):
        return self.X.size(0)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        return self.X[index], self.Y[index], self.mask[index], index