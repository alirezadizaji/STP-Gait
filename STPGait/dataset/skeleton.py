from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

class SkeletonDataset(Dataset):
    def __init__(self, 
            X: np.ndarray,
            names: np.ndarray,
            Y: np.ndarray) -> None:

        super().__init__()
        self.X: torch.Tensor = torch.from_numpy(X).float()
        self.Y: torch.Tensor = torch.from_numpy(Y)
        self.names = names

        assert X.size(0) == names.size == Y.size(0), f"Mismatch number of samples between data ({X.size(0)}), names ({names.size}) and labels ({Y.size(0)})."
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        if not isinstance(index, list):
            index = [index]
        return self.X[index], self.Y[index], self.names[index]